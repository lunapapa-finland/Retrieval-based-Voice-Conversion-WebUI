#!/usr/bin/env python3
"""
RVC batch inference for WEEKLY content (env-only, Piper-style traversal).

Run from project root:
  python ./auto-video-pipeline/rvc/rvc_batch_infer_weekly.py

Reads env: config/rvc_config.env
Input : <RVC_INPUT_ROOT>/<lang>/<week>/*.wav    (excludes 'addon')
Output: <RVC_OUTPUT_ROOT>/<lang>/<week>/*_converted.wav
Skip  : <RVC_LOG> (records processed output dirs)
"""

import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- env loader ----------
def _clean(v: str) -> str:
    return v.strip()

def load_env(env_path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = _clean(v)
    return cfg

def cfg_get(cfg: Dict[str, str], k: str, default: str = "") -> str:
    return cfg.get(k, default).strip() if cfg.get(k) is not None else default

def gather_files(folder: Path, patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(folder.glob(pat))
    return sorted(set(out))

def pick_latest_ckpt(folder: Path, prefix: str = "") -> Optional[Path]:
    if not folder.exists():
        return None
    best = None
    for p in folder.glob(f"{prefix}*.pth" if prefix else "*.pth"):
        m = re.search(r"_e(\d+)", p.name)
        score = int(m.group(1)) if m else -1
        if best is None or score > best[0]:
            best = (score, p)
    return best[1] if best else None

def main() -> int:
    script_path = Path(__file__).resolve()
    # Mirror Piper: project base is .../auto-video-pipeline
    base_root = script_path.parents[1]  # auto-video-pipeline/
    cfg_path  = base_root / "config" / "rvc_config.env"
    cfg = load_env(cfg_path)

    # Resolve I/O under base_root (just like Piper)
    step2_root = (base_root / cfg_get(cfg, "RVC_INPUT_ROOT", "data/step2_wav-tts")).resolve()
    step3_root = (base_root / cfg_get(cfg, "RVC_OUTPUT_ROOT", "data/step3_wav-converted")).resolve()
    log_path   = (base_root / cfg_get(cfg, "RVC_LOG", "data/step3_wav-converted/rvced.log")).resolve()

    # infer_cli path relative to base_root first (then fallback)
    infer_cli_cfg = cfg_get(cfg, "RVC_INFER_CLI", "rvc/infer_cli.py")
    infer_cli  = (base_root / infer_cli_cfg).resolve() if not infer_cli_cfg.startswith("/") else Path(infer_cli_cfg)
    if not infer_cli.exists():
        # fallback to sibling rvc/ at base_root
        fallback = base_root / "rvc" / "infer_cli.py"
        if fallback.exists():
            infer_cli = fallback
        else:
            print("ERR: infer_cli.py not found at configured locations.")
            return 1

    # checkpoint dir under base_root first (then fallback)
    ckpt_dir_cfg = cfg_get(cfg, "RVC_CHECKPOINT_DIR", "rvc/CHECKPOINT")
    ckpt_dir = (base_root / ckpt_dir_cfg).resolve() if not ckpt_dir_cfg.startswith("/") else Path(ckpt_dir_cfg)
    if not ckpt_dir.exists():
        alt = base_root / "rvc" / "CHECKPOINT"
        if alt.exists():
            ckpt_dir = alt

    ckpt_name = cfg_get(cfg, "RVC_CHECKPOINT")
    if not ckpt_name:
        prefix = cfg_get(cfg, "RVC_CKPT_PREFIX", "")
        ckpt = pick_latest_ckpt(ckpt_dir, prefix=prefix) or pick_latest_ckpt(ckpt_dir)
        if not ckpt:
            print(f"ERR: no checkpoint found in {ckpt_dir}")
            return 1
        ckpt_name = ckpt.name

    rvc_py = cfg_get(cfg, "RVC_PYTHON_CMD") or sys.executable

    wav_globs = [s.strip() for s in cfg_get(cfg, "RVC_WAV_GLOBS", "*.wav,*.WAV").split(",") if s.strip()]
    week_glob = cfg_get(cfg, "WEEK_GLOB", "*????w??")
    week_regex = cfg_get(cfg, "WEEK_REGEX", r"^\d{4}w\d{2}$")
    week_pat = re.compile(week_regex)

    weekly_langs = [s.strip() for s in cfg_get(cfg, "RVC_WEEKLY_LANGS", "").split(",") if s.strip()]

    # Skip log
    processed: set[str] = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                processed.add(s)

    # iterate language folders under step2 root, excluding 'addon'
    cand_langs = [p for p in step2_root.iterdir() if p.is_dir() and p.name != "addon"]
    if weekly_langs:
        cand_langs = [p for p in cand_langs if p.name in weekly_langs]

    total = 0
    for lang_dir in sorted(cand_langs):
        lang = lang_dir.name
        # discover weekly dirs (glob first, fallback to regex)
        week_dirs = sorted(d for d in lang_dir.glob(week_glob) if d.is_dir())
        if not week_dirs:
            week_dirs = sorted(d for d in lang_dir.iterdir() if d.is_dir() and week_pat.match(d.name))
        if not week_dirs:
            print(f"skip({lang}): no weekly folders")
            continue

        for wdir in week_dirs:
            week = wdir.name
            out_rel = Path(cfg_get(cfg, "RVC_OUTPUT_ROOT", "data/step3_wav-converted")) / lang / week
            key = str(out_rel).replace("\\", "/")
            if key in processed:
                print(f"skip({lang}/{week}): already in log")
                continue

            wavs = gather_files(wdir, wav_globs)
            if not wavs:
                print(f"skip({lang}/{week}): no wavs")
                continue

            out_abs = step3_root / lang / week
            out_abs.mkdir(parents=True, exist_ok=True)
            print(f"-- {lang}/{week}: {len(wavs)} wav(s) --")

            wrote = False
            for w in wavs:
                ckpt_base = Path(ckpt_name).stem
                out_wav = out_abs / f"{w.stem}_{ckpt_base}_converted.wav"
                cmd = [
                    rvc_py, str(infer_cli),
                    "--model_name", ckpt_name,
                    "--input_path", str(w),
                    "--opt_path", str(out_wav),
                    "--f0method", cfg_get(cfg, "F0_METHOD", "rmvpe"),
                    "--rms_mix_rate", cfg_get(cfg, "RMS_MIX_RATE", "0.25"),
                    "--protect", cfg_get(cfg, "PROTECT", "0.33"),
                    "--filter_radius", cfg_get(cfg, "FILTER_RADIUS", "3"),
                ]
                print(f"  infer: {w.relative_to(base_root)}")
                res = subprocess.run(cmd)
                if res.returncode != 0:
                    print(f"  ERR ({res.returncode}): {w.name}")
                    continue
                wrote = True
                total += 1
                print(f"  ✅ {out_wav.relative_to(base_root)}")

            if wrote:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(key + "\n")

    print(f"\n[weekly] total converted: {total}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
