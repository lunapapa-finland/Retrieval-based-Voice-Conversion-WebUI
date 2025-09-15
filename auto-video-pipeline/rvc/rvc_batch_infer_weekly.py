#!/usr/bin/env python3
"""
RVC batch inference for WEEKLY content (no language layer).
Run from project root:
  python ./auto-video-pipeline/rvc/rvc_batch_infer_weekly.py

Input : data/step2_wav-tts/<week>/*.wav        (week dirs sit directly under step2_wav-tts)
Output: data/step3_wav-converted/<week>/*_converted.wav
Skip  : data/step3_wav-converted/rvced.log
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- env loader ----------
def load_env(env_path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
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
    best: Optional[Tuple[int, Path]] = None
    for p in folder.glob(f"{prefix}*.pth" if prefix else "*.pth"):
        m = re.search(r"_e(\d+)", p.name)
        score = int(m.group(1)) if m else -1
        if best is None or score > best[0]:
            best = (score, p)
    return best[1] if best else None

def main() -> int:
    script_path = Path(__file__).resolve()
    base_root = script_path.parents[1]  # auto-video-pipeline/
    cfg = load_env(base_root / "config" / "rvc_config.env")

    step2_root = (base_root / cfg_get(cfg, "RVC_INPUT_ROOT", "data/step2_wav-tts")).resolve()
    step3_root = (base_root / cfg_get(cfg, "RVC_OUTPUT_ROOT", "data/step3_wav-converted")).resolve()
    log_path   = (base_root / cfg_get(cfg, "RVC_LOG", "data/step3_wav-converted/rvced.log")).resolve()

    # infer_cli + checkpoint
    infer_cli_cfg = cfg_get(cfg, "RVC_INFER_CLI", "rvc/infer_cli.py")
    infer_cli  = (base_root / infer_cli_cfg).resolve() if not infer_cli_cfg.startswith("/") else Path(infer_cli_cfg)
    if not infer_cli.exists():
        fallback = base_root / "rvc" / "infer_cli.py"
        if fallback.exists():
            infer_cli = fallback
        else:
            print("ERR: infer_cli.py not found.")
            return 1

    ckpt_dir_cfg = cfg_get(cfg, "RVC_CHECKPOINT_DIR", "rvc/CHECKPOINT")
    ckpt_dir = (base_root / ckpt_dir_cfg).resolve() if not ckpt_dir_cfg.startswith("/") else Path(ckpt_dir_cfg)
    if not ckpt_dir.exists():
        ckpt_dir = base_root / "rvc" / "CHECKPOINT"

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

    # load processed set
    processed: set[str] = set()
    if log_path.exists():
        processed = {s.strip() for s in log_path.read_text(encoding="utf-8").splitlines() if s.strip()}

    # discover week folders directly under step2_root (exclude 'addon')
    week_glob = cfg_get(cfg, "WEEK_GLOB", "*????w??")
    week_regex = cfg_get(cfg, "WEEK_REGEX", "")
    week_pat = re.compile(week_regex) if week_regex else None

    week_dirs = sorted(d for d in step2_root.glob(week_glob) if d.is_dir() and d.name != "addon")
    if not week_dirs and week_pat:
        week_dirs = sorted(d for d in step2_root.iterdir() if d.is_dir() and d.name != "addon" and week_pat.match(d.name))

    if not week_dirs:
        print(f"skip: no week folders under {step2_root}")
        return 0

    total = 0
    ckpt_base = Path(ckpt_name).stem

    for wdir in week_dirs:
        week = wdir.name
        out_rel = Path(cfg_get(cfg, "RVC_OUTPUT_ROOT", "data/step3_wav-converted")) / week
        if str(out_rel).replace("\\", "/") in processed:
            print(f"skip({week}): already in log")
            continue

        wavs = gather_files(wdir, wav_globs)
        if not wavs:
            print(f"skip({week}): no wavs")
            continue

        out_abs = step3_root / week
        out_abs.mkdir(parents=True, exist_ok=True)
        print(f"-- {week}: {len(wavs)} wav(s) --")

        wrote = False
        for w in wavs:
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
                f.write(str(out_rel).replace("\\", "/") + "\n")

    print(f"\n[weekly] total converted: {total}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
