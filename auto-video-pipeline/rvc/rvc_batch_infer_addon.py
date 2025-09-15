#!/usr/bin/env python3
"""
RVC batch inference for ADDON content (no language layer).
Run from project root:
  python ./auto-video-pipeline/rvc/rvc_batch_infer_addon.py

Input : data/step2_wav-tts/addon/*.wav
Output: data/step3_wav-converted/addon/*_converted.wav
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

    step2_root = (base_root / cfg_get(cfg, "RVC_INPUT_ROOT_ADDON", "data/step2_wav-tts")).resolve()
    step3_root = (base_root / cfg_get(cfg, "RVC_OUTPUT_ROOT_ADDON", "data/step3_wav-converted")).resolve()
    log_path   = (base_root / cfg_get(cfg, "RVC_LOG_ADDON", "data/step0_addon/rvc/rvced.log")).resolve()

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

    addon_in  = step2_root 
    addon_out = step3_root
    if not addon_in.exists():
        print(f"skip: addon input missing at {addon_in}")
        return 0

    # skip if output dir already logged
    key = str(Path(cfg_get(cfg, "RVC_OUTPUT_ROOT_ADDON", "data/step3_wav-converted"))).replace("\\", "/")
    if key in processed:
        print("skip(addon): already in log")
        return 0
    
    wavs = gather_files(addon_in, wav_globs)
    if not wavs:
        print("skip(addon): no wavs")
        return 0

    addon_out.mkdir(parents=True, exist_ok=True)
    print(f"-- addon: {len(wavs)} wav(s) --")

    total = 0
    ckpt_base = Path(ckpt_name).stem
    for w in wavs:
        out_wav = addon_out / f"{w.stem}_{ckpt_base}_converted.wav"
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
        total += 1
        print(f"  ✅ {out_wav.relative_to(base_root)}")

    # mark processed
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(key + "\n")

    print(f"\n[addon] total converted: {total}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
