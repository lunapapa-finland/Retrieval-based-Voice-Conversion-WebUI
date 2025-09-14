#!/usr/bin/env python3
"""
Batch RVC inference that mirrors your step2->step3 structure and skip-log behavior.

Run from project root:
  python ./auto-video-pipeline/rvc/rvc_batch_infer.py

- Reads wavs from: data/step2_wav-tts/<addon|lang>/<week?>/*.wav
- Writes to:       data/step3_wav-converted/<addon|lang>/<week?>/*.wav
- Skip-log:        data/step2_wav-tts/rvced.log   (records processed output dirs)
- Checkpoint:      auto-video-pipeline/rvc/CHECKPOINT/<*.pth>
- Calls:           python ./auto-video-pipeline/rvc/infer_cli.py --model_name <CKPT_NAME> ...
                   (NOTE: infer_cli prepends its own rvc/CHECKPOINT/, so we pass only the filename)
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- tiny .env loader ----------
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

def cfg_get(cfg: Dict[str, str], key: str, default: str) -> str:
    return cfg.get(key, default)

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
        # Expect names like EXP_e130_s58890.pth -> grab the e### number
        m = re.search(r"_e(\d+)", p.name)
        score = int(m.group(1)) if m else -1
        if best is None or score > best[0]:
            best = (score, p)
    return best[1] if best else None

def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]  # project root

    # default locations
    step2_dir = repo_root / "data" / "step2_wav-tts"
    step3_dir = repo_root / "data" / "step3_wav-converted"
    log_path  = step2_dir / "rvced.log"

    # load optional env (reuse tts env keys if you like, not required here)
    cfg = {}
    # read checkpoint folder
    ckpt_dir = repo_root / "auto-video-pipeline" / "rvc" / "CHECKPOINT"
    if not ckpt_dir.exists():
        # fallback to project rvc/CHECKPOINT
        ckpt_dir = repo_root / "rvc" / "CHECKPOINT"

    # choose checkpoint: prefer EXP from env var, otherwise latest by epoch
    exp = os.environ.get("EXP", "")  # if you set EXP=Formal etc. will bias the pick
    ckpt_path = pick_latest_ckpt(ckpt_dir, prefix=exp) or pick_latest_ckpt(ckpt_dir)
    if not ckpt_path:
        print(f"ERR: no checkpoint found in {ckpt_dir}")
        return 1

    # IMPORTANT: infer_cli joins its own rvc/CHECKPOINT/ + args.model_name.
    # So we must pass ONLY THE BASENAME (not an absolute or relative path).
    ckpt_name = ckpt_path.name

    # infer_cli path
    infer_cli = repo_root / "auto-video-pipeline" / "rvc" / "infer_cli.py"
    if not infer_cli.exists():
        infer_cli = repo_root / "rvc" / "infer_cli.py"
    if not infer_cli.exists():
        print(f"ERR: infer_cli.py not found")
        return 1

    # python to run
    py = sys.executable

    week_regex = r"^\d{4}w\d{2}$"
    week_pat   = re.compile(week_regex)

    # load processed (output-dir keys as relative paths)
    processed: set[str] = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                processed.add(s)

    print("[rvc-infer]")
    print(f"  input  : {step2_dir}")
    print(f"  output : {step3_dir}")
    print(f"  log    : {log_path}")
    print(f"  weekrx : {week_regex}")
    print(f"  ckpt   : {ckpt_path}")
    print(f"  py     : {py}")
    print(f"  cli    : {infer_cli}")
    print()

    if not step2_dir.exists():
        print(f"ERR: input dir missing: {step2_dir}")
        return 1
    step3_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0

    def run_folder(src_dir: Path, dst_dir_rel: Path) -> int:
        """convert all wavs in src_dir -> corresponding files in step3/dst_dir_rel"""
        nonlocal total_written
        dst_dir_abs = step3_dir / dst_dir_rel
        key = str(Path("data") / "step3_wav-converted" / dst_dir_rel).replace("\\", "/")

        # skip if already processed
        if key in processed:
            print(f"skip({dst_dir_rel}): already in log")
            return 0

        wavs = gather_files(src_dir, ["*.wav", "*.WAV"])
        if not wavs:
            # nothing to do
            return 0

        dst_dir_abs.mkdir(parents=True, exist_ok=True)
        print(f"-- {dst_dir_rel}: {len(wavs)} wav(s) --")

        wrote_any = False
        for w in wavs:
            # build output filename: append _<ckpt>_converted.wav (ckpt base)
            ckpt_base = ckpt_name.rsplit(".", 1)[0]
            out_name = f"{w.stem}_{ckpt_base}_converted.wav"
            out_path = dst_dir_abs / out_name

            print(f"    infer: {w.relative_to(repo_root)}")

            cmd = [
                py, str(infer_cli),
                "--model_name", ckpt_name,   # <— pass BASENAME only
                "--input_path", str(w),
                "--opt_path", str(out_path),
                "--f0method", os.environ.get("F0_METHOD", "rmvpe"),
                "--rms_mix_rate", os.environ.get("RMS_MIX_RATE", "0.25"),
                "--protect", os.environ.get("PROTECT", "0.33"),
                "--filter_radius", os.environ.get("FILTER_RADIUS", "3"),
            ]
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"    ERR: infer_cli failed ({res.returncode}): {w.name}")
                continue

            wrote_any = True
            total_written += 1

        if wrote_any:
            # mark processed
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(key + "\n")
        return 0

    # 1) addon/<lang>
    addon_dir = step2_dir / "addon"
    if addon_dir.exists():
        for lang_dir in sorted(p for p in addon_dir.iterdir() if p.is_dir()):
            src = lang_dir
            dst_rel = Path("addon") / lang_dir.name
            run_folder(src, dst_rel)

    # 2) weekly: <lang>/<YYYYwWW>/ (sections already flattened to wavs by TTS)
    for lang_dir in sorted(p for p in step2_dir.iterdir() if p.is_dir() and p.name != "addon"):
        for wdir in sorted(d for d in lang_dir.iterdir() if d.is_dir() and week_pat.match(d.name)):
            src = wdir
            dst_rel = Path(lang_dir.name) / wdir.name
            run_folder(src, dst_rel)

    print("\n--- RVC Summary ---")
    print(f"Total files written: {total_written}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
