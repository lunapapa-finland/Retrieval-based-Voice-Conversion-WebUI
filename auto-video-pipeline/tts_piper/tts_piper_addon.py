#!/usr/bin/env python3
"""
Addon TTS renderer (no language layer).
Run from project root:
  python ./auto-video-pipeline/tts_piper/tts_piper_addon.py

Inputs:
  SCRIPTS_DIR/addon/*.txt
Models:
  MODELS_DIR/*.onnx (+ .onnx.json)
Outputs:
  OUTPUT_DIR/addon/out_<section>_<voice>.wav
Skip log:
  LOG_FILE (from env)
"""

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------- tiny .env loader (strips inline comments & quotes) ----------
def _clean_val(v: str) -> str:
    v = v.split("#", 1)[0].strip()
    if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
        v = v[1:-1].strip()
    return v

def load_env(env_path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = _clean_val(v)
    return cfg

def cfg_get(cfg: Dict[str, str], key: str, default: str) -> str:
    return cfg.get(key, default)

def gather_files(folder: Path, exts: List[str]) -> List[Path]:
    out: List[Path] = []
    for ext in exts:
        out.extend(folder.glob(f"*{ext}"))
    return sorted(set(out))

def gather_voices(models_dir: Path) -> List[Tuple[Path, Path, str]]:
    voices: List[Tuple[Path, Path, str]] = []
    for onnx in sorted(models_dir.glob("*.onnx")):
        base = onnx.stem
        cfg_json = onnx.with_suffix(".onnx.json")
        if cfg_json.exists():
            voices.append((onnx, cfg_json, base))
    return voices

# ---------- main ----------
def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]          # auto-video-pipeline/
    cfg_path = repo_root / "config" / "tts-piper_config.env"
    cfg = load_env(cfg_path)

    # scripts_dir = repo_root / cfg_get(cfg, "SCRIPTS_DIR", "data/step1_scripts")
    addon_dir   = repo_root / cfg_get(cfg, "ADDON_DIR",   "data/step0_addon")
    models_dir  = repo_root / cfg_get(cfg, "MODELS_DIR",  "tts_piper/models")
    out_root    = repo_root / cfg_get(cfg, "OUTPUT_DIR",  "data/step2_wav-tts")

    log_rel  = cfg_get(cfg, "LOG_FILE_addon", "data/step2_wav-tts/pipered.log")
    log_path = (repo_root / log_rel).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    piper_cmd_raw = cfg.get("PIPER_CMD", "").strip() or f"{sys.executable} -m piper"
    piper_cmd     = shlex.split(piper_cmd_raw)
    extra_args    = shlex.split(cfg.get("PIPER_EXTRA_ARGS", ""))
    length_scale  = cfg.get("LENGTH_SCALE", "0.9")
    sent_sil      = cfg.get("SENTENCE_SILENCE", "0.4")
    text_exts     = [e.strip() for e in cfg.get("TEXT_EXTS", ".txt").split(",") if e.strip()]

    processed: set[str] = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                processed.add(s)

    addon_in  = addon_dir 
    addon_out = addon_dir / "tts_addon"

    print("[addon]")
    print(f"  scripts: {addon_in}")
    print(f"  models : {models_dir}")
    print(f"  output : {addon_out}")
    print(f"  log    : {log_path}")
    print(f"  piper  : {' '.join(piper_cmd)}")
    if extra_args:
        print(f"  extra  : {' '.join(extra_args)}")
    print()

    if not addon_in.exists():
        print(f"skip: addon/ missing at {addon_in}")
        return 0

    # Voices (no lang subfolder)
    voices = gather_voices(models_dir)
    if not voices:
        print(f"ERR: no valid voices found in {models_dir} (need .onnx + .onnx.json)")
        return 1

    # Skip if output dir already processed
    out_dir_rel = Path(cfg_get(cfg, "OUTPUT_DIR", "data/step2_wav-tts")) / "addon"
    key = str(out_dir_rel).replace("\\", "/")
    if key in processed:
        print("skip(addon): already in log")
        return 0

    txt_files = gather_files(addon_in, text_exts)
    if not txt_files:
        print("skip(addon): no text files")
        return 0

    addon_out.mkdir(parents=True, exist_ok=True)
    print(f"-- addon: {len(txt_files)} files × {len(voices)} voices --")

    total_written = 0
    for txt in txt_files:
        text = txt.read_text(encoding="utf-8").strip()
        if not text:
            print(f"  skip empty: {txt.name}")
            continue
        stem = txt.stem

        for model_path, cfg_json, vname in voices:
            out_wav = addon_out / f"out_{stem}_{vname}.wav"
            cmd = [
                *piper_cmd,
                "-m", str(model_path.resolve()),
                "-c", str(cfg_json.resolve()),
                "--length-scale", str(length_scale),
                "--sentence-silence", str(sent_sil),
                "-f", str(out_wav),
                *extra_args,
            ]
            rel_out = out_wav.relative_to(repo_root)
            print(f"  TTS: {txt.name} + {vname} -> {rel_out}")
            res = subprocess.run(cmd, input=text.encode("utf-8"))
            if res.returncode != 0:
                print(f"  ERR: piper failed ({res.returncode}) for {vname} on {txt.name}")
                continue
            total_written += 1
            print(f"  ✅ wrote {rel_out}")

    # mark processed
    with log_path.open("a", encoding="utf-8") as f:
        f.write(str(out_dir_rel).replace("\\", "/") + "\n")

    print(f"\nTotal files written: {total_written}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
