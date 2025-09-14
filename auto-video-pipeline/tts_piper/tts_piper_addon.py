#!/usr/bin/env python3
"""
Addon TTS renderer (no CLI args).
Run from project root:
  python ./tts_piper/tts_piper_addon.py
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


# ---------- main ----------
def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]          # project root
    cfg_path = repo_root / "config" / "tts-piper_config.env"
    cfg = load_env(cfg_path)

    scripts_dir = repo_root / cfg_get(cfg, "SCRIPTS_DIR", "data/step1_scripts")
    models_dir  = repo_root / cfg_get(cfg, "MODELS_DIR",  "tts_piper/models")
    out_root    = repo_root / cfg_get(cfg, "OUTPUT_DIR",  "data/step2_wav-tts")

    # unified incremental log
    log_rel  = cfg_get(cfg, "LOG_FILE", "data/step1_scripts/pipered.log")
    log_path = (repo_root / log_rel).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    out_root.mkdir(parents=True, exist_ok=True)

    # Piper cmd + params
    piper_cmd_raw = cfg.get("PIPER_CMD", "").strip()
    if not piper_cmd_raw:
        piper_cmd_raw = f"{sys.executable} -m piper"
    piper_cmd = shlex.split(piper_cmd_raw)

    extra_args   = shlex.split(cfg.get("PIPER_EXTRA_ARGS", ""))
    length_scale = cfg.get("LENGTH_SCALE", "0.9")
    sent_sil     = cfg.get("SENTENCE_SILENCE", "0.4")
    text_exts    = [e.strip() for e in cfg.get("TEXT_EXTS", ".txt").split(",") if e.strip()]

    # Load processed keys (relative output dirs recorded previously)
    processed: set[str] = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                processed.add(s)

    addon_root = scripts_dir / "addon"

    print("[addon]")
    print(f"  addon  : {addon_root}")
    print(f"  models : {models_dir}")
    print(f"  output : {out_root}")
    print(f"  log    : {log_path}")
    print(f"  piper  : {' '.join(piper_cmd) or '(empty)'}")
    if extra_args:
        print(f"  extra  : {' '.join(extra_args)}")
    print()

    if not addon_root.exists():
        print(f"skip: addon/ missing at {addon_root}")
        return 0

    total_written = 0
    generated: List[Tuple[str, str, Path]] = []  # (lang, voice, path)

    # each language under addon/
    for lang_dir in sorted(p for p in addon_root.iterdir() if p.is_dir()):
        lang = lang_dir.name

        # voices for lang
        vdir = models_dir / lang
        if not vdir.exists():
            print(f"skip(addon/{lang}): no models at {vdir}")
            continue
        voices: List[Tuple[Path, Path, str]] = []
        for onnx in sorted(vdir.glob("*.onnx")):
            base = onnx.stem
            cfg_json = onnx.with_suffix(".onnx.json")
            if cfg_json.exists():
                voices.append((onnx, cfg_json, base))
            else:
                print(f"warn({lang}): missing config for {onnx.name} (expected {cfg_json.name})")
        if not voices:
            print(f"skip(addon/{lang}): no valid voices")
            continue

        # out dir for addon/lang
        out_dir_rel = Path(cfg_get(cfg, "OUTPUT_DIR", "data/step2_wav-tts")) / "addon" / lang
        out_dir_abs = repo_root / out_dir_rel

        # skip if already logged
        if str(out_dir_rel).replace("\\", "/") in processed:
            print(f"skip(addon/{lang}): already in log")
            continue

        # text files
        txt_files = gather_files(lang_dir, text_exts)
        if not txt_files:
            print(f"skip(addon/{lang}): no text files")
            continue

        out_dir_abs.mkdir(parents=True, exist_ok=True)
        print(f"-- addon/{lang}: {len(txt_files)} files × {len(voices)} voices --")

        wrote_any = False
        for txt in txt_files:
            text = txt.read_text(encoding="utf-8").strip()
            if not text:
                print(f"  skip empty: {txt.name}")
                continue
            stem = txt.stem

            for model_path, cfg_json, vname in voices:
                out_wav = out_dir_abs / f"out_{stem}_{vname}.wav"
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
                print(f"  TTS: addon/{lang}/{txt.name} + {vname} -> {rel_out}")
                res = subprocess.run(cmd, input=text.encode("utf-8"))
                if res.returncode != 0:
                    print(f"  ERR: piper failed ({res.returncode}) for {vname} on {txt.name}")
                    continue

                total_written += 1
                wrote_any = True
                generated.append((lang, vname, out_wav))
                print(f"  ✅ wrote {rel_out}")

        # mark processed if anything written
        if wrote_any:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(str(out_dir_rel).replace("\\", "/") + "\n")

    print("\n--- Addon Summary ---")
    if generated:
        for lang, vname, path in generated:
            print(f"addon/{lang}: {vname} -> {path.relative_to(repo_root)}")
    else:
        print("No addon audio generated.")
    print(f"\nTotal files written: {total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
