#!/usr/bin/env python3
"""
tts_piper.py

For each language folder in data/script/<lang>, read every text file
(e.g., *.txt by default) and synthesize with every model in
tts_piper/models/<lang>. Outputs WAVs to data/wav-tts:

  data/wav-tts/<lang>/out_<filename>_<voicename>.wav

Also writes: data/wav-tts/generated_wavs.txt

Config is read from: config/tts-piper_config.env (relative to repo root)
or an overridden path via --config.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# ---------------- helpers ----------------
def load_env(env_path: Path) -> Dict[str, str]:
    """Very small .env reader: KEY=VALUE lines, '#' comments allowed."""
    cfg: Dict[str, str] = {}
    if not env_path.exists():
        return cfg
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg


def get_cfg(cfg: Dict[str, str], key: str, default: str) -> str:
    return cfg.get(key, default)


def split_cmd(s: str) -> List[str]:
    """Very light split; assume no fancy quoting needed."""
    return [part for part in s.split() if part]


def resolve_under(base: Path, maybe_rel: str) -> Path:
    """Resolve path that may be absolute or relative to 'base'."""
    p = Path(maybe_rel)
    return p if p.is_absolute() else (base / p)


def gather_files(folder: Path, exts: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(folder.glob(f"*{ext}"))
    # dedupe & sort
    return sorted({f.resolve() for f in files})


# --------------- core logic ---------------
def run_tts(
    repo_root: Path,
    config_path: Path,
    langs: Iterable[str] | None = None,
    dry_run: bool = False,
) -> List[Tuple[str, str, str, str]]:
    """
    Execute TTS generation.

    Returns: list of (lang, filename_stem, voice_basename, output_wav_path)
    """
    env = load_env(config_path)

    # Directories (defaults are RELATIVE TO REPO ROOT)
    scripts_root = resolve_under(repo_root, get_cfg(env, "SCRIPTS_DIR", "data/script"))
    models_root  = resolve_under(repo_root, get_cfg(env, "MODELS_DIR",  "tts_piper/models"))
    out_root     = resolve_under(repo_root, get_cfg(env, "OUTPUT_DIR",  "data/wav-tts"))
    out_root.mkdir(parents=True, exist_ok=True)

    # Piper invocation pieces
    # Default: current Python interpreter runs `-m piper`
    default_piper_cmd = f"{sys.executable} -m piper"
    piper_cmd_str     = get_cfg(env, "PIPER_CMD", default_piper_cmd).strip()
    extra_args_str    = get_cfg(env, "PIPER_EXTRA_ARGS", "").strip()

    # Tunables
    length_scale     = get_cfg(env, "LENGTH_SCALE", "0.9")
    sentence_silence = get_cfg(env, "SENTENCE_SILENCE", "0.4")

    # Text file extensions (comma-separated)
    text_exts = [e.strip() for e in get_cfg(env, "TEXT_EXTS", ".txt").split(",") if e.strip()]

    # Optional language filter
    lang_filter = set(langs) if langs else None

    # Announce config
    print("[cfg]")
    print(f"  REPO_ROOT      = {repo_root}")
    print(f"  CONFIG         = {config_path}")
    print(f"  SCRIPTS_DIR    = {scripts_root}")
    print(f"  MODELS_DIR     = {models_root}")
    print(f"  OUTPUT_DIR     = {out_root}")
    print(f"  PIPER_CMD      = {piper_cmd_str}")
    if extra_args_str:
        print(f"  PIPER_EXTRA    = {extra_args_str}")
    print(f"  LENGTH_SCALE   = {length_scale}")
    print(f"  SENT_SILENCE   = {sentence_silence}")
    print(f"  TEXT_EXTS      = {text_exts}")
    if lang_filter:
        print(f"  LANGS_FILTER   = {sorted(lang_filter)}")
    if dry_run:
        print(f"  DRY_RUN        = True")

    piper_cmd = split_cmd(piper_cmd_str)
    extra_args = split_cmd(extra_args_str) if extra_args_str else []

    generated: List[Tuple[str, str, str, str]] = []

    # Languages are subfolders under data/script/<lang>
    if not scripts_root.exists():
        print(f"ERR: scripts dir not found: {scripts_root}")
        return generated

    for lang_dir in sorted(p for p in scripts_root.iterdir() if p.is_dir()):
        lang = lang_dir.name
        if lang_filter and (lang not in lang_filter):
            continue

        # Gather text files for this language
        txt_files = gather_files(lang_dir, text_exts)
        if not txt_files:
            print(f"skip: no text files {text_exts} in {lang_dir}")
            continue

        # Gather models for this language
        mdir = models_root / lang
        if not mdir.exists():
            print(f"skip: no models for '{lang}' at {mdir}")
            continue

        voices: List[Tuple[Path, Path, str]] = []
        for onnx in sorted(mdir.glob("*.onnx")):
            base = onnx.stem
            cfg = onnx.with_suffix(".onnx.json")
            if cfg.exists():
                voices.append((onnx, cfg, base))
            else:
                print(f"warn: missing cfg for {onnx.name} (expected {cfg.name})")

        if not voices:
            print(f"skip: no valid models in {mdir}")
            continue

        # Ensure output subdir
        out_lang_dir = out_root / lang
        out_lang_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n== {lang} :: {len(txt_files)} texts × {len(voices)} models ==")

        for txt_file in txt_files:
            text = txt_file.read_text(encoding="utf-8").strip()
            if not text:
                print(f"skip: empty {txt_file}")
                continue
            fname = txt_file.stem

            for model_path, cfg_path, vbase in voices:
                out_wav = out_lang_dir / f"out_{fname}_{vbase}.wav"
                cmd = [
                    *piper_cmd,
                    "-m", str(model_path.resolve()),
                    "-c", str(cfg_path.resolve()),
                    "--length-scale", length_scale,
                    "--sentence-silence", sentence_silence,
                    "-f", str(out_wav),
                    *extra_args,
                ]
                print(f"TTS: {lang}/{txt_file.name} + {vbase} -> {out_wav.relative_to(repo_root)}")

                if dry_run:
                    generated.append((lang, fname, vbase, str(out_wav)))
                    continue

                res = subprocess.run(cmd, input=text.encode("utf-8"))
                if res.returncode != 0:
                    print(f"ERR: piper failed for {vbase} on {txt_file.name} (exit {res.returncode})")
                    continue

                generated.append((lang, fname, vbase, str(out_wav)))
                print(f"✅ wrote {out_wav}")

    # Write summary
    summary_path = out_root / "generated_wavs.txt"
    if not dry_run:
        with summary_path.open("w", encoding="utf-8") as f:
            for lang, fname, vbase, path in generated:
                f.write(f"{lang}\t{fname}\t{vbase}\t{path}\n")

    print("\n--- Summary ---")
    if generated:
        for lang, fname, vbase, path in generated:
            rel = Path(path)
            try:
                rel = rel.resolve().relative_to(repo_root)
            except Exception:
                pass
            print(f"{lang}/{fname}: {vbase} -> {rel}")
        if not dry_run:
            print(f"\nList saved to: {summary_path}")
    else:
        print("No audio generated.")

    return generated


# --------------- CLI ---------------
def main(argv: Sequence[str] | None = None) -> int:
    # repo_root = project root (one level up from this file: .../<repo>/tts_piper/tts_piper.py)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]  # <repo>

    default_cfg = repo_root / "config" / "tts-piper_config.env"

    ap = argparse.ArgumentParser(description="Batch TTS with Piper over language/model grids.")
    ap.add_argument("--config", type=str, default=str(default_cfg),
                    help="Path to tts-piper_config.env (default: config/tts-piper_config.env)")
    ap.add_argument("--lang", action="append", dest="langs",
                    help="Restrict to one or more languages (repeatable). Example: --lang en --lang fi")
    ap.add_argument("--dry-run", action="store_true", help="List actions without running Piper.")
    args = ap.parse_args(argv)

    try:
        run_tts(
            repo_root=repo_root,
            config_path=Path(args.config).resolve(),
            langs=args.langs,
            dry_run=args.dry_run,
        )
        return 0
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
