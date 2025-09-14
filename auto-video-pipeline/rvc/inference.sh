#!/bin/sh

# Exit on error
set -e

# Config (loaded from shared file to keep definitions split and well maintained)
. auto-video-pipeline/config/rvc_config.env

# Error handling
die() { echo "ERROR: $*" >&2; exit 1; }


# Set macOS-specific environment variables
# (These are helpful for MPS fallback, but not strictly required every run. Keep minimal.)
if [ "$(uname)" = "Darwin" ]; then
  export PYTORCH_ENABLE_MPS_FALLBACK=1
elif [ "$(uname)" != "Linux" ]; then
  die "Unsupported operating system"
fi

# Check dependencies
# Provide cross-platform install hint for ffmpeg
[ -x "$(command -v ffmpeg)" ] || die "ffmpeg not found. Install it (macOS: brew install ffmpeg, Ubuntu/Debian: sudo apt-get install ffmpeg, Arch: sudo pacman -S ffmpeg)"

# Require Python 3.8+ (was exact 3.8 before). If you rely on a venv, you may remove this block.
python - <<'PY' || exit 1
import sys
major, minor = sys.version_info[:2]
sys.exit(0 if (major > 3 or (major == 3 and minor >= 8)) else 1)
PY

# Check input folder
[ -d "$INPUT_DIR" ] || die "Input folder not found: $INPUT_DIR"
mkdir -p "$RESULT_DIR"

# Print config
echo "Checkpoint:         $CHECKPOINT"
echo "Input folder:       $INPUT_DIR"
echo "Output folder:      $RESULT_DIR"
echo "f0method:           $F0_METHOD"
echo "rms_mix_rate:       $RMS_MIX_RATE"
echo "protect:            $PROTECT"
echo "filter_radius:      $FILTER_RADIUS"
echo

# Gather .wav files (case-insensitive). Using find for robustness.
audio_files=$(find "$INPUT_DIR" -maxdepth 1 -type f -iname '*.wav' -print)
[ -n "$audio_files" ] || die "No .wav files found in $INPUT_DIR"

# Count total files (short form)
set -- $audio_files
total=$#

# Run inference for each file
i=0
for INPUT in $audio_files; do
  i=$((i + 1))
  base=$(basename "$INPUT")
  name="${base%.*}"

  # Include checkpoint identifier in the output filename (strip directory + .pth)
  ckpt_base=$(basename "$CHECKPOINT" .pth)
  OUTPUT="$RESULT_DIR/${name}_${ckpt_base}_converted.wav"

  echo "[$i/$total] Converting: $INPUT"
  echo " -> $OUTPUT"

  python ./auto-video-pipeline/rvc/infer_cli.py     --model_name "$CHECKPOINT"     --input_path "$INPUT"     --opt_path "$OUTPUT"     --f0method "$F0_METHOD"     --rms_mix_rate "$RMS_MIX_RATE"     --protect "$PROTECT"     --filter_radius "$FILTER_RADIUS" || die "Inference failed for $INPUT"

  echo
done

echo "All done. Files saved under: $RESULT_DIR "
