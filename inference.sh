#!/bin/sh

# Exit on error
set -e

# Config
EXP="Test"                              # Experiment name
INPUT_DIR="data/voicetobechanged"       # Input folder with .wav files
RESULT_DIR="results/${EXP}"             # Output folder
F0_METHOD="rmvpe"                       # Pitch extraction method
RMS_MIX_RATE="0.25"                     # RMS mix rate
PROTECT="0.33"                          # Protect unvoiced consonants
FILTER_RADIUS="3"                       # Pitch filter radius

# Error handling
die() { echo "ERROR: $*" >&2; exit 1; }

# Dynamically find the checkpoint with the highest epoch starting with EXP
CHECKPOINT=""
if ls ./assets/weights/"${EXP}"*.pth >/dev/null 2>&1; then
  CHECKPOINT=$(ls ./assets/weights/"${EXP}"*.pth | while read -r file; do
    basename=$(basename "$file")
    epoch=$(echo "$basename" | grep -o 'e[0-9]\+' | sed 's/e//')
    echo "$epoch $basename"
  done | sort -nr | head -n1 | cut -d' ' -f2)
fi
[ -n "$CHECKPOINT" ] || die "No checkpoint found in ./assets/weights/ starting with $EXP"

# Set macOS-specific environment variables
if [ "$(uname)" = "Darwin" ]; then
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  export PYTORCH_MPS_FALLBACK_DEVICE=cpu
  python -c "import multiprocessing; multiprocessing.set_start_method('spawn', force=True)" 2>/dev/null || die "Failed to set multiprocessing spawn"
elif [ "$(uname)" != "Linux" ]; then
  die "Unsupported operating system"
fi

# Check dependencies
[ -x "$(command -v ffmpeg)" ] || die "ffmpeg not found. Install with: brew install ffmpeg"
[ -x "$(command -v python)" ] && python --version | grep -q "3.8" || die "Python 3.8 required"

# Check input folder
[ -d "$INPUT_DIR" ] || die "Input folder not found: $INPUT_DIR"
mkdir -p "$RESULT_DIR"

# Print config
echo "Using experiment:    $EXP"
echo "Checkpoint:         $CHECKPOINT"
echo "Input folder:       $INPUT_DIR"
echo "Output folder:      $RESULT_DIR"
echo "f0method:           $F0_METHOD"
echo "rms_mix_rate:       $RMS_MIX_RATE"
echo "protect:            $PROTECT"
echo "filter_radius:      $FILTER_RADIUS"
echo

# Gather .wav files (case-insensitive)
audio_files=""
for file in "$INPUT_DIR"/*.wav "$INPUT_DIR"/*.WAV; do
  [ -f "$file" ] && audio_files="$audio_files $file"
done
[ -n "$audio_files" ] || die "No .wav files found in $INPUT_DIR"

# Count total files
total=0
for file in $audio_files; do
  total=$((total + 1))
done

# Run inference for each file
i=0
for INPUT in $audio_files; do
  i=$((i + 1))
  base=$(basename "$INPUT")
  name="${base%.*}"
  OUTPUT="$RESULT_DIR/${name}_converted.wav"

  echo "[$i/$total] Converting: $INPUT"
  echo " -> $OUTPUT"

  python tools/infer_cli.py \
    --model_name "$CHECKPOINT" \
    --input_path "$INPUT" \
    --opt_path "$OUTPUT" \
    --f0method "$F0_METHOD" \
    --rms_mix_rate "$RMS_MIX_RATE" \
    --protect "$PROTECT" \
    --filter_radius "$FILTER_RADIUS" || die "Inference failed for $INPUT"

  echo
done

echo "All done. Files saved under: $RESULT_DIR "