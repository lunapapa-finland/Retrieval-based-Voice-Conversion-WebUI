#!/usr/bin/env bash
set -euo pipefail

# --- macOS / Linux checks ---
if [[ "$(uname)" == "Darwin" ]]; then
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
elif [[ "$(uname)" != "Linux" ]]; then
  echo "Unsupported operating system."
  exit 1
fi

# --- Python version gate: allow 3.9–3.11 ---
pyok="$(
python - <<'EOF'
import sys
print("OK" if (3,8) <= sys.version_info[:2] <= (3,11) else "NO")
EOF
)"
if [[ "$pyok" != "OK" ]]; then
  echo "Please use Python 3.8–3.11 for best Mac M-series compatibility."
  exit 1
fi

# --- Dependencies present? ---
if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c not found. Install with: brew install aria2"
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install with: brew install ffmpeg"
  exit 1
fi

# --- Download models (idempotent) ---
echo "Downloading models..."
chmod +x tools/dlmodels.sh
./tools/dlmodels.sh

# --- Launch WebUI ---
echo "Environment configuration is complete."
