#!/bin/sh

# Set macOS-specific environment variables
if [ "$(uname)" = "Darwin" ]; then
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
elif [ "$(uname)" != "Linux" ]; then
  echo "Unsupported operating system."
  exit 1
fi

# Allow Python 3.9–3.11 (recommended on Apple Silicon)
if ! python - <<'EOF'
import sys
ok = (3,9) <= sys.version_info[:2] <= (3,11)
print("OK" if ok else "NO")
EOF
 | grep -q OK; then
  echo "Please use Python 3.9–3.11 for best Mac M-series compatibility."
  exit 1
fi

# Check if aria2c is installed
if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c not found. Please install aria2 using Homebrew: 'brew install aria2'"
  exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Please install ffmpeg using Homebrew: 'brew install ffmpeg'"
  exit 1
fi

# Download models
echo "Downloading models..."
chmod +x tools/dlmodels.sh
./tools/dlmodels.sh

if [ $? -ne 0 ]; then
  echo "Model download failed."
  exit 1
fi

# Run the main script
python infer-web.py --pycmd python