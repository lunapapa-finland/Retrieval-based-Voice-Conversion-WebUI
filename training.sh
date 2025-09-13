#!/usr/bin/env bash
set -euo pipefail

# ========= LOAD CONFIG =========
. ./config.env

# ========= ENV (macOS/MPS) =========
# Keep minimal, only on macOS. These help with MPS fallback but aren't strictly required on Linux.
if [[ "$(uname)" == "Darwin" ]]; then
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
elif [[ "$(uname)" != "Linux" ]]; then
  echo "ERROR: Unsupported operating system"
  exit 1
fi

# (Optional) Ensure Python 3.8+; if you rely on a virtualenv, you may remove this block.
python - <<'PY' || { echo "ERROR: Python 3.8+ required"; exit 1; }
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if (major > 3 or (major == 3 and minor >= 8)) else 1)
PY

# ========= AUTO-TUNE WORKERS (optional) =========
# If NPROC is 0, auto-detect logical CPUs.
if [[ "${NPROC}" == "0" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    NPROC="$(nproc)"
  elif [[ "$(uname)" == "Darwin" ]]; then
    NPROC="$(sysctl -n hw.ncpu)"
  else
    NPROC="4"
  fi
fi

# ========= VALIDATE PRETRAINED WEIGHTS =========
if [[ ! -f "${PRETRAIN_G}" ]]; then
  echo "ERROR: Generator weights not found: ${PRETRAIN_G}"
  exit 1
fi
if [[ ! -f "${PRETRAIN_D}" ]]; then
  echo "ERROR: Discriminator weights not found: ${PRETRAIN_D}"
  exit 1
fi

# ========= PATHS =========
LOG_DIR="./logs/${EXP}"
RESULT_DIR="./results/${EXP}"
WAV_16K_DIR="${LOG_DIR}/1_16k_wavs"
GT_WAV_DIR="${LOG_DIR}/0_gt_wavs"
FEAT_DIR="${LOG_DIR}/3_feature${FEAT_DIM}"
F0_DIR_A="${LOG_DIR}/2a_f0"
F0_DIR_NSF="${LOG_DIR}/2b-f0nsf"
FILELIST="${LOG_DIR}/filelist.txt"
CFG_JSON="${LOG_DIR}/config.json"

echo "==> Preparing directories"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${DATA_DIR}"

# ========= 1) PREPROCESS =========
echo "==> Preprocess"
python infer/modules/train/preprocess.py   "${DATA_DIR}"   "${SAMPLE_RATE_HZ}"   "${NPROC}"   "${LOG_DIR}"   False   3.0

# ========= 2) FEATURE EXTRACTION (HuBERT) =========
# Device comes from config.env (DEVICE=mps|cuda|cpu)
echo "==> Feature extraction (HuBERT -> 3_feature${FEAT_DIM}) on ${DEVICE}"
python infer/modules/train/extract_feature_print.py   "${DEVICE}"   1   0   "${LOG_DIR}"   "${VERSION}"   False

# ========= 3) PITCH (F0) EXTRACTION =========
echo "==> Pitch (F0) extraction with RMVPE"
python infer/modules/train/extract/extract_f0_print.py   "${LOG_DIR}"   "${NPROC}"   rmvpe

# ========= 4) (OPTIONAL) FAISS INDEX =========
if [[ "${TRAIN_INDEX}" == "1" ]]; then
  echo "==> Train FAISS index (saved to ${LOG_DIR})"
  python tools/train_index.py     --exp "${EXP}"     --feat-dim "${FEAT_DIM}"     --kmeans "${KMEANS}"
fi

# ========= 5) SAFELY GENERATE filelist.txt =========
# Format per line:
# <0_gt_wavs/XYZ.wav>|<3_feature*/XYZ.npy>|<2a_f0/XYZ.wav.npy>|<2b-f0nsf/XYZ.wav.npy>|0
echo "==> Generate filelist.txt"

# Ensure required dirs exist
for d in "${GT_WAV_DIR}" "${FEAT_DIR}" "${F0_DIR_A}" "${F0_DIR_NSF}"; do
  if [[ ! -d "$d" ]]; then
    echo "ERROR: expected directory not found: $d"
    exit 1
  fi
done

# Build filelist atomically then move into place
tmp_filelist="${FILELIST}.tmp"
trap 'rm -f "${tmp_filelist}"' EXIT
: > "${tmp_filelist}"

# Use NUL-delimited find to handle spaces safely; stable sort.
find "${GT_WAV_DIR}" -maxdepth 1 -type f -name "*.wav" -print0 2>/dev/null   | LC_ALL=C sort -z   | while IFS= read -r -d '' wav; do
      base="$(basename "$wav")"             # e.g. 12_3.wav
      stem="${base%.wav}"                   # e.g. 12_3
      feat="${FEAT_DIR}/${stem}.npy"
      f0a="${F0_DIR_A}/${stem}.wav.npy"
      f0nsf="${F0_DIR_NSF}/${stem}.wav.npy"

      # Only include rows where every file exists
      if [[ -f "$feat" && -f "$f0a" && -f "$f0nsf" ]]; then
        printf "%s|%s|%s|%s|0\n" "$wav" "$feat" "$f0a" "$f0nsf" >> "${tmp_filelist}"
      else
        echo "WARN: skip ${stem} (missing one of feature/F0 files)" >&2
      fi
    done

# Strip CRs and blank lines, then install
tr -d '\r' < "${tmp_filelist}" | awk 'NF>0' > "${FILELIST}"

# Validate: every line must have exactly 5 fields
bad_lines=$(awk -F'|' 'NF!=5{print NR}' "${FILELIST}" || true)
if [[ -n "${bad_lines}" ]]; then
  echo "ERROR: filelist has malformed lines (not 5 fields) on rows: ${bad_lines}"
  exit 1
fi

count=$(wc -l < "${FILELIST}" | tr -d ' ')
echo "filelist.txt -> ${FILELIST} (${count} items)"
if [[ "${count}" -eq 0 ]]; then
  echo "ERROR: filelist is empty; nothing to train on."
  exit 1
fi

# ========= 6) WRITE config.json LIKE WEBUI =========
echo "==> Generate config.json -> ${CFG_JSON}"
cat > "${CFG_JSON}" <<EOF
{
  "data": {
    "filter_length": 2048,
    "hop_length": 400,
    "max_wav_value": 32768.0,
    "mel_fmax": null,
    "mel_fmin": 0.0,
    "n_mel_channels": 125,
    "sampling_rate": ${SAMPLE_RATE_HZ},
    "win_length": 2048,
    "training_files": "${FILELIST}"
  },
  "model": {
    "filter_channels": 768,
    "gin_channels": 256,
    "hidden_channels": 192,
    "inter_channels": 192,
    "kernel_size": 3,
    "n_heads": 2,
    "n_layers": 6,
    "p_dropout": 0,
    "resblock": "1",
    "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
    "resblock_kernel_sizes": [3,7,11],
    "spk_embed_dim": 109,
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16,16,4,4],
    "upsample_rates": [10,10,2,2],
    "use_spectral_norm": false
  },
  "train": {
    "batch_size": ${BATCH_SIZE},
    "betas": [0.8, 0.99],
    "c_kl": 1.0,
    "c_mel": 45,
    "epochs": 20000,
    "eps": 1e-09,
    "fp16_run": false,
    "init_lr_ratio": 1,
    "learning_rate": 0.0001,
    "log_interval": 200,
    "lr_decay": 0.999875,
    "seed": 1234,
    "segment_size": 12800,
    "warmup_epochs": 0
  }
}
EOF
echo "Wrote ${CFG_JSON}"

# ========= 7) TRAIN =========
echo "==> Train model"
python infer/modules/train/train.py   -e "${EXP}"   -sr "${SR}"   -f0 1   -bs "${BATCH_SIZE}"   -te "${TOTAL_EPOCH}"   -se "${SAVE_EVERY}"   -pg "${PRETRAIN_G}"   -pd "${PRETRAIN_D}"   -l 0   -c 0   -sw 1   -v "${VERSION}"
