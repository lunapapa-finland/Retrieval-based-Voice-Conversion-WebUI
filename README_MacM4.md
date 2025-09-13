# Retrieval-based Voice Conversion WebUI — Mac (M1/M2/M3/M4)

Setup instructions for running RVC WebUI on Apple Silicon.

---

## 1. Prerequisites

```bash
brew install ffmpeg aria2
xcode-select --install
```

---

## 2. Conda environment

Python 3.10 or 3.11 is recommended:

```bash
conda create -n rvc310 python=3.10 -y
conda activate rvc310
```

---

## 3. Python dependencies

```bash
pip install -r requirements.txt
```

The requirements pin `faiss-cpu==1.7.0` for macOS stability.
If FAISS crashes, rebuild from source:

```bash
pip uninstall -y faiss-cpu
pip install --no-binary=faiss-cpu faiss-cpu==1.7.0
```

---

## 4. Run WebUI

```bash
chmod +x run.sh
./run.sh
```

This script sets macOS environment variables, checks dependencies, downloads models if missing, and launches the UI on [http://0.0.0.0:7865](http://0.0.0.0:7865).

---

## 5. Workflow

1. **Data processing**

   * Converts audio into `0_gt_wavs/` and `1_16k_wavs/`.

2. **Feature extraction**

   * Uses the GPU (MPS) to run the content encoder (e.g. HuBERT/ContentVec).
   * Outputs `3_feature768/` (or `3_feature256/`).

3. **Pitch extraction**

   * Uses CPU-based methods (`rmvpe`, `dio`, `harvest`) or optionally `crepe`.
   * Outputs raw F0 (`2a_f0/`) and smoothed F0 (`2b-f0nsf/`).

4. **Training**

   * Runs the trainer with features + F0.
   * Checkpoints appear in `logs/<experiment>/`.

5. **Train feature index (optional)**

   * Builds a FAISS index on the content features (`3_feature***`).
   * Improves inference clarity and retrieval speed but is not required.

---

## 6. Apple Silicon notes

* MPS (Apple GPU) is used automatically; GPU index setting is ignored.
* Pitch extraction usually stays on CPU; feature extraction can use MPS.
* Training runs slower than on CUDA but is functional.

