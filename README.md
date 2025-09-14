# Fork of RVC_WEBUI — Mac (M1/M2/M3/M4) Compatible

This is a fork of the original **RVC_WEBUI**, modified to work smoothly on Apple Silicon Macs.

---

## 1. Prerequisites

Install required system tools:

```bash
brew install ffmpeg aria2 cmake
xcode-select --install
````

---

## 2. Conda Environment


```bash
conda create -n video-pipeline python=3.9 -y
conda activate video-pipeline
python -m pip install --upgrade "pip<24.1"
```

---

## 3. Python Dependencies

Install Python packages:

```bash
pip install -r requirements.txt
```

<!-- If `faiss-cpu` installation fails, adjust manually:

```bash
# In requirements.txt, set faiss-cpu==1.7.3
pip uninstall -y faiss-cpu
pip install --no-binary=faiss-cpu faiss-cpu==1.7.1.post3
``` -->

---

## 4. Run Environment Config

```bash
chmod +x run.sh
./run.sh
```

---

## 5. Prepare Training Data (`./data/myvoice`)

* Record **20–30 minutes** of clean speech with [Audacity](https://www.audacityteam.org/download/).

### Recommended Audacity Settings

* **Recording Device:** External microphone or good-quality headphones
* **Channels:** `1 (Mono)`
* **Project Sample Rate:** `48000 Hz`
* **Default Sample Rate:** `48000 Hz`
* **Default Sample Format:** `24-bit`

### Effects (apply before exporting)

* High-pass filter
* Noise reduction
* Trim silences

### Export Settings

* Format: `WAV (PCM)`
* Sample rate: `48000 Hz`
* Bit depth: `24-bit` (or `16-bit` if smaller files are preferred)
* Channel: `Mono`
* Single-file export is fine

---

## 6. Configuration if necessary


```bash
nano config.env
```

---

## 7. Training

```bash
chmod +x training.sh
./training.sh
```

---

## 8. Inference

```bash
chmod +x inference.sh
./inference.sh
```

---