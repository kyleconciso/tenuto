# Tenuto

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyleconciso/tenuto/blob/main/notebooks/colab_training.ipynb)

Predicts human performance nuance (rubato, micro-timing, velocity, articulation, sustain pedal) from sheet music or MIDI scores using a non-autoregressive PyTorch model.

Instead of predicting events note-by-note, Tenuto runs a single forward pass over the score grid.

---

## 🚀 The Easiest Way to Train: Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyleconciso/tenuto/blob/main/notebooks/colab_training.ipynb)

We highly recommend using our **Google Colab Notebook**. It is a fully configured, one-click pipeline that will:
1. Automatically download the **ASAP** and **PianoCoRe** datasets (over 130,000+ score-performance pairs!)
2. Unpack HuggingFace `.parquet` database files natively.
3. Preprocess and chunk the data into highly expressive 256-note tensors.
4. Train the 5-million parameter Transformer model using a T4 GPU.
5. Generate an A/B audio comparison between a mechanical score, a real human, and the Tenuto AI.

---

## Local Quick Start

If you want to run it locally, follow these steps:

### 1. Setup
```bash
git clone https://github.com/kyleconciso/tenuto.git
cd tenuto
pip install -r requirements.txt
```

### 2. Download Datasets
Automatically fetch the ASAP and PianoCoRe datasets (requires `huggingface_hub`):
```bash
PYTHONPATH=. python -m src.download_dataset --dataset combined
```

### 3. Preprocess
Extract 40D features and ground-truth alignments from the raw XML/MIDI and Parquet files:
```bash
PYTHONPATH=. python -m src.preprocess --data_dir ./data --processed_dir ./data/processed
```

### 4. Train
Train the full Transformer backbone:
```bash
PYTHONPATH=. python -m src.train --model_type transformer --in_features 40 --epochs 20 --batch_size 16 --lr 1e-4
```

### 5. Predict (Inference)
Feed a new flat score into the model to generate expressive MIDI:
```bash
PYTHONPATH=. python -m src.infer --score data/asap/Balakirev/Islamey/xml_score.musicxml --checkpoint checkpoints/best_transformer_model.pth --model_type transformer --output_midi output_expressive.mid
```

---

## Project Layout

```
tenuto/
├── notebooks/
│   └── colab_training.ipynb  <-- START HERE
├── src/
│   ├── download_dataset.py
│   ├── dataset.py
│   ├── model.py
│   ├── preprocess.py
│   ├── features.py
│   ├── alignment.py
│   ├── audio.py
│   ├── train.py
│   └── infer.py
├── data/
│   └── (Auto-populated by download script)
├── requirements.txt
└── README.md
```

---

## How It Works

- **Features (40D):** Extracted with `partitura` (pitch, duration, beat offset, dynamics, voicing).
- **Backbone:** 6-layer Bidirectional Transformer Encoder (~5M Params).
- **Outputs:** Tempo scale $S(b)$, timing shift $\Delta t$, velocity $v$, articulation scale $d$, pedal CC64.
- **Loss:** Huber ($\Delta t$) + MSE (velocity, articulation, pedal) + 2nd-order smooth penalty on tempo scale.

---

## License
MIT
