# Tenuto

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyleconciso/tenuto/blob/main/notebooks/colab_training.ipynb)

Predicts human performance nuance (rubato, micro-timing, velocity, articulation, sustain pedal) from sheet music or MIDI scores using a non-autoregressive PyTorch model.

Instead of predicting events note-by-note, Tenuto runs a single forward pass over the score grid.

---

## Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyleconciso/tenuto/blob/main/notebooks/colab_training.ipynb)

If you don't have datasets ready yet, `train.py` runs on synthetic data by default so you can test the code immediately.

---

## Quick Start

### 1. Setup
```bash
git clone https://github.com/kyleconciso/tenuto.git
cd tenuto
pip install -r requirements.txt
```

### 2. Test Run
```bash
python -m src.train --model_type bigru --in_features 10 --epochs 2
```

### 3. Preprocess
Drop `.xml` or `.mid` files in `./data/raw/`:
```bash
python -m src.preprocess --raw_dir ./data/raw --processed_dir ./data/processed
```

### 4. Train

BiGRU:
```bash
python -m src.train --model_type bigru --in_features 10 --epochs 10 --batch_size 16
```

Transformer:
```bash
python -m src.train --model_type transformer --in_features 40 --epochs 20 --batch_size 16 --lr 1e-4
```

### 5. Predict
```bash
python -m src.infer --score sample.xml --checkpoint checkpoints/best_transformer_model.pth --model_type transformer
```

---

## Project Layout

```
tenuto/
├── notebooks/
│   └── colab_training.ipynb
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   ├── infer.py
│   └── utils.py
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
└── README.md
```

---

## How It Works

- **Features (40D):** Extracted with `partitura` (pitch, duration, beat offset, dynamics, voicing).
- **Backbone:** 6-layer Bidirectional Transformer Encoder.
- **Outputs:** Tempo scale $S(b)$, timing shift $\Delta t$, velocity $v$, articulation scale $d$, pedal CC64.
- **Loss:** Huber ($\Delta t$) + MSE (velocity) + 2nd-order smooth penalty on tempo scale.

---

## License
MIT
