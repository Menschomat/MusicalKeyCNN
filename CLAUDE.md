# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

This project uses `uv`. Run scripts with `uv run <script>` or activate the venv at `.venv/`.

Install dependencies:
```sh
uv sync
```

## Common Commands

```sh
# Predict keys for an MP3 file or folder
uv run predict_keys.py -f path/to/song.mp3
uv run predict_keys.py -f path/to/folder/ -m checkpoints/keynet.pt --device cpu

# Start the FastAPI inference server
uv run uvicorn api:app --reload

# Preprocess datasets (both training and eval sets)
uv run preprocess_data.py

# Train the model
uv run train.py

# Evaluate with MIREX metrics
uv run eval.py

# Lint / format
uv run ruff check .
uv run black .

# Run tests
uv run pytest
```

## Architecture

The pipeline has four stages, each with its own script:

**1. Preprocessing (`preprocess_data.py`)**  
Reads raw `.mp3` files from the GiantSteps datasets, applies pitch-shifting augmentation (−4 to +7 semitones), computes log-magnitude CQT spectrograms via librosa (105 bins, hop=8820, fmin=65 Hz, bins_per_octave=24), and saves each variant as a `.pkl` file. For the evaluation set (GiantSteps, not MTG), `create_annotations_txt()` must be called first to convert per-track `.key` files into a unified `annotations.txt`.

**2. Model (`model.py`)**  
`KeyNet` is a fully-convolutional CNN (no dense layers): 9 `BasicConv2d` blocks (Conv2d → BN → ELU), 3 max-pool + Dropout2d stages, a 1×1 final conv to 24 channels, and global average pooling. Input shape: `(B, 1, 105, T)`. Output: 24 logits (Camelot Wheel classes).

**3. Dataset (`dataset.py`)**  
`KeyDataset` loads `.pkl` spectrograms, randomly selects a pitch-shifted variant at each `__getitem__`, adjusts the Camelot label to match the shift, and returns a random `chunk_samples=100` time-frame window. Only high-confidence annotations (`confidence == 2`) are included. Key labels use the **Camelot Wheel** encoding: indices 0–11 = minor keys, 12–23 = major keys; enharmonic equivalents share the same index.

**4. Inference (`predict_keys.py`, `api.py`)**  
`predict_keys.py` is the CLI; `api.py` is a FastAPI wrapper exposing `POST /predict` (upload `.mp3`, receive JSON with `class_id`, `camelot`, `key`) and `GET /health`. Both reuse `preprocess_mp3()` from `predict_keys.py` and `load_model()` from `eval.py`. The model is loaded once at startup via FastAPI's lifespan handler.

## Dataset Layout

```
Dataset/
    giantsteps-mtg-key-dataset/   # Training data (MTG)
        audio/
        annotations/annotations.txt
    giantsteps-key-dataset/       # Evaluation data (GiantSteps)
        audio/
        annotations/giantsteps/*.key  → converted to annotations.txt by preprocess_data.py
    mtg-preprocessed-audio/       # Output of preprocess_data.py (training)
    giantsteps-preprocessed-audio/ # Output of preprocess_data.py (eval)
checkpoints/keynet.pt             # Saved model weights
```

## Key Design Decisions

- **No dense layers**: `KeyNet` uses a 1×1 conv + global avg pool instead of a fully-connected head, following Korzeniowski & Widmer (2018), to avoid overfitting and support variable-length input.
- **Camelot Wheel label space**: 24 classes rather than the standard 24-key music theory representation; enharmonic equivalents are collapsed. The pitch-shift augmentation in `KeyDataset.__getitem__` adjusts labels using modular arithmetic over the wheel (separate mod-12 arithmetic for minor/major halves).
- **Training uses early stopping with LR halving**: patience=50 epochs; LR halves on plateau, stops when LR < 1e-7.
- **Eval uses full-track inference**: `chunk_samples=float('inf')` and `pitch_range=(0,0)` in `eval.py` so the whole track is fed at once with no augmentation.
