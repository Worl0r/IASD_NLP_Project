## References

# IASD NLP Project – Linformer & Transformer Classifier

## Overview

This project provides an advanced implementation of Transformer and Linformer models for text classification tasks, with a special focus on memory and computational efficiency through linear attention (Linformer). It includes a complete pipeline: data preparation, training, validation, and TensorBoard visualization.

- **Linformer Paper**: [arXiv:2006.04768](https://arxiv.org/pdf/2006.04768.pdf)
- **Original Code**: https://github.com/tatp22/linformer-pytorch

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Results & Visualization](#results--visualization)
- [References](#references)

---

## Features

- Implementation of both Linformer (linear attention) and standard Transformer.
- End-to-end pipeline for text classification (example: SST-2).
- Custom preprocessing, tokenization, padding, and masking.
- Training and validation with TensorBoard logging.
- Modular design for easy experimentation with other datasets or architectures.

---

## Installation

### Prerequisites

- Python ≥ 3.11
- GPU recommended (CUDA supported)
- [Poetry](https://python-poetry.org/) or pip

### Install dependencies

```bash
# With poetry
poetry install

# Or with pip
pip install -r requirements.txt
```

Main dependencies:

- torch
- transformers
- datasets
- tensorboard
- pyyaml
- tqdm
- keras

---

## Usage

### Start training

```bash
python main_NLP.py
```

### Visualize metrics

```bash
tensorboard --logdir runs
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

---

## Project Structure

```
IASD_NLP_Project/
│
├── main_NLP.py                # Main training/validation script
├── dataset.py                 # Preprocessing and DataCollator
├── configuration.yaml         # Configuration file (batch_size, etc.)
├── transformers_project/
│   ├── linformer/             # Linformer implementation (attention, layers, etc.)
│   ├── models.py              # Other architectures (e.g., vanilla Transformer)
│   └── utils/                 # Utility functions
├── runs/                      # TensorBoard logs
├── input/                     # Input data (optional)
├── README.md
└── pyproject.toml / requirements.txt
```

---

---

## Configuration

The `configuration.yaml` file allows you to set some global hyperparameters:

```yaml
DATA:
  n_samples: 10000
  test_ratio: 0.2
  batch_size: 64
MODEL:
  activation: "swish"
```

Other hyperparameters (dimensions, number of layers, etc.) are defined in `main_NLP.py`.

---

## Technical Details

### Models

- **TransformerClassifier**: Based on PyTorch `nn.TransformerEncoder`.
- **LinformerEnc**: Faithful implementation of the paper, with linear projection of keys/values to reduce memory and computational complexity.

### Pipeline

1. **Data loading**: Using `datasets` (e.g., SST-2).
2. **Tokenization**: Via `transformers.BertTokenizer`.
3. **Preprocessing**: Padding, masking, tensor conversion.
4. **Training**: Standard PyTorch loop, TensorBoard logging.
5. **Validation**: Compute loss and accuracy on the validation set.

### Customization

- Easily switch dataset, tokenizer, or architecture.
- Linformer modules are highly configurable (number of heads, projection dimension, projection sharing, etc.).

---

## Results & Visualization

- Training and validation metrics are logged in `runs/`.
- Use TensorBoard to visualize loss, accuracy, and compare architectures.

---

## References

- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/pdf/2006.04768.pdf)
- [Original Linformer PyTorch Code](https://github.com/tatp22/linformer-pytorch)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

---

## Authors

- Project developed for the IASD LLM module, 2025.
- Contact:

  - Brice CONVERS - brice.convers@dauphine.eu

  - Paul MALET - paul.malet@dauphine.eu
  - Shijie TIAN - shijie.tian@dauphine.eu

---

Feel free to open an issue or pull request for any suggestion or improvement!
