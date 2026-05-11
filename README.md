# EAGP — ESM2-Augmented GAN for Protein Classification

EAGP is a deep learning pipeline for classifying protein sequences, combining **[ESM2](https://github.com/facebookresearch/esm)** protein language model embeddings with **WGAN-GP**-based data augmentation to address class imbalance. It supports both binary and multi-class classification of phage structural proteins.

## Pipeline Overview

```
FASTA → ESM2 Embedding → [WGAN-GP Augmentation] → MLP Classifier → Evaluation
```

1. **Embedding** — Encode protein sequences into fixed-size vectors using ESM2 (650M)
2. **Augmentation** (optional) — Train a WGAN-GP per minority class to generate synthetic embeddings, balancing the training set
3. **Classification** — Train a 3-layer MLP classifier on the embeddings
4. **Evaluation** — Compute accuracy, weighted F1, MCC, per-class metrics, and confusion matrix

## Project Structure

| File | Purpose |
|---|---|
| `data_by_imbalance.py` | Convert FASTA to CSV, then encode sequences with ESM2 for binary classification |
| `data_embedding.py` | Encode sequences from CSV with ESM2 for multi-class classification |
| `data_embedding_case_study.py` | Encode sequences for case study / inference |
| `model_binary.py` | MLP classifier for binary classification |
| `model_multi.py` | MLP classifier for multi-class classification (7 classes) |
| `train_binary.py` | Train binary classifier with optional WGAN-GP augmentation |
| `train_multi.py` | Train multi-class classifier with optional WGAN-GP augmentation |
| `test_binary.py` | Evaluate a trained binary classifier on test data |
| `test_multi.py` | Evaluate a trained multi-class classifier on test data |
| `case_study.py` | Run inference and output predictions to CSV |

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12 (GPU recommended)
- [ESM2 model weights](https://github.com/facebookresearch/esm) (loaded via HuggingFace transformers)
- Other dependencies: `pandas`, `numpy`, `scikit-learn`, `transformers`, `tqdm`, `biopython`

### Setup

```bash
pip install torch pandas numpy scikit-learn transformers tqdm biopython
```

Download data and pre-computed embeddings from:  
https://drive.google.com/drive/folders/1imtcW-fUSuOm-w_O2KiWk2AHYR_YGwRz

## Usage

### Step 1 — Embed Sequences with ESM2

For binary classification (FASTA input):
```bash
python data_by_imbalance.py \
    --model-name ESM2_650M \
    --data-root imbalance_data/IR_9 \
    --out-root outputs_imbalance/IR_9 \
    --device cuda
```

For multi-class classification (CSV input):
```bash
python data_embedding.py \
    --model-name ESM2_650M \
    --input-csv minor_capsid.csv \
    --output-csv outputs/minor_capsid_emb.csv \
    --device cuda
```

### Step 2 — Train Classifier

**Binary classification** (without augmentation):
```bash
python train_binary.py \
    --train outputs_imbalance/IR_9/train/embeddings.csv \
    --dev outputs_imbalance/IR_9/valid/embeddings.csv \
    --test outputs_imbalance/IR_9/test/embeddings.csv \
    --epochs 20 --device cuda --save-dir ./binary_checkpoints
```

**Binary classification** (with WGAN-GP augmentation):
```bash
python train_binary.py \
    --train outputs_imbalance/IR_9/train/embeddings.csv \
    --dev outputs_imbalance/IR_9/valid/embeddings.csv \
    --test outputs_imbalance/IR_9/test/embeddings.csv \
    --use-wgan --k-classes 1 --wgan-epochs 1000 \
    --epochs 20 --device cuda --save-dir ./binary_checkpoints_wgan
```

**Multi-class classification** (with WGAN-GP augmentation):
```bash
python train_multi.py \
    --train outputs_time/multi_esm2_encode/train/embeddings.csv \
    --dev outputs_time/multi_esm2_encode/valid/embeddings.csv \
    --test outputs_time/multi_esm2_encode/test/embeddings.csv \
    --use-wgan --k-classes 2 --wgan-epochs 3000 \
    --epochs 20 --device cuda --save-dir ./multi_checkpoints
```

### Step 3 — Evaluate

```bash
# Binary
python test_binary.py \
    --test outputs_time/binary_esm2_encode/test/embeddings.csv \
    --model-path ./weight/spilt_by_time_binary/best.pt \
    --device cuda

# Multi-class
python test_multi.py \
    --test outputs_smi/multi_esm2_encode_40/test/embeddings.csv \
    --model-path ./weight/spilt_by_smi_multi_40_wgan/best.pt \
    --device cuda
```

### Step 4 — Inference on New Data (Case Study)

```bash
python case_study.py \
    --test outputs_case_study/case_study/embeddings.csv \
    --model-path ./spilt_by_time_multi/best.pt \
    --out-csv outputs_case_study/multi_prediction_results.csv \
    --device cuda
```

## Quick Test

Use the lightweight test scripts to verify setup:

```bash
python test_binary.py
python test_multi.py
```

## Class Labels

### Multi-class (7 types)
| ID | Class |
|----|-------|
| 0 | minor capsid |
| 1 | tail fiber |
| 2 | major tail |
| 3 | portal |
| 4 | minor tail |
| 5 | baseplate |
| 6 | major capsid |

### Binary
| ID | Class |
|----|-------|
| 0 | non-PVP |
| 1 | PVP |

## Dataset Statistics

> All data from `baseline/` directory.

### Binary Classification — Imbalance Ratio (IR) Splits

Each IR configuration creates a different train/val/test split by controlling the PVP:non-PVP ratio in the training set.

**IR = 9** (most imbalanced):
| Split | Total | non-PVP | PVP |
|---|---|---|---|
| Train | 31,212 | 28,017 | 3,195 |
| Val | 10,404 | 9,453 | 951 |
| Test | 10,404 | 9,348 | 1,056 |

**IR = 3**:
| Split | Total | non-PVP | PVP |
|---|---|---|---|
| Train | 37,459 | 28,099 | 9,360 |
| Val | 12,486 | 9,350 | 3,136 |
| Test | 12,487 | 9,375 | 3,112 |

**IR = 1** (balanced):
| Split | Total | non-PVP | PVP |
|---|---|---|---|
| Train | 42,196 | 21,112 | 21,084 |
| Val | 14,065 | 6,962 | 7,103 |
| Test | 14,067 | 7,090 | 6,977 |

Other IR settings: IR_5, IR_7 (intermediate imbalance levels).

### Multi-class — Time-based Split

| Split | Total | minor capsid | tail fiber | major tail | portal | minor tail | baseplate | major capsid |
|---|---|---|---|---|---|---|---|---|
| Train | 9,582 | 252 | 1,258 | 786 | 1,544 | 2,565 | 1,819 | 1,358 |
| Val | 4,107 | 108 | 539 | 337 | 662 | 1,099 | 780 | 582 |
| Test | 4,329 | 38 | 506 | 335 | 564 | 1,419 | 965 | 502 |

### Multi-class — SMI Split (40% train)

| Split | Total | minor capsid | tail fiber | major tail | portal | minor tail | baseplate | major capsid |
|---|---|---|---|---|---|---|---|---|
| Train | 6,031 | 132 | 880 | 558 | 1,028 | 1,700 | 1,372 | 361 |
| Val | 2,585 | 57 | 377 | 239 | 441 | 729 | 588 | 154 |
| Test | 7,748 | 184 | 841 | 633 | 1,193 | 2,399 | 1,367 | 1,131 |

### Binary — Time-based Split

| Split | Total | non-PVP | PVP |
|---|---|---|---|
| Train | 49,932 | 24,966 | 24,966 |
| Val | 5,476 | 2,738 | 2,738 |
| Test | 17,593 | 10,093 | 7,500 |

### Binary — SMI Split (40% train)

| Split | Total | non-PVP | PVP |
|---|---|---|---|
| Train | 12,053 | 6,023 | 6,030 |
| Val | 5,160 | 2,579 | 2,581 |
| Test | 15,475 | 7,739 | 7,736 |

## Key Parameters

| Argument | Description | Default |
|---|---|---|
| `--use-wgan` | Enable WGAN-GP data augmentation | disabled |
| `--k-classes` | Number of minority classes to augment | 1 (binary) / 2 (multi) |
| `--wgan-epochs` | WGAN-GP training epochs | 1000 |
| `--z-dim` | Latent noise dimension for GAN | 32 (binary) / 4 (multi) |
| `--epochs` | Classifier training epochs | 20 |
| `--batch-size` | Batch size | 8 (binary) / 64 (multi) |
| `--device` | Device to use | cuda |

## Method

- **ESM2** (650M parameters) encodes each protein sequence into a 640-dimensional embedding via mean pooling of the last hidden layer
- **WGAN-GP** generates synthetic minority-class embeddings to balance the training set (upsamples to ~2500–3000 samples per class)
- The **classifier** is a 3-layer MLP with GELU activation, dropout regularization, and cross-entropy loss
- Models are selected by best validation weighted F1-score
