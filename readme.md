# GELATO

**GELATO** is a deep-learning framework that infers **metabolic and ecophysiological phenotypes** of prokaryotes from nucleic-acid–derived sequence features. It supports **141 phenotypic traits** and can **recommend culture media** from predicted metabolic potential and genomic context.

---

## Key features

| | |
|---|---|
| **Automated annotation** | Pipeline-friendly integration with **Prokka**, **KofamScan**, and **Diamond** for high-throughput, consistent gene-level annotation. |
| **Phenotype prediction** | **GELATO** models predict binary or continuous traits from genome-derived token sequences. |
| **Media recommendation** | Suggests growth media by combining predicted compound utilization / requirements with similarity-based reasoning. |

---

## Overview

![GELATO overview 1](https://cdn.nlark.com/yuque/0/2026/png/35707472/1777465490258-6e6d5e5f-32cd-430b-9234-a7c674287c8f.png?x-oss-process=image%2Fformat%2Cwebp)

![GELATO overview 2](https://cdn.nlark.com/yuque/0/2026/png/35707472/1774410238050-88b99a34-e37e-4b0d-8fe1-a466c0d278e9.png)

---

## Usage

Full installation, inputs, outputs, and command-line examples are documented here:

**[Hugging Face — GELATO README](https://huggingface.co/NazarickArchivist/gelato/blob/main/readme.md)**

---

## Online platform

Try GELATO in the browser:

**[ATCG — GELATO](https://atcg.cncb.ac.cn/gelato/)**

---

## Training

`train_model.py` is the **single-trait training entry point**. It loads **train / validation / test** CSVs, reads genome annotation text from paths in the `path` column, builds a vocabulary and PyTorch `DataLoader`, and trains either:

- **`GELATO`** — binary classification (`--type c`), or  
- **`GELATO_r`** — regression (`--type r`).

It saves the vocabulary, the best checkpoint, and **test-set metrics** under `--model_dir`.

### Tasks at a glance

| Item | Description |
|------|-------------|
| **Task** | `--type c`: binary classification (BCE + sigmoid; Acc / AUC). `--type r`: regression (MSE; R² / RMSE). |
| **Input** | Each row’s `path` points to a **genome annotation text file** (e.g. output from `/GELATO/annot_genome.py`). |
| **Model** | **`GELATO`** for classification, **`GELATO_r`** for regression. |
| **Hardware** | CUDA GPU: `cuda:{--device}`. |

### Data format (`/data/benchmark_data/species_split`)

Each CSV must include:

1. **`path`** — absolute path to the genome annotation text file for that sample.  
2. **Label column** — the column name passed as **`--c`**:  
   - **Classification:** labels in `{0, 1}` (or compatible floats for BCE).  
   - **Regression:** continuous values.

### Command-line arguments

| Argument | Description |
|----------|-------------|
| `--model_dir` | Root directory for outputs (creates `vocab/` and `model/`). |
| `--type` | `c` = classification; `r` = regression. |
| `--train` | Path to training CSV. |
| `--valid` | Path to validation CSV. |
| `--test` | Path to test CSV. |
| `--c` | Phenotype column name (must match the CSV header), e.g. `acetate`, `Temperature_optgrow`. |
| `--device` | GPU id (default: `0`). |

**Pretrained embeddings (expected in the working directory when using pretrained mode):**

- Classification: `vectors_dim50.txt`  
- Regression: `vectors_dim300.txt`

### Example

```bash
# Binary classification
python train_model.py \
  --model_dir /path/to/run_cls \
  --type c \
  --train /path/to/train.csv \
  --valid /path/to/valid.csv \
  --test /path/to/test.csv \
  --c maltose \
  --device 0

# Regression
python train_model.py \
  --model_dir /path/to/run_reg \
  --type r \
  --train /path/to/train.csv \
  --valid /path/to/valid.csv \
  --test /path/to/test.csv \
  --c Temperature_optgrow \
  --device 0
```


