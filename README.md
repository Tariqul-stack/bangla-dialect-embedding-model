# Bangla Dialect Embedding Model

A dialect-aware Bangla embedding model using Mamba2/Mamba3 SSM architecture with contrastive learning, developed at the Edge Intelligence Lab.

## Project Overview

This project builds a dialect-aware embedding model for Bangla using Mamba2 and Mamba3 (State Space Model) architectures. The model captures dialectal variations across 8 Bangla dialects and produces high-quality embeddings using contrastive learning.

**Novelty:**
- First application of Mamba2/Mamba3 architecture to Bangla dialect embedding
- Covers 8 dialects simultaneously (most prior work covers 2-3)
- Parallel corpus + contrastive learning for embedding (not classification)
- Config-driven architecture switching (Mamba2/Mamba3) and loss selection (NT-Xent/Triplet)

**Dialects covered:** Standard, Rajshahi, Sylheti, Chittagong, Rangpur, Mymensingh, Barishal, Rakhain

## Project Structure

```
bangla-dialect-embedding-model/
├── data/
│ └── raw/
│ └── Local_Language_Dataset.xlsx
├── src/
│ ├── dataset.py
│ ├── data_preprocessing.py
│ ├── tokenizer.py
│ ├── model.py
│ └── train.py
├── configs/
│ ├── config.yaml
│ └── config_mamba2.yaml
├── evaluate/
│ └── eval.py
├── results.md
├── requirements.txt
└── README.md
```

## Pipeline

- **Data**: Parallel Bangla dialect corpus — 3,452 sentences × 8 dialects (7,703 total samples)
- **Preprocessing**: Unicode NFC normalization, Bangla character filtering
- **Tokenizer**: BanglaBERT (`sagorsarker/bangla-bert-base`) — vocab size: 101,975
- **Model**: Mamba2/Mamba3 SSM architecture (pure-PyTorch fallback)
- **Loss**: NT-Xent (SimCLR) or Triplet Loss — config-driven
- **Training**: AdamW optimizer + Cosine LR scheduler + gradient clipping

## Setup

```bash
pip install -r requirements.txt
```

## Run Training

```bash
# Mamba2 + NT-Xent (default)
python -m src.train

# Switch architecture or loss in configs/config.yaml:
# model.architecture: "mamba2" or "mamba3"
# training.loss: "nt_xent" or "triplet"
```

## Run Evaluation

```bash
python evaluate/eval.py
```

## Results Summary

| Experiment | Gap | Val Loss |
|------------|-----|----------|
| SSM Placeholder (Baseline) | 0.4843 | 0.1495 |
| Mamba2 + NT-Xent (30ep) | 0.8086 | 0.3816 |
| Mamba2 + Triplet (30ep) | 0.8873 | 0.2452 |
| Mamba3 + Triplet (30ep) | 0.8886 | 0.2440 |

Full results in [results.md](results.md)

## Key Findings

- Mamba2/Mamba3 outperforms SSM placeholder by +83% similarity gap
- Triplet Loss outperforms NT-Xent on current small dataset
- Mamba3 achieves marginally better gap (0.8886) with better negative separation
- Mamba3 is ~60% slower per epoch than Mamba2
- Rakhain dialect scores lowest due to linguistic distance from Standard Bangla

## Status

- [x] Parallel dialect dataset (3,452 sentences × 8 dialects)
- [x] Data preprocessing pipeline
- [x] BanglaBERT tokenizer
- [x] Mamba2 SSM architecture (pure-PyTorch fallback)
- [x] Mamba3 SSM architecture (complex-valued states, BCNorm, ET discretization)
- [x] NT-Xent contrastive loss
- [x] Triplet Loss (ablation)
- [x] Config-driven architecture + loss selection
- [x] Training loop with validation + checkpointing
- [x] Cosine similarity evaluation + per-dialect breakdown
- [x] Ablation study (NT-Xent vs Triplet)
- [x] Architecture comparison (Mamba2 vs Mamba3)
- [ ] Mamba1 baseline experiment
- [ ] BanglaBERT baseline experiment
- [ ] Real Mamba2/Mamba3 CUDA kernel (RunPod A100)
- [ ] New dataset integration
- [ ] Paper writing

## Lab

Edge Intelligence Lab
Supervisor: Shah Nawaz Haider