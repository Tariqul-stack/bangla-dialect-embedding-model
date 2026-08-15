# Bangla Dialect Embedding Model

A dialect-aware Bangla embedding model using Mamba2 SSM architecture with contrastive learning, developed at the Edge Intelligence Lab.

## Project Overview

This project builds a dialect-aware embedding model for Bangla using Mamba2 (State Space Model) architecture. The model captures dialectal variations across 8 Bangla dialects and produces high-quality embeddings using contrastive learning.

**Novelty:**
- First application of Mamba2 architecture to Bangla dialect embedding
- Covers 8 dialects simultaneously (most prior work covers 2-3)
- Parallel corpus + NT-Xent contrastive learning for embedding (not classification)

**Dialects covered:** Standard, Rajshahi, Sylheti, Chittagong, Rangpur, Mymensingh, Barishal, Rakhain

## Project Structure

```
bangla-dialect-embedding-model/
├── data/
│ └── raw/
│ └── Local_Language_Dataset.xlsx # Parallel dialect corpus (3452 sentences × 8 dialects)
├── src/
│ ├── dataset.py # Dialect-aware PyTorch Dataset + DataLoader
│ ├── data_preprocessing.py # Excel loading, cleaning, flattening
│ ├── tokenizer.py # BanglaBERT tokenizer wrapper
│ ├── model.py # Mamba2 SSM model + NT-Xent Contrastive Loss
│ └── train.py # Training loop with validation + checkpointing
├── configs/
│ ├── config.yaml # Main experiment config
│ └── config_mamba2.yaml # Mamba2 full experiment config
├── evaluate/
│ └── eval.py # Cosine similarity evaluation + per-dialect breakdown
├── requirements.txt
└── README.md
```

## Pipeline

- **Data**: Parallel Bangla dialect corpus — 3,452 sentences × 8 dialects (7,703 total samples)
- **Preprocessing**: Unicode NFC normalization, Bangla character filtering
- **Tokenizer**: BanglaBERT (`sagorsarker/bangla-bert-base`) — vocab size: 101,975
- **Model**: Mamba2 SSM architecture (pure-PyTorch fallback for environments without CUDA extension)
- **Loss**: NT-Xent (SimCLR) contrastive loss — temperature=0.07
- **Training**: AdamW optimizer + Cosine LR scheduler + gradient clipping

## Setup

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python -m src.train
```

## Run Evaluation

```bash
python evaluate/eval.py
```

## Results

### Experiment 1: SSM Placeholder (Baseline)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.9555 |
| Negative Similarity | 0.4711 |
| Similarity Gap | 0.4843 |
| Best Val Loss | 0.1495 |

### Experiment 2: Mamba2 Fallback (Pure PyTorch, 10 epochs)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.7919 |
| Negative Similarity | -0.0030 |
| Similarity Gap | 0.7949 |
| Best Val Loss | 0.4512 |

### Per-dialect Similarity (Mamba2 Fallback)
| Dialect | Similarity |
|---------|------------|
| Standard | 1.0000 |
| Barishal | 0.7286 |
| Sylheti | 0.6902 |
| Mymensingh | 0.6837 |
| Chittagong | 0.6719 |
| Rangpur | 0.5802 |
| Rajshahi | 0.5759 |
| Rakhain | 0.1608 |

## Status

- [x] Project structure
- [x] Parallel dialect dataset (3,452 sentences × 8 dialects)
- [x] Data preprocessing pipeline
- [x] Dialect-aware Dataset class with dialect labels
- [x] Train / Val / Test split (80/10/10)
- [x] BanglaBERT tokenizer
- [x] Mamba2 SSM architecture (with pure-PyTorch fallback)
- [x] NT-Xent contrastive loss
- [x] Training loop with validation + checkpointing
- [x] Cosine similarity evaluation + per-dialect breakdown
- [ ] Real Mamba2 CUDA kernel integration
- [ ] 30 epoch full training run
- [ ] Mamba3 experiment

## Lab

Edge Intelligence Lab
Supervisor: Shah Nawaz Haider