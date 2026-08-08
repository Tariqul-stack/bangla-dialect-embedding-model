# Bangla Dialect Embedding Model

A dialect-aware Bangla embedding model using Mamba2/SSM architecture with contrastive learning, developed at the Edge Intelligence Lab.

## Project Overview

This project aims to build a dialect-aware embedding model for the Bangla language using state-space model (SSM/Mamba2) architecture. The goal is to capture dialectal variations across 7 Bangla dialects and produce high-quality embeddings for downstream NLP tasks.

**Dialects covered:** Standard, Rajshahi, Sylheti, Chittagong, Rangpur, Mymensingh, Barishal, Rakhain

## Project Structure

```
bangla-dialect-embedding-model/
├── data/
│   ├── raw/
│   │   └── Local_Language_Dataset.xlsx   # Parallel dialect corpus (3452 sentences × 8 dialects)
│   └── processed/
├── src/
│   ├── dataset.py              # Dialect-aware PyTorch Dataset + DataLoader
│   ├── data_preprocessing.py   # Excel loading, cleaning, flattening
│   ├── tokenizer.py            # BanglaBERT tokenizer wrapper
│   ├── model.py                # SSM model + Contrastive Loss
│   └── train.py                # Training loop with validation + checkpointing
├── configs/
│   └── config.yaml
├── notebooks/
│   └── data_exploration.ipynb
├── evaluate/
│   └── eval.py
├── requirements.txt
└── README.md
```

## Pipeline

- **Data**: Parallel Bangla dialect corpus — 3,452 sentences across 8 dialects (27,616 total pairs)
- **Preprocessing**: Unicode NFC normalization, Bangla character filtering, dialect-aware cleaning
**Tokenizer**: BanglaBERT tokenizer (`sagorsarker/bangla-bert-base`) — placeholder, custom tokenizer planned
- **Model**: Mamba2/SSM-based embedding architecture (SSM placeholder — Mamba2 integration in progress)
- **Loss**: Contrastive Loss — pulls same-meaning dialect pairs together, pushes different sentences apart
- **Training**: AdamW optimizer + Cosine LR scheduler, configurable via `configs/config.yaml`

## Setup

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python -m src.train
```

## Status

- [x] Project structure
- [x] Parallel dialect dataset (3,452 sentences × 8 dialects)
- [x] Data preprocessing pipeline (Excel → clean samples → DataLoader)
- [x] Dialect-aware Dataset class with dialect labels
- [x] Train / Val / Test split (80/10/10)
- [x] Tokenizer setup
- [x] SSM model skeleton with residual connections
- [x] Contrastive Loss implementation
- [x] Training loop with validation, checkpointing, LR scheduling
- [ ] Mamba2 architecture integration (in progress)
- [ ] Evaluation pipeline
- [ ] Additional dialect data collection

## Lab

Edge Intelligence Lab  
Supervisor: Shah Nawaz Haider