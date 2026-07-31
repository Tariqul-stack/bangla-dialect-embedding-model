# Bangla Dialect Embedding Model

A dialect-aware Bangla embedding model using Mamba/SSM architecture, developed at the Edge Intelligence Lab.

## Project Overview

This project aims to build a dialect-aware embedding model for the Bangla language using state-space model (SSM/Mamba) architecture. The goal is to capture dialectal variations in Bangla text and produce high-quality embeddings for downstream NLP tasks.

## Project Structure

```
bangla-dialect-embedding-model/
├── data/
│ ├── raw/
│ └── processed/
├── src/
│ ├── dataset.py
│ ├── data_preprocessing.py
│ ├── tokenizer.py
│ ├── model.py
│ └── train.py
├── configs/
│ └── config.yaml
├── notebooks/
│ └── data_exploration.ipynb
├── evaluate/
│ └── eval.py
├── requirements.txt
└── README.md
```

## Pipeline

- **Data**: Bangla dialect text corpus (existing public corpora + custom dataset in progress)
- **Preprocessing**: Unicode normalization, dialect-specific text cleaning
- **Tokenizer**: BanglaBERT tokenizer (placeholder — custom tokenizer TBD)
- **Model**: Mamba/SSM-based embedding architecture (TBD — architecture meeting scheduled)
- **Training**: AdamW optimizer, configurable via `configs/config.yaml`

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
- [x] Data preprocessing pipeline
- [x] Dataset class
- [x] Tokenizer setup
- [x] Model skeleton
- [x] Training loop
- [ ] Mamba/SSM architecture (architecture meeting TBD)
- [ ] Custom dialect dataset
- [ ] Evaluation pipeline

## Lab

Edge Intelligence Lab