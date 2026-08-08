# Evaluation Results

## Model: Bangla Dialect Embedding Model (SSM Placeholder + Contrastive Loss)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | SSM Placeholder (Mamba2 integration in progress) |
| Loss Function | Contrastive Loss (margin=1.0) |
| Tokenizer | BanglaBERT (`sagorsarker/bangla-bert-base`) |
| Vocab Size | 101,975 |
| Hidden Dim | 256 |
| Num Layers | 4 |
| Batch Size | 32 |
| Epochs | 10 |
| Optimizer | AdamW (lr=0.0001, weight_decay=0.01) |
| LR Scheduler | Cosine Annealing |
| Train/Val/Test Split | 80/10/10 |

### Dataset

| Property | Value |
|----------|-------|
| Source | Local parallel dialect corpus |
| Sentences | 3,452 |
| Dialects | 8 (Standard, Rajshahi, Sylheti, Chittagong, Rangpur, Mymensingh, Barishal, Rakhain) |
| Total samples | 7,703 |

### Training Loss

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.2857 | 0.2118 |
| 2 | 0.2136 | 0.1866 |
| 3 | 0.1856 | 0.1710 |
| 4 | 0.1687 | 0.1644 |
| 5 | 0.1571 | 0.1589 |
| 6 | 0.1485 | 0.1541 |
| 7 | 0.1426 | 0.1531 |
| 8 | 0.1391 | 0.1507 |
| 9 | 0.1354 | **0.1495** ← best |
| 10 | 0.1340 | 0.1499 |

### Similarity Evaluation (Test Set)

| Metric | Score |
|--------|-------|
| Positive pair similarity | 0.9555 ↑ |
| Negative pair similarity | 0.4711 ↓ |
| Similarity gap | 0.4843 |

### Per-Dialect Similarity (vs Standard Bangla)

| Dialect | Cosine Similarity |
|---------|------------------|
| Standard | 1.0000 |
| Barishal | 0.9571 |
| Sylheti | 0.9453 |
| Mymensingh | 0.9396 |
| Chittagong | 0.9302 |
| Rangpur | 0.9189 |
| Rajshahi | 0.8746 |
| Rakhain | 0.8327 |

### Observations

- Model successfully learns dialect-aware embeddings with high positive pair similarity (0.9555)
- Clear separation between positive and negative pairs (gap: 0.4843)
- Rakhain dialect shows lowest similarity (0.8327), consistent with its linguistic distance from standard Bangla
- No overfitting observed — val loss closely tracks train loss throughout training

### Next Steps

- [ ] Replace SSM placeholder with Mamba2 architecture
- [ ] Expand dataset with more dialect samples
- [ ] Add downstream task evaluation (dialect classification, retrieval)
- [ ] Compare with transformer-based baseline (BanglaBERT)