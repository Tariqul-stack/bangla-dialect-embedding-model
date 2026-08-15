# Bangla Dialect Embedding Model — Results

## Experiment 1: SSM Placeholder (Baseline)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.9555 |
| Negative Similarity | 0.4711 |
| Similarity Gap | 0.4843 |
| Best Val Loss | 0.1495 |

## Experiment 2: Mamba2 Fallback (Pure PyTorch, 10 epochs)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.7919 |
| Negative Similarity | -0.0030 |
| Similarity Gap | 0.7949 |
| Best Val Loss | 0.4512 |

## Experiment 3: Mamba2 Fallback (Pure PyTorch, 30 epochs)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.8084 |
| Negative Similarity | -0.0001 |
| Similarity Gap | 0.8086 |
| Best Val Loss | 0.3816 |
| Best Checkpoint | Epoch 24 |

## Per-dialect Similarity Comparison
| Dialect | 10 epoch | 30 epoch | Change |
|---------|----------|----------|--------|
| Standard | 1.0000 | 1.0000 | — |
| Barishal | 0.7286 | 0.7621 | +0.033 |
| Sylheti | 0.6902 | 0.7175 | +0.027 |
| Mymensingh | 0.6837 | 0.7150 | +0.031 |
| Chittagong | 0.6719 | 0.7031 | +0.031 |
| Rangpur | 0.5802 | 0.6198 | +0.040 |
| Rajshahi | 0.5759 | 0.6011 | +0.025 |
| Rakhain | 0.1608 | 0.1438 | -0.017 |

## Key Improvements
- Similarity gap: 0.4843 → 0.8086 (+67% from baseline)
- Negative similarity: 0.4711 → -0.0001 (near zero separation)
- All dialects improved with more epochs except Rakhain (most distant dialect)