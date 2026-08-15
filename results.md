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

## Per-dialect Similarity (Mamba2 Fallback)
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

## Key Improvement
- Similarity gap: 0.4843 → 0.7949 (+64% improvement)
- Negative similarity: 0.4711 → -0.0030 (near zero, much better separation)