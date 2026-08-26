# Bangla Dialect Embedding Model — Results

## Experiment 1: SSM Placeholder (Baseline)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.9555 |
| Negative Similarity | 0.4711 |
| Similarity Gap | 0.4843 |
| Best Val Loss | 0.1495 |

## Experiment 2: Mamba2 + NT-Xent (10 epochs)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.7919 |
| Negative Similarity | -0.0030 |
| Similarity Gap | 0.7949 |
| Best Val Loss | 0.4512 |
| Best Epoch | 10 |

## Experiment 3: Mamba2 + NT-Xent (30 epochs)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.8084 |
| Negative Similarity | -0.0001 |
| Similarity Gap | 0.8086 |
| Best Val Loss | 0.3816 |
| Best Epoch | 24 |

## Experiment 4: Mamba2 + Triplet Loss (30 epochs)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.8871 |
| Negative Similarity | -0.0003 |
| Similarity Gap | 0.8873 |
| Best Val Loss | 0.2452 |
| Best Epoch | 27 |

## Experiment 5: Mamba3 + Triplet Loss (30 epochs)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.8865 |
| Negative Similarity | -0.0021 |
| Similarity Gap | 0.8886 |
| Best Val Loss | 0.2440 |
| Best Epoch | 27 |
| Parameters | 13,444,480 |

## Ablation Study — Loss Function Comparison (Mamba2)
| Loss Function | Gap | Val Loss | Winner |
|---------------|-----|----------|--------|
| NT-Xent | 0.8086 | 0.3816 | — |
| Triplet Loss | 0.8873 | 0.2452 |

## Architecture Comparison — Mamba2 vs Mamba3
| Metric | Mamba2+Triplet | Mamba3+Triplet | Winner |
|--------|----------------|----------------|--------|
| Similarity Gap | 0.8873 | 0.8886 | Mamba3 |
| Val Loss | 0.2452 | 0.2440 | Mamba3 |
| Negative Sim | -0.0003 | -0.0021 | Mamba3 |
| Speed | ~31s/epoch | ~50s/epoch | Mamba2 |

## Per-dialect Similarity — Full Comparison
| Dialect | Placeholder | Mamba2+NT-Xent | Mamba2+Triplet | Mamba3+Triplet |
|---------|-------------|----------------|----------------|----------------|
| Standard | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Barishal | 0.9571 | 0.7621 | 0.8848 | 0.8870 |
| Sylheti | 0.9453 | 0.7175 | 0.8923 | 0.8807 |
| Chittagong | 0.9302 | 0.7031 | 0.8595 | 0.8666 |
| Mymensingh | 0.9396 | 0.7150 | 0.8297 | 0.8171 |
| Rangpur | 0.9189 | 0.6198 | 0.7855 | 0.7853 |
| Rajshahi | 0.8746 | 0.6011 | 0.7211 | 0.7196 |
| Rakhain | 0.8327 | 0.1438 | 0.3010 | 0.3279 |

## Overall Progress
| Experiment | Gap | Improvement vs Baseline |
|------------|-----|------------------------|
| SSM Placeholder | 0.4843 | — |
| Mamba2 + NT-Xent 10ep | 0.7949 | +64% |
| Mamba2 + NT-Xent 30ep | 0.8086 | +67% |
| Mamba2 + Triplet 30ep | 0.8873 | +83% |
| Mamba3 + Triplet 30ep | 0.8886 | +83.5% |

## Key Findings
- Mamba2 and Mamba3 both significantly outperform SSM placeholder (+83%)
- Triplet Loss outperformed NT-Xent on small dataset (7,703 samples)
- Mamba3 achieves marginally better overall gap (0.8886 vs 0.8873)
- Mamba3 provides better negative separation (-0.0021 vs -0.0003)
- Mamba3 is ~60% slower per epoch than Mamba2
- Rakhain remains lowest due to linguistic distance from Standard Bangla

## Next Steps
- Mamba1 baseline experiment
- BanglaBERT baseline experiment
- Retrain with new dataset
- RunPod A100 — Real Mamba2/Mamba3 final run
