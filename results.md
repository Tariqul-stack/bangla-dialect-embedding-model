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

## Experiment 6: Mamba1 + Triplet Loss (30 epochs)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.8871 |
| Negative Similarity | -0.0003 |
| Similarity Gap | 0.8873 |
| Best Val Loss | 0.2452 |
| Best Epoch | 27 |
| Parameters | 13,418,752 |

## Experiment 7: BanglaBERT + Triplet Loss (30 epochs)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.8671 |
| Negative Similarity | -0.0029 |
| Similarity Gap | 0.8700 |
| Best Val Loss | 0.2501 |
| Best Epoch | 30 |
| Parameters | 165,777,536 |

## Ablation Study — Loss Function Comparison (Mamba2)
| Loss Function | Gap | Val Loss | Winner |
|---------------|-----|----------|--------|
| NT-Xent | 0.8086 | 0.3816 | — |
| Triplet Loss | 0.8873 | 0.2452 |

## Architecture Comparison (Triplet Loss, 30 epochs)
| Architecture | Gap | Val Loss | Parameters | Speed |
|--------------|-----|----------|------------|-------|
| SSM Placeholder | 0.4843 | 0.1495 | — | — |
| BanglaBERT | 0.8700 | 0.2501 | 165.7M | ~93s/ep |
| Mamba1 | 0.8873 | 0.2452 | 13.4M | ~31s/ep |
| Mamba2 | 0.8873 | 0.2452 | 13.4M | ~31s/ep |
| Mamba3 | 0.8886 | 0.2440 | 13.4M | ~50s/ep |

## Per-dialect Similarity — Full Comparison
| Dialect | Placeholder | BanglaBERT | Mamba1 | Mamba2 | Mamba3 |
|---------|-------------|------------|--------|--------|--------|
| Standard | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Sylheti | 0.9453 | 0.8152 | 0.8923 | 0.8923 | 0.8807 |
| Barishal | 0.9571 | 0.8552 | 0.8848 | 0.8848 | 0.8870 |
| Chittagong | 0.9302 | 0.7802 | 0.8595 | 0.8595 | 0.8666 |
| Mymensingh | 0.9396 | 0.8086 | 0.8297 | 0.8297 | 0.8171 |
| Rangpur | 0.9189 | 0.6825 | 0.7855 | 0.7855 | 0.7853 |
| Rajshahi | 0.8746 | 0.7706 | 0.7211 | 0.7211 | 0.7196 |
| Rakhain | 0.8327 | 0.2554 | 0.3010 | 0.3010 | 0.3279 |

## Overall Progress
| Experiment | Gap | Improvement vs Baseline |
|------------|-----|------------------------|
| SSM Placeholder | 0.4843 | — |
| Mamba2 + NT-Xent 10ep | 0.7949 | +64% |
| Mamba2 + NT-Xent 30ep | 0.8086 | +67% |
| BanglaBERT + Triplet | 0.8700 | +80% |
| Mamba1 + Triplet | 0.8873 | +83% |
| Mamba2 + Triplet | 0.8873 | +83% |
| Mamba3 + Triplet | 0.8886 | +83.5% |

## Key Findings
- Mamba3 achieves best overall gap (0.8886) with best negative separation (-0.0021)
- All Mamba models outperform BanglaBERT despite being 12x smaller (13.4M vs 165.7M)
- Triplet Loss outperforms NT-Xent on current small dataset
- Mamba1 and Mamba2 show identical results in pure-PyTorch fallback mode
- Mamba3 is ~60% slower per epoch but achieves best results
- Rajshahi is the only dialect where BanglaBERT outperforms Mamba models
- Rakhain scores lowest due to linguistic distance from Standard Bangla

## Next Steps
- Gemma2 baseline experiment
- RunPod A100 — Real Mamba2/Mamba3 final run
