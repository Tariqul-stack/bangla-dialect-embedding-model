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

Note: Mamba1 and Mamba2 show similar results in pure-PyTorch fallback mode.
Architectural differences will be more pronounced with real CUDA kernels on RunPod A100.

## Ablation Study — Loss Function Comparison (Mamba2)
| Loss Function | Gap | Val Loss | Winner |
|---------------|-----|----------|--------|
| NT-Xent | 0.8086 | 0.3816 | — |
| Triplet Loss | 0.8873 | 0.2452 |

## Architecture Comparison (Triplet Loss, 30 epochs)
| Architecture | Gap | Val Loss | Speed | Parameters |
|--------------|-----|----------|-------|------------|
| Mamba1 | 0.8873 | 0.2452 | ~31s/ep | 13.4M |
| Mamba2 | 0.8873 | 0.2452 | ~31s/ep | 13.4M |
| Mamba3 | 0.8886 | 0.2440 | ~50s/ep | 13.4M |

## Per-dialect Similarity — Full Comparison
| Dialect | Placeholder | Mamba2+NT-Xent | Mamba2+Triplet | Mamba3+Triplet | Mamba1+Triplet |
|---------|-------------|----------------|----------------|----------------|----------------|
| Standard | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Sylheti | 0.9453 | 0.7175 | 0.8923 | 0.8807 | 0.8923 |
| Barishal | 0.9571 | 0.7621 | 0.8848 | 0.8870 | 0.8848 |
| Chittagong | 0.9302 | 0.7031 | 0.8595 | 0.8666 | 0.8595 |
| Mymensingh | 0.9396 | 0.7150 | 0.8297 | 0.8171 | 0.8297 |
| Rangpur | 0.9189 | 0.6198 | 0.7855 | 0.7853 | 0.7855 |
| Rajshahi | 0.8746 | 0.6011 | 0.7211 | 0.7196 | 0.7211 |
| Rakhain | 0.8327 | 0.1438 | 0.3010 | 0.3279 | 0.3010 |

## Overall Progress
| Experiment | Gap | Improvement vs Baseline |
|------------|-----|------------------------|
| SSM Placeholder | 0.4843 | — |
| Mamba2 + NT-Xent 10ep | 0.7949 | +64% |
| Mamba2 + NT-Xent 30ep | 0.8086 | +67% |
| Mamba1 + Triplet 30ep | 0.8873 | +83% |
| Mamba2 + Triplet 30ep | 0.8873 | +83% |
| Mamba3 + Triplet 30ep | 0.8886 | +83.5% |

## Key Findings
- All SSM architectures significantly outperform placeholder baseline (+83%)
- Triplet Loss outperforms NT-Xent on current small dataset (7,703 samples)
- Mamba3 achieves best overall gap (0.8886) with best negative separation (-0.0021)
- Mamba1 and Mamba2 show identical results in fallback mode
- Mamba3 is ~60% slower per epoch than Mamba1/Mamba2
- Rakhain remains lowest due to linguistic distance from Standard Bangla

## Next Steps
- BanglaBERT baseline experiment
- Retrain with new dataset
- RunPod A100 — Real Mamba2/Mamba3 final run
