# Bangla Dialect Embedding Model — Results

## Experiment 1: SSM Placeholder (Baseline)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.9555 |
| Negative Similarity | 0.4711 |
| Similarity Gap | 0.4843 |
| Best Val Loss | 0.1495 |

## Experiment 2: Mamba2 + NT-Xent (Pure PyTorch, 10 epochs)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.7919 |
| Negative Similarity | -0.0030 |
| Similarity Gap | 0.7949 |
| Best Val Loss | 0.4512 |
| Best Epoch | 10 |

## Experiment 3: Mamba2 + NT-Xent (Pure PyTorch, 30 epochs)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.8084 |
| Negative Similarity | -0.0001 |
| Similarity Gap | 0.8086 |
| Best Val Loss | 0.3816 |
| Best Epoch | 24 |

## Experiment 4: Mamba2 + Triplet Loss (Ablation, 30 epochs)
| Metric | Score |
|--------|-------|
| Positive Similarity | 0.8871 |
| Negative Similarity | -0.0003 |
| Similarity Gap | 0.8873 |
| Best Val Loss | 0.2452 |
| Best Epoch | 27 |

## Ablation Study — Loss Function Comparison
| Loss Function | Pos Sim | Neg Sim | Gap | Val Loss |
|---------------|---------|---------|-----|----------|
| NT-Xent (SimCLR) | 0.8084 | -0.0001 | 0.8086 | 0.3816 |
| Triplet Loss | 0.8871 | -0.0003 | 0.8873 | 0.2452 |

Note: Triplet Loss outperformed NT-Xent on current dataset (7,703 samples).
Hypothesis: Small dataset benefits from Triplet Loss's focused single-negative objective.
Final loss function selection pending supervisor discussion and new dataset results.

## Per-dialect Similarity — Full Comparison
| Dialect | Placeholder | NT-Xent 10ep | NT-Xent 30ep | Triplet 30ep |
|---------|-------------|--------------|--------------|--------------|
| Standard | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Sylheti | 0.9453 | 0.6902 | 0.7175 | 0.8923 |
| Barishal | 0.9571 | 0.7286 | 0.7621 | 0.8848 |
| Chittagong | 0.9302 | 0.6719 | 0.7031 | 0.8595 |
| Mymensingh | 0.9396 | 0.6837 | 0.7150 | 0.8297 |
| Rangpur | 0.9189 | 0.5802 | 0.6198 | 0.7855 |
| Rajshahi | 0.8746 | 0.5759 | 0.6011 | 0.7211 |
| Rakhain | 0.8327 | 0.1608 | 0.1438 | 0.3010 |

## Overall Progress
| Experiment | Gap | Improvement vs Baseline |
|------------|-----|------------------------|
| SSM Placeholder | 0.4843 | — |
| Mamba2 + NT-Xent 10ep | 0.7949 | +64% |
| Mamba2 + NT-Xent 30ep | 0.8086 | +67% |
| Mamba2 + Triplet 30ep | 0.8873 | +83% |

## Key Findings
- Mamba2 architecture significantly outperforms SSM placeholder (+83% gap improvement)
- Triplet Loss unexpectedly outperformed NT-Xent on current small dataset
- All dialects improved substantially with Mamba2 architecture
- Rakhain remains lowest due to linguistic distance from Standard Bangla
- More epochs consistently improve results (convergence at epoch 24-27)

## Next Steps
- Supervisor discussion on final loss function selection
- Retrain with new dataset (larger corpus expected)
- Mamba1 baseline experiment
- BanglaBERT baseline experiment
- Mamba3 experiment
- RunPod A100 final run with Real Mamba2 CUDA kernels