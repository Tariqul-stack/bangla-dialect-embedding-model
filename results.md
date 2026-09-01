## Hardware Deployment Benchmark

### Model Compression
| Model | Size (FP32) | Size (INT8) | Parameters |
|-------|-------------|-------------|------------|
| Teacher (Mamba3) | 161.4 MB | 52.9 MB | 13.4M |
| Student (KD) | 79.8 MB | 26.4 MB | 6.6M |

Student vs Teacher: 50% smaller parameters, 1.2% gap reduction (0.8886 → 0.8782)

### Platform ① Kaggle CPU-only
| Model | Format | Avg (ms) | P95 (ms) |
|-------|--------|----------|----------|
| Teacher | ONNX FP32 | 41.1 | 48.5 |
| Teacher | PyTorch INT8 | 34.1 | 36.2 |
| Student | ONNX FP32 | 9.3 | 9.9 |
| Student | PyTorch INT8 | 13.0 | 14.2 |

### Platform ② Docker Simulation
| Config | Model | Avg (ms) | P95 (ms) |
|--------|-------|----------|----------|
| RPi5-class (4c/4GB) | Teacher | 74.5 | 98.7 |
| RPi5-class (4c/4GB) | Student | 16.2 | 61.2 |
| IoT-class (2c/2GB) | Teacher | 208.6 | 286.0 |
| IoT-class (2c/2GB) | Student | 49.5 | 93.6 |
| Edge server (8c/8GB) | Teacher | 37.5 | 49.1 |
| Edge server (8c/8GB) | Student | 8.9 | 13.3 |

### Platform ③ Apple M1 (Real NPU Hardware)
| Compute Mode | Avg (ms) | P95 (ms) |
|--------------|----------|----------|
| CPU Only | 7.08 | 7.63 |
| CPU + GPU | 4.73 | 5.37 |
| CPU + GPU + Neural Engine | 4.82 | 5.40 |


## Retrieval Evaluation — Recall@K (Mamba3 + Triplet, 30 epochs)
| Metric | Score |
|--------|-------|
| Recall@1 | 0.5590 (55.9%) |
| Recall@5 | 0.7160 (71.6%) |
| Recall@10 | 0.7795 (78.0%) |
| MRR | 0.6340 |

Recall@K measures whether the correct Standard Bangla match
is found within the top-K most similar candidates out of all
771 test set candidates — simulating a real retrieval scenario.

### Key Hardware Findings
- Student model (4MB-27MB range) runs efficiently on all tested platforms
- Real-time capable even on 2-core/2GB IoT-class constrained hardware (49.5ms)
- Apple M1 achieves sub-5ms inference with GPU/ANE acceleration
- For this model scale, GPU acceleration matches or exceeds ANE performance
- Complete offline deployment feasible with no cloud dependency