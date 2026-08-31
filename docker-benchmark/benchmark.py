import time
import numpy as np
import onnxruntime as ort
import psutil
import os

dummy_ids = np.random.randint(1, 101975, (1, 128)).astype(np.int64)
dummy_mask = np.ones((1, 128), dtype=np.int64)

def benchmark(model_path, name, n_runs=200):
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    for _ in range(10):
        sess.run(None, {'input_ids': dummy_ids, 'attention_mask': dummy_mask})
    
    times = []
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / 1e6
    
    for _ in range(n_runs):
        start = time.perf_counter()
        sess.run(None, {'input_ids': dummy_ids, 'attention_mask': dummy_mask})
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    ram_after = process.memory_info().rss / 1e6
    avg = np.mean(times)
    p95 = np.percentile(times, 95)
    size = os.path.getsize(model_path) / 1e6
    ram = ram_after - ram_before
    
    print(f"{name}")
    print(f"  Size: {size:.1f}MB | Avg: {avg:.1f}ms | P95: {p95:.1f}ms | RAM: {ram:.1f}MB")
    return avg, p95, size, ram

print("=" * 55)
print("Docker CPU Benchmark")
print("=" * 55)
benchmark('/models/teacher_fp32.onnx', 'Teacher FP32 (Mamba3, 13.4M)')
benchmark('/models/student_fp32.onnx', 'Student FP32 (KD, 6.6M)')
