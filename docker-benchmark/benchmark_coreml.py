import coremltools as ct
import numpy as np
import time

model_path = '/Users/tariqulislam/bangla-dialect-models/student_coreml.mlpackage'

dummy_ids = np.random.randint(1, 101975, (1, 128)).astype(np.int32)
dummy_mask = np.ones((1, 128), dtype=np.int32)

def benchmark(compute_units, name, n_runs=200):
    model = ct.models.MLModel(model_path, compute_units=compute_units)

    # Warmup
    for _ in range(10):
        model.predict({'input_ids': dummy_ids, 'attention_mask': dummy_mask})

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict({'input_ids': dummy_ids, 'attention_mask': dummy_mask})
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg = np.mean(times)
    p95 = np.percentile(times, 95)
    print(f"{name}")
    print(f"  Avg latency: {avg:.2f} ms")
    print(f"  P95 latency: {p95:.2f} ms")
    return avg, p95

print("=" * 55)
print("Apple M1 Benchmark — Student Model (CoreML)")
print("=" * 55)

# CPU only
r1 = benchmark(ct.ComputeUnit.CPU_ONLY, "CPU Only")
print()

# CPU + GPU
r2 = benchmark(ct.ComputeUnit.CPU_AND_GPU, "CPU + GPU")
print()

# CPU + GPU + ANE (all)
r3 = benchmark(ct.ComputeUnit.ALL, "CPU + GPU + Neural Engine (ANE)")

print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"{'Mode':<35} {'Avg(ms)':>10} {'P95(ms)':>10}")
print("-" * 55)
print(f"{'CPU Only':<35} {r1[0]:>9.2f} {r1[1]:>9.2f}")
print(f"{'CPU + GPU':<35} {r2[0]:>9.2f} {r2[1]:>9.2f}")
print(f"{'CPU + GPU + ANE':<35} {r3[0]:>9.2f} {r3[1]:>9.2f}")