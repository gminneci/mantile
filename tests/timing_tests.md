# Layer Timing Tests

This document contains test cases for verifying the timing calculations in layer implementations: compute time, weight load time, communication time, and wall clock time.

## Overview

These tests verify the correctness of:
- **Compute time**: Time to execute FLOPs based on peak hardware throughput
- **Weight load time**: Time to load weights from HBM based on memory bandwidth
- **Communication time**: Time for inter-chip communication (latency + transfer)
- **Wall clock time**: Total layer execution time accounting for overlap

All tests use simplified hardware specs with round numbers for easy manual verification.

## Conventions Used in All Tests

**Time Calculation Formulas:**

```
compute_time_ms = (flops_per_chip / peak_flops) * 1000
                = (flops_per_chip / (peak_tflops * 10^12)) * 1000

weight_load_time_ms = (weight_memory_bytes / memory_bandwidth) * 1000
                    = (weight_memory_bytes / (memory_bw_gb_s * 10^9)) * 1000

communication_time_ms = latency_ms + transfer_time_ms
                      = (latency_us / 1000) + (comm_bytes / (interconnect_bw_gb_s * 10^9)) * 1000

wall_clock_time_ms:
  if supports_overlap:
      wall_clock_time_ms = max(compute_time_ms, communication_time_ms)
  else:
      wall_clock_time_ms = compute_time_ms + communication_time_ms
```

**Test Hardware Specs (simplified for easy calculation):**

For these tests, we define a hypothetical hardware configuration:
- **Peak compute**: 1000 TFLOPS (10^15 FLOPS) = 1 PFLOP
- **Memory bandwidth**: 1000 GB/s (10^12 bytes/s) = 1 TB/s
- **Interconnect bandwidth**: 100 GB/s (10^11 bytes/s)
- **Interconnect latency**: 1.0 microseconds

These round numbers make manual verification straightforward.

**Memory Conventions:**
- `bytes_per_elem = 2` (BF16/FP16)

---

# Test 0 — Compute-bound layer (simple MLP, single chip)

This test verifies compute time calculation for a compute-bound workload where the arithmetic intensity is high and compute time dominates.

### Test case parameters

**Layer config:**
* Layer type: MLP (2-projection FFN)
* hidden_size (`d`) = **4096**
* intermediate_size (`d_ff`) = **16384**
* batch_size (`B`) = **8**
* seq_len (`S`) = **1024**
* num_chips = **1**
* tensor_parallel = **1**
* bytes_per_elem = **2**

**Hardware config (simplified):**
* peak_tflops = **1000** (1 PFLOP)
* memory_bandwidth_gb_s = **1000** (1 TB/s)
* interconnect_bandwidth_gb_s = **100**
* interconnect_latency_us = **1.0**
* supports_overlap = **false**

Derived:
* M = B × S = 8 × 1024 = **8192** tokens

### Expected calculations (step-by-step)

**1) FLOPs calculation**

* GEMM1: `x[M,d] @ W1[d,d_ff]`
  FLOPs = `2 × M × d × d_ff`
  = `2 × 8192 × 4096 × 16384`
  = **1,099,511,627,776** (≈1.1 TFLOPs)

* GEMM2: `h[M,d_ff] @ W2[d_ff,d]`
  FLOPs = `2 × M × d_ff × d`
  = `2 × 8192 × 16384 × 4096`
  = **1,099,511,627,776** (≈1.1 TFLOPs)

* **flops_per_chip = 2,199,023,255,552** (≈2.2 TFLOPs)

**2) Compute time**

```
compute_time_ms = (flops_per_chip / peak_flops) × 1000
                = (2,199,023,255,552 / (1000 × 10^12)) × 1000
                = (2,199,023,255,552 / 10^15) × 1000
                = 2.199... × 1000
                = 2.199 ms
```

**compute_time_ms ≈ 2.199 ms**

**3) Weight memory**

* `|W1| = d × d_ff = 4096 × 16384 = 67,108,864` elements
* `|W2| = d_ff × d = 16384 × 4096 = 67,108,864` elements
* Total elements = **134,217,728**
* Total bytes = `134,217,728 × 2 = 268,435,456` bytes (256 MB)

**weight_memory_per_chip = 268,435,456 bytes**

**4) Weight load time**

```
weight_load_time_ms = (weight_bytes / memory_bandwidth) × 1000
                    = (268,435,456 / (1000 × 10^9)) × 1000
                    = (268,435,456 / 10^12) × 1000
                    = 0.000268... × 1000
                    = 0.268 ms
```

**weight_load_time_ms ≈ 0.268 ms**

**5) Communication**

* Single chip, no parallelism: **communication_bytes = 0**
* **communication_time_ms = 0** (or None)

**6) Wall clock time**

No communication, so:
```
wall_clock_time_ms = compute_time_ms = 2.199 ms
```

**wall_clock_time_ms ≈ 2.199 ms**

**7) Analysis: Compute vs Memory bound**

Arithmetic intensity = FLOPs / Bytes loaded
= 2,199,023,255,552 / 268,435,456
= **8192 FLOPs/byte**

This is highly compute-bound. For reference:
- Compute time: 2.199 ms
- Weight load time: 0.268 ms
- Ratio: 8.2× (compute takes 8× longer than memory)

### Expected results

**Per-chip metrics**
* flops_per_chip: **2,199,023,255,552**
* weight_memory_per_chip: **268,435,456**

**Timing metrics**
* compute_time_ms: **≈2.199**
* weight_load_time_ms: **≈0.268**
* communication_time_ms: **0** (or None)
* wall_clock_time_ms: **≈2.199**

---

# Test 1 — Memory-bound layer (decode phase, small batch)

This test verifies timing for a memory-bound workload typical of decode phase with small batch size (batch=1, seq_len=1).

### Test case parameters

**Layer config:**
* Layer type: MLP (2-projection FFN)
* hidden_size (`d`) = **4096**
* intermediate_size (`d_ff`) = **16384**
* batch_size (`B`) = **1**
* seq_len (`S`) = **1** (decode: single token)
* num_chips = **1**
* tensor_parallel = **1**
* bytes_per_elem = **2**

**Hardware config (simplified):**
* peak_tflops = **1000**
* memory_bandwidth_gb_s = **1000**
* interconnect_bandwidth_gb_s = **100**
* interconnect_latency_us = **1.0**
* supports_overlap = **false**

Derived:
* M = B × S = 1 × 1 = **1** token

### Expected calculations (step-by-step)

**1) FLOPs calculation**

* GEMM1 FLOPs = `2 × 1 × 4096 × 16384 = 134,217,728`
* GEMM2 FLOPs = `2 × 1 × 16384 × 4096 = 134,217,728`
* **flops_per_chip = 268,435,456** (≈268 MFLOPs)

**2) Compute time**

```
compute_time_ms = (268,435,456 / 10^15) × 1000
                = 0.000268... ms
                ≈ 0.000268 ms (268 nanoseconds)
```

**compute_time_ms ≈ 0.000268 ms**

**3) Weight memory**

Same as Test 0 (weights don't change with batch size):
* **weight_memory_per_chip = 268,435,456 bytes** (256 MB)

**4) Weight load time**

Same as Test 0:
* **weight_load_time_ms ≈ 0.268 ms**

**5) Communication**

* **communication_bytes = 0**
* **communication_time_ms = 0**

**6) Wall clock time**

```
wall_clock_time_ms = compute_time_ms = 0.000268 ms
```

Note: In practice, memory-bound layers would use `max(compute_time, weight_load_time)` for effective time, but the current implementation returns just compute_time for wall_clock.

**wall_clock_time_ms ≈ 0.000268 ms**

**7) Analysis: Memory-bound**

Arithmetic intensity = 268,435,456 / 268,435,456 = **1 FLOP/byte**

This is severely memory-bound:
- Compute time: 0.000268 ms
- Weight load time: 0.268 ms
- Ratio: 0.001× (memory takes 1000× longer than compute)

In real systems, decode is bottlenecked by memory bandwidth, not compute.

### Expected results

**Per-chip metrics**
* flops_per_chip: **268,435,456**
* weight_memory_per_chip: **268,435,456**

**Timing metrics**
* compute_time_ms: **≈0.000268**
* weight_load_time_ms: **≈0.268**
* communication_time_ms: **0**
* wall_clock_time_ms: **≈0.000268**

---

# Test 2 — Communication time with Tensor Parallelism (TP=4)

This test verifies communication time calculation including both latency and bandwidth components.

### Test case parameters

**Layer config:**
* Layer type: MLP (2-projection FFN)
* hidden_size (`d`) = **4096**
* intermediate_size (`d_ff`) = **16384**
* batch_size (`B`) = **4**
* seq_len (`S`) = **512**
* tensor_parallel `tp` = **4**
* num_chips = **4**
* bytes_per_elem = **2**

**Hardware config (simplified):**
* peak_tflops = **1000**
* memory_bandwidth_gb_s = **1000**
* interconnect_bandwidth_gb_s = **100**
* interconnect_latency_us = **10.0** (larger latency to show its effect)
* supports_overlap = **false**

Derived:
* M = B × S = 4 × 512 = **2048** tokens

### Expected calculations (step-by-step)

**1) FLOPs per chip (with TP=4)**

Total FLOPs (same math as before):
* Total GEMM FLOPs = `2 × (2 × M × d × d_ff)`
  = `4 × 2048 × 4096 × 16384`
  = **549,755,813,888**

Per chip (TP shards the work):
* **flops_per_chip = 549,755,813,888 / 4 = 137,438,953,472**

**2) Compute time**

```
compute_time_ms = (137,438,953,472 / 10^15) × 1000
                = 0.137... ms
                ≈ 0.137 ms
```

**compute_time_ms ≈ 0.137 ms**

**3) Communication bytes**

TP all-reduce on output `y` of shape `(M × d)`:
* Elements = `M × d = 2048 × 4096 = 8,388,608`
* Bytes = `8,388,608 × 2 = 16,777,216` bytes (16 MB)

**communication_bytes = 16,777,216**

**4) Communication time**

```
latency_ms = interconnect_latency_us / 1000
           = 10.0 / 1000
           = 0.01 ms

transfer_time_ms = (comm_bytes / interconnect_bw) × 1000
                 = (16,777,216 / (100 × 10^9)) × 1000
                 = (16,777,216 / 10^11) × 1000
                 = 0.000168... × 1000
                 = 0.168 ms

communication_time_ms = latency_ms + transfer_time_ms
                      = 0.01 + 0.168
                      = 0.178 ms
```

**communication_time_ms ≈ 0.178 ms**

**5) Wall clock time (no overlap)**

```
wall_clock_time_ms = compute_time_ms + communication_time_ms
                   = 0.137 + 0.178
                   = 0.315 ms
```

**wall_clock_time_ms ≈ 0.315 ms (without overlap)**

### Expected results

**Per-chip metrics**
* flops_per_chip: **137,438,953,472**
* communication_bytes: **16,777,216**

**Timing metrics**
* compute_time_ms: **≈0.137**
* communication_time_ms: **≈0.178** (latency: 0.01, transfer: 0.168)
* wall_clock_time_ms: **≈0.315** (sum, no overlap)

---

# Test 3 — Wall clock time with compute/communication overlap

This test verifies that wall clock time correctly uses `max()` when hardware supports overlap.

### Test case parameters

Same as Test 2, but with overlap enabled:

**Layer config:**
* hidden_size = **4096**, intermediate_size = **16384**
* batch_size = **4**, seq_len = **512**
* tensor_parallel = **4**, num_chips = **4**
* bytes_per_elem = **2**

**Hardware config:**
* peak_tflops = **1000**
* memory_bandwidth_gb_s = **1000**
* interconnect_bandwidth_gb_s = **100**
* interconnect_latency_us = **10.0**
* **supports_overlap = true** (key difference)

### Expected calculations

From Test 2:
* compute_time_ms ≈ 0.137 ms
* communication_time_ms ≈ 0.178 ms

**Wall clock time (with overlap):**

```
wall_clock_time_ms = max(compute_time_ms, communication_time_ms)
                   = max(0.137, 0.178)
                   = 0.178 ms
```

**wall_clock_time_ms ≈ 0.178 ms (with overlap)**

This represents a **43% improvement** over the non-overlapped case (0.178 vs 0.315).

### Expected results

**Timing metrics**
* compute_time_ms: **≈0.137**
* communication_time_ms: **≈0.178**
* wall_clock_time_ms: **≈0.178** (max, with overlap)
* compute_communication_overlap: **true**

---

# Test 4 — Communication-dominated with high latency

This test verifies timing when communication latency dominates (many small messages or high-latency interconnect).

### Test case parameters

**Layer config:**
* Layer type: MLP
* hidden_size = **1024**
* intermediate_size = **4096**
* batch_size = **1**
* seq_len = **32**
* tensor_parallel = **8**
* num_chips = **8**
* bytes_per_elem = **2**

**Hardware config:**
* peak_tflops = **1000**
* memory_bandwidth_gb_s = **1000**
* interconnect_bandwidth_gb_s = **100**
* **interconnect_latency_us = 100.0** (high latency scenario)
* supports_overlap = **false**

Derived:
* M = 1 × 32 = 32 tokens

### Expected calculations

**1) FLOPs per chip**

Total FLOPs = `4 × M × d × d_ff = 4 × 32 × 1024 × 4096 = 536,870,912`

Per chip (TP=8):
* **flops_per_chip = 536,870,912 / 8 = 67,108,864**

**2) Compute time**

```
compute_time_ms = (67,108,864 / 10^15) × 1000
                ≈ 0.000067 ms
```

**3) Communication bytes**

All-reduce payload: `M × d = 32 × 1024 = 32,768` elements = **65,536 bytes**

**4) Communication time**

```
latency_ms = 100.0 / 1000 = 0.1 ms

transfer_time_ms = (65,536 / 10^11) × 1000
                 = 0.000655... ms
                 ≈ 0.000656 ms

communication_time_ms = 0.1 + 0.000656 = 0.101 ms
```

**communication_time_ms ≈ 0.101 ms** (latency-dominated: 99% is latency)

**5) Wall clock time**

```
wall_clock_time_ms = 0.000067 + 0.101 ≈ 0.101 ms
```

### Expected results

**Timing metrics**
* compute_time_ms: **≈0.000067**
* communication_time_ms: **≈0.101** (latency: 0.1, transfer: 0.0007)
* wall_clock_time_ms: **≈0.101**

**Analysis:** With 100μs latency, the interconnect latency dominates even for moderate message sizes. This shows why reducing latency is critical for small-batch inference.

---

## Verification

To verify these tests, pass a hardware dict with the simplified specs to the layer's `compute_metrics()` method:

```python
from backend.layers import MLPLayer

# Simplified hardware config for testing
test_hardware = {
    "compute_tflops": 1000,           # 1 PFLOP
    "memory_bandwidth_gb_s": 1000,    # 1 TB/s
    "interconnect_bandwidth_gb_s": 100,
    "interconnect_latency_us": 1.0,
    "supports_overlap": False
}

# Test 0: Compute-bound large batch
layer = MLPLayer(
    name='mlp', layer_idx=0,
    hidden_size=4096, intermediate_size=16384,
    parallelism={'tensor_parallel': 1}
)
metrics = layer.compute_metrics(
    batch_size=8, seq_len=1024, phase='prefill', dtype='bf16',
    hardware=test_hardware
)
print(f"Test 0: compute_time={metrics.compute_time_ms:.6f} ms, "
      f"weight_load_time={metrics.weight_load_time_ms:.6f} ms")

# Test 1: Memory-bound decode
metrics = layer.compute_metrics(
    batch_size=1, seq_len=1, phase='decode', dtype='bf16',
    hardware=test_hardware
)
print(f"Test 1: compute_time={metrics.compute_time_ms:.6f} ms, "
      f"weight_load_time={metrics.weight_load_time_ms:.6f} ms")

# Test 2: Communication with TP=4 (no overlap)
test_hardware_tp = {**test_hardware, "interconnect_latency_us": 10.0}
layer_tp = MLPLayer(
    name='mlp', layer_idx=0,
    hidden_size=4096, intermediate_size=16384,
    parallelism={'tensor_parallel': 4}
)
metrics = layer_tp.compute_metrics(
    batch_size=4, seq_len=512, phase='prefill', dtype='bf16',
    hardware=test_hardware_tp
)
print(f"Test 2: compute={metrics.compute_time_ms:.6f} ms, "
      f"comm={metrics.communication_time_ms:.6f} ms, "
      f"wall_clock={metrics.wall_clock_time_ms:.6f} ms")

# Test 3: With overlap
test_hardware_overlap = {**test_hardware_tp, "supports_overlap": True}
metrics = layer_tp.compute_metrics(
    batch_size=4, seq_len=512, phase='prefill', dtype='bf16',
    hardware=test_hardware_overlap
)
print(f"Test 3: wall_clock={metrics.wall_clock_time_ms:.6f} ms (with overlap)")
```

## Summary

| Test | Scenario | Compute (ms) | Comm (ms) | Wall Clock (ms) | Bottleneck |
|------|----------|--------------|-----------|-----------------|------------|
| 0 | Large batch, single chip | 2.199 | 0 | 2.199 | Compute |
| 1 | Decode (B=1, S=1) | 0.000268 | 0 | 0.000268 | Memory* |
| 2 | TP=4, no overlap | 0.137 | 0.178 | 0.315 | Communication |
| 3 | TP=4, with overlap | 0.137 | 0.178 | 0.178 | Communication |
| 4 | High latency (100μs) | 0.000067 | 0.101 | 0.101 | Latency |

*Test 1 shows compute_time, but real decode is memory-bound; the implementation's wall_clock reflects compute only.
