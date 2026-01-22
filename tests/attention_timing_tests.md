# Attention Layer Timing Tests

This document contains test cases for verifying timing calculations in attention layers, with special focus on how KV cache is handled differently in prefill vs decode phases.

## Overview

These tests verify:
- **Compute time**: FLOPs / peak throughput
- **Weight load time**: Weight memory / memory bandwidth
- **KV cache handling**: Computed in prefill, loaded in decode
- **Communication time**: TP all-reduce, CP softmax reduction
- **Wall clock time**: With and without overlap

## Key Insight: KV Cache in Prefill vs Decode

| Phase | KV Cache Behavior | Memory Impact |
|-------|-------------------|---------------|
| **Prefill** | KV cache is **computed** (Q, K, V projections) and **written** to HBM | Write bandwidth (often overlapped with compute) |
| **Decode** | KV cache is **read** from HBM to compute attention scores | Read bandwidth (often the bottleneck!) |

In decode phase, the time to load KV cache from memory often dominates because:
- Only 1 new token is processed (low compute)
- Entire KV cache must be read for attention (high memory)

## Conventions

**Timing Formulas (current implementation):**
```
compute_time_ms = flops_per_chip / (peak_pflops × 10^15) × 1000

load_time_ms:
    if phase == PREFILL:
        load_bytes = weight_memory_bytes
    else:  # DECODE
        load_bytes = weight_memory_bytes + kv_cache_bytes  # KV cache read during decode!
    load_time_ms = load_bytes / (memory_bw_gb_s × 10^9) × 1000

communication_time_ms = latency_us/1000 + comm_bytes / (interconnect_bw_gb_s × 10^9) × 1000

wall_clock_time_ms:
    # Memory load and compute can overlap (prefetching), so take max
    effective_compute = max(compute_time, load_time)

    if supports_overlap:
        wall_clock = max(effective_compute, communication_time)
    else:
        wall_clock = effective_compute + communication_time
```

**Test Hardware (simplified for verification):**
- Peak compute: **1.0 PFLOPs** (1000 TFLOPS)
- Memory bandwidth: **1000 GB/s** (1 TB/s)
- Interconnect bandwidth: **100 GB/s**
- Interconnect latency: **1.0 μs**

**Model Parameters (used across tests):**
- hidden_size `d` = 4096
- num_heads `h` = 32
- head_dim `dh` = 128 (d = h × dh)
- bytes_per_elem = 2 (BF16)

---

# Test ATT-TIME-1 — Prefill Phase: Compute-Dominated

Prefill processes the full input sequence. KV cache is computed (not loaded).

### Test case parameters

**Layer config:**
* hidden_size `d` = **4096**
* num_heads `h` = **32**
* head_dim `dh` = **128**
* batch_size `B` = **4**
* seq_len `S` = **2048**
* phase = **prefill**
* tensor_parallel = **1**
* bytes_per_elem = **2**

**Hardware config:**
* peak_tflops = **1000**
* memory_bandwidth_gb_s = **1000**
* interconnect_bandwidth_gb_s = **100**
* interconnect_latency_us = **1.0**
* supports_overlap = **false**

Derived:
* M = B × S = 4 × 2048 = **8192** tokens

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Q/K/V projections**
Each is `X[M,d] @ W[d,d]`:
* FLOPs per projection = `2 × M × d × d`
* = `2 × 8192 × 4096 × 4096 = 274,877,906,944`
* QKV total = `3 × 274,877,906,944 = 824,633,720,832`

**(b) Attention scores QK^T**
Per batch & head: `(S×dh) @ (dh×S)` → `2×S×S×dh`
* FLOPs = `2 × B × h × S × S × dh`
* = `2 × 4 × 32 × 2048 × 2048 × 128`
* = `2 × 4 × 32 × 4,194,304 × 128`
* = `137,438,953,472`

**(c) Apply attention to V**
Same as (b): `137,438,953,472`

**(d) Output projection**
* FLOPs = `2 × M × d × d = 274,877,906,944`

**Total FLOPs:**
* = `824,633,720,832 + 137,438,953,472 + 137,438,953,472 + 274,877,906,944`
* = **1,374,389,534,720** (≈1.37 TFLOPs)

#### 2) Compute time

```
compute_time_ms = flops / (peak_tflops × 10^12) × 1000
                = 1,374,389,534,720 / 10^15 × 1000
                = 1.374 ms
```

**compute_time_ms ≈ 1.374 ms**

#### 3) Weight memory

4 projection matrices (Wq, Wk, Wv, Wo), each `d × d`:
* Total params = `4 × d × d = 4 × 4096 × 4096 = 67,108,864`
* Bytes = `67,108,864 × 2 = 134,217,728` (128 MB)

**weight_memory_per_chip = 134,217,728 bytes**

#### 4) Weight load time

```
load_time_ms = 134,217,728 / 10^12 × 1000
                    = 0.134 ms
```

**load_time_ms ≈ 0.134 ms**

#### 5) KV cache (written, not loaded)

In prefill, KV cache is computed and written to memory:
* K cache: `B × h × S × dh = 4 × 32 × 2048 × 128 = 33,554,432` elements
* V cache: same
* Total = `2 × 33,554,432 = 67,108,864` elements
* Bytes = `67,108,864 × 2 = 134,217,728` (128 MB)

**kv_cache_per_chip = 134,217,728 bytes**

**Note:** In prefill, KV cache is written (not read), so it doesn't add to load time. Write bandwidth is typically overlapped with compute.

#### 6) Communication

Single chip: **communication_bytes = 0**

#### 7) Wall clock time

```
wall_clock_time_ms = compute_time_ms = 1.374 ms
```

**wall_clock_time_ms ≈ 1.374 ms**

#### 8) Analysis

Arithmetic intensity = FLOPs / bytes_loaded
= 1,374,389,534,720 / 134,217,728
= **10,240 FLOPs/byte**

This is highly **compute-bound**:
- Compute time: 1.374 ms
- Weight load time: 0.134 ms
- Ratio: 10.2× (compute dominates)

### Expected results

**Per-chip metrics**
* flops_per_chip: **1,374,389,534,720**
* weight_memory_per_chip: **134,217,728**
* kv_cache_per_chip: **134,217,728**

**Timing metrics**
* compute_time_ms: **≈1.374**
* load_time_ms: **≈0.134** (weights only)
* communication_time_ms: **0**
* wall_clock_time_ms: **≈1.374** (compute-bound: max(1.374, 0.134) = 1.374)

---

# Test ATT-TIME-2 — Decode Phase: Memory-Dominated (KV Cache Loading)

Decode processes 1 new token but must read the entire KV cache. This is typically memory-bound.

### Test case parameters

**Layer config:**
* hidden_size `d` = **4096**
* num_heads `h` = **32**
* head_dim `dh` = **128**
* batch_size `B` = **1**
* seq_len `S_past` = **2048** (past context length)
* phase = **decode**
* tensor_parallel = **1**
* bytes_per_elem = **2**

**Hardware config:**
* peak_tflops = **1000**
* memory_bandwidth_gb_s = **1000**
* interconnect_bandwidth_gb_s = **100**
* interconnect_latency_us = **1.0**
* supports_overlap = **false**

Derived:
* S_new = 1 (single new token)
* M_new = B × S_new = 1

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Q/K/V projections (for new token only)**
* Q: `2 × M_new × d × d = 2 × 1 × 4096 × 4096 = 33,554,432`
* K: `2 × 1 × 4096 × 4096 = 33,554,432`
* V: `2 × 1 × 4096 × 4096 = 33,554,432`
* Total QKV = `3 × 33,554,432 = 100,663,296`

**(b) Attention scores QK^T (new query vs all past keys)**
Query for 1 token attends to S_past + 1 keys:
* FLOPs = `2 × B × h × 1 × (S_past + 1) × dh`
* = `2 × 1 × 32 × 1 × 2049 × 128`
* = `16,785,408`

**(c) Apply attention to V**
* FLOPs = `2 × B × h × 1 × (S_past + 1) × dh = 16,785,408`

**(d) Output projection**
* FLOPs = `2 × M_new × d × d = 33,554,432`

**Total FLOPs:**
* = `100,663,296 + 16,785,408 + 16,785,408 + 33,554,432`
* = **167,788,544** (≈168 MFLOPs)

#### 2) Compute time

```
compute_time_ms = 167,788,544 / 10^15 × 1000
                = 0.000168 ms (168 nanoseconds)
```

**compute_time_ms ≈ 0.000168 ms**

#### 3) Weight memory

Same as prefill: **weight_memory_per_chip = 134,217,728 bytes** (128 MB)

#### 4) Weight load time

```
load_time_ms = 134,217,728 / 10^12 × 1000
                    = 0.134 ms
```

**load_time_ms ≈ 0.134 ms**

#### 5) KV cache (READ from memory!)

In decode, we must read the entire KV cache to compute attention:
* K cache: `B × h × S_past × dh = 1 × 32 × 2048 × 128 = 8,388,608` elements
* V cache: same
* Total = `2 × 8,388,608 = 16,777,216` elements
* Bytes = `16,777,216 × 2 = 33,554,432` (32 MB)

**kv_cache_per_chip = 33,554,432 bytes** (to be loaded)

#### 6) KV cache load time

```
kv_cache_load_time_ms = 33,554,432 / 10^12 × 1000
                      = 0.0336 ms
```

**kv_cache_load_time_ms ≈ 0.0336 ms**

#### 7) Total memory load time

In decode, both weights AND KV cache must be loaded:
```
total_load_time_ms = weight_load_time + kv_cache_load_time
                   = 0.134 + 0.0336
                   = 0.168 ms
```

**total_memory_load_time_ms ≈ 0.168 ms**

#### 8) Communication

Single chip: **communication_bytes = 0**

#### 9) Wall clock time (current implementation)

The current implementation only accounts for weight_load_time, not KV cache:
```
wall_clock_time_ms = max(compute_time, communication_time) or compute_time
                   = 0.000168 ms
```

**Note:** This underestimates actual decode time! A more accurate model would be:
```
actual_wall_clock ≈ max(compute_time, weight_load_time + kv_cache_load_time)
                  ≈ max(0.000168, 0.168)
                  ≈ 0.168 ms
```

#### 10) Analysis

Arithmetic intensity = FLOPs / bytes_loaded
= 167,788,544 / (134,217,728 + 33,554,432)
= 167,788,544 / 167,772,160
= **1.0 FLOPs/byte**

This is severely **memory-bound**:
- Compute time: 0.000168 ms (168 ns)
- Memory load time: 0.168 ms
- Ratio: 0.001× (memory dominates by 1000×)

### Expected results

**Per-chip metrics**
* flops_per_chip: **167,788,544**
* weight_memory_per_chip: **134,217,728**
* kv_cache_per_chip: **33,554,432**

**Timing metrics**
* compute_time_ms: **≈0.000168**
* load_time_ms: **≈0.168** (weights 0.134 + KV cache 0.034)
* communication_time_ms: **0**
* wall_clock_time_ms: **≈0.168** (memory-bound: max(0.000168, 0.168) = 0.168)

---

# Test ATT-TIME-3 — Decode with TP=4: Communication Overhead

Decode with tensor parallelism adds all-reduce communication.

### Test case parameters

**Layer config:**
* hidden_size `d` = **4096**
* num_heads `h` = **32**
* head_dim `dh` = **128**
* batch_size `B` = **1**
* seq_len `S_past` = **2048**
* phase = **decode**
* tensor_parallel `tp` = **4**
* bytes_per_elem = **2**

**Hardware config:**
* peak_tflops = **1000**
* memory_bandwidth_gb_s = **1000**
* interconnect_bandwidth_gb_s = **100**
* interconnect_latency_us = **1.0**
* supports_overlap = **false**

Derived:
* h_per_chip = h / tp = 32 / 4 = 8 heads per chip
* S_new = 1

### Expected calculations (step-by-step)

#### 1) FLOPs per chip

**(a) Projections (sharded by TP)**
* Q: `2 × 1 × d × (d/tp) = 2 × 1 × 4096 × 1024 = 8,388,608`
* K: `8,388,608`
* V: `8,388,608`
* O: `2 × 1 × (d/tp) × d = 8,388,608`
* Total projections = `4 × 8,388,608 = 33,554,432`

**(b) Attention (local heads only)**
* QK^T: `2 × B × h_per × 1 × (S_past + 1) × dh`
* = `2 × 1 × 8 × 1 × 2049 × 128 = 4,196,352`
* Apply V: `4,196,352`

**Total FLOPs per chip:**
* = `33,554,432 + 4,196,352 + 4,196,352`
* = **41,947,136**

#### 2) Compute time

```
compute_time_ms = 41,947,136 / 10^15 × 1000
                = 0.000042 ms (42 ns)
```

**compute_time_ms ≈ 0.000042 ms**

#### 3) Weight memory per chip

Weights sharded by TP:
* params_per_chip = `4 × d × d / tp = 4 × 4096 × 4096 / 4 = 16,777,216`
* Bytes = `16,777,216 × 2 = 33,554,432` (32 MB)

**weight_memory_per_chip = 33,554,432 bytes**

#### 4) Weight load time

```
load_time_ms = 33,554,432 / 10^12 × 1000
                    = 0.0336 ms
```

**load_time_ms ≈ 0.0336 ms**

#### 5) KV cache per chip (sharded by TP)

* K cache: `B × h_per × S_past × dh = 1 × 8 × 2048 × 128 = 2,097,152` elements
* V cache: same
* Total = `2 × 2,097,152 = 4,194,304` elements
* Bytes = `4,194,304 × 2 = 8,388,608` (8 MB)

**kv_cache_per_chip = 8,388,608 bytes**

#### 6) KV cache load time

```
kv_cache_load_time_ms = 8,388,608 / 10^12 × 1000
                      = 0.00839 ms
```

**kv_cache_load_time_ms ≈ 0.00839 ms**

#### 7) Communication bytes

TP all-reduce on output (1 token, full hidden dimension):
* Elements = `B × S_new × d = 1 × 1 × 4096 = 4,096`
* Bytes = `4,096 × 2 = 8,192`

**communication_bytes = 8,192**

#### 8) Communication time

```
latency_ms = 1.0 / 1000 = 0.001 ms
transfer_time_ms = 8,192 / 10^11 × 1000 = 0.000082 ms

communication_time_ms = 0.001 + 0.000082 = 0.00108 ms
```

**communication_time_ms ≈ 0.00108 ms**

#### 9) Wall clock time (no overlap)

```
wall_clock_time_ms = compute_time + communication_time
                   = 0.000042 + 0.00108
                   = 0.00112 ms
```

**wall_clock_time_ms ≈ 0.00112 ms** (current implementation)

#### 10) Analysis

With TP=4:
- Compute time: 0.000042 ms
- Weight load time: 0.0336 ms
- KV cache load time: 0.00839 ms
- Communication time: 0.00108 ms
- Total memory load: 0.0420 ms

The actual bottleneck is **memory loading** (weights + KV cache), not compute or communication.

### Expected results

**Per-chip metrics**
* flops_per_chip: **41,947,136**
* weight_memory_per_chip: **33,554,432**
* kv_cache_per_chip: **8,388,608**
* communication_bytes: **8,192**

**Timing metrics**
* compute_time_ms: **≈0.000042**
* load_time_ms: **≈0.0420** (weights 0.0336 + KV cache 0.00839)
* communication_time_ms: **≈0.00108**
* wall_clock_time_ms: **≈0.0431** (max(0.000042, 0.0420) + 0.00108)

---

# Test ATT-TIME-4 — Prefill with TP=4 and Overlap

Tests compute/communication overlap during prefill.

### Test case parameters

**Layer config:**
* hidden_size `d` = **4096**
* num_heads `h` = **32**
* head_dim `dh` = **128**
* batch_size `B` = **2**
* seq_len `S` = **1024**
* phase = **prefill**
* tensor_parallel `tp` = **4**
* bytes_per_elem = **2**

**Hardware config:**
* peak_tflops = **1000**
* memory_bandwidth_gb_s = **1000**
* interconnect_bandwidth_gb_s = **100**
* interconnect_latency_us = **1.0**
* **supports_overlap = true**

Derived:
* M = B × S = 2048
* h_per = 32 / 4 = 8

### Expected calculations (step-by-step)

#### 1) FLOPs per chip

**(a) Projections (sharded by TP)**
* Q/K/V: each `2 × M × d × (d/tp) = 2 × 2048 × 4096 × 1024 = 17,179,869,184`
* O: same
* Total = `4 × 17,179,869,184 = 68,719,476,736`

**(b) Attention (local heads only)**
* QK^T: `2 × B × h_per × S × S × dh = 2 × 2 × 8 × 1024 × 1024 × 128 = 4,294,967,296`
* Apply V: `4,294,967,296`

**Total FLOPs per chip:**
* = `68,719,476,736 + 4,294,967,296 + 4,294,967,296`
* = **77,309,411,328** (≈77.3 GFLOPs)

#### 2) Compute time

```
compute_time_ms = 77,309,411,328 / 10^15 × 1000
                = 0.0773 ms
```

**compute_time_ms ≈ 0.0773 ms**

#### 3) Weight memory per chip

* Bytes = `(4 × d × d / tp) × 2 = (4 × 4096 × 4096 / 4) × 2 = 33,554,432`

**weight_memory_per_chip = 33,554,432 bytes**

#### 4) Communication bytes

TP all-reduce on output:
* Elements = `B × S × d = 2 × 1024 × 4096 = 8,388,608`
* Bytes = `8,388,608 × 2 = 16,777,216`

**communication_bytes = 16,777,216**

#### 5) Communication time

```
latency_ms = 0.001 ms
transfer_time_ms = 16,777,216 / 10^11 × 1000 = 0.168 ms

communication_time_ms = 0.001 + 0.168 = 0.169 ms
```

**communication_time_ms ≈ 0.169 ms**

#### 6) Load time

```
load_time_ms = 33,554,432 / 10^12 × 1000
             = 0.0336 ms
```

**load_time_ms ≈ 0.0336 ms** (weights only, prefill)

#### 7) Wall clock time (WITH overlap)

```
effective_compute = max(compute_time, load_time)
                  = max(0.0773, 0.0336)
                  = 0.0773 ms

wall_clock_time_ms = max(effective_compute, communication_time)
                   = max(0.0773, 0.169)
                   = 0.169 ms
```

**wall_clock_time_ms ≈ 0.169 ms** (communication-bound)

#### 8) Comparison: Without overlap

```
wall_clock_no_overlap = effective_compute + communication_time
                      = 0.0773 + 0.169
                      = 0.246 ms
```

Overlap provides **31% improvement** (0.169 vs 0.246 ms).

### Expected results

**Per-chip metrics**
* flops_per_chip: **77,309,411,328**
* weight_memory_per_chip: **33,554,432**
* communication_bytes: **16,777,216**

**Timing metrics**
* compute_time_ms: **≈0.0773**
* load_time_ms: **≈0.0336** (weights only)
* communication_time_ms: **≈0.169**
* wall_clock_time_ms: **≈0.169** (with overlap: max(max(0.0773, 0.0336), 0.169))
* wall_clock_time_ms: **≈0.246** (without overlap: max(0.0773, 0.0336) + 0.169)

---

# Test ATT-TIME-5 — Long Context Decode (KV Cache Dominates)

Tests decode with very long context where KV cache loading is the dominant cost.

### Test case parameters

**Layer config:**
* hidden_size `d` = **4096**
* num_heads `h` = **32**
* head_dim `dh` = **128**
* batch_size `B` = **1**
* seq_len `S_past` = **32768** (32K context)
* phase = **decode**
* tensor_parallel = **1**
* bytes_per_elem = **2**

**Hardware config:**
* peak_tflops = **1000**
* memory_bandwidth_gb_s = **1000**
* interconnect_bandwidth_gb_s = **100**
* interconnect_latency_us = **1.0**
* supports_overlap = **false**

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Projections (for 1 new token)**
* QKV + O: `4 × 2 × 1 × 4096 × 4096 = 134,217,728`

**(b) Attention (against 32K+ tokens)**
* QK^T: `2 × 1 × 32 × 1 × 32769 × 128 = 268,443,648`
* Apply V: `268,443,648`

**Total FLOPs:**
* = `134,217,728 + 268,443,648 + 268,443,648`
* = **671,105,024** (≈671 MFLOPs)

#### 2) Compute time

```
compute_time_ms = 671,105,024 / 10^15 × 1000
                = 0.000671 ms
```

**compute_time_ms ≈ 0.000671 ms**

#### 3) Weight memory

**weight_memory_per_chip = 134,217,728 bytes** (128 MB)

#### 4) Weight load time

```
load_time_ms = 134,217,728 / 10^12 × 1000
                    = 0.134 ms
```

**load_time_ms ≈ 0.134 ms**

#### 5) KV cache (32K context!)

* K cache: `B × h × S_past × dh = 1 × 32 × 32768 × 128 = 134,217,728` elements
* V cache: same
* Total = `2 × 134,217,728 = 268,435,456` elements
* Bytes = `268,435,456 × 2 = 536,870,912` (512 MB!)

**kv_cache_per_chip = 536,870,912 bytes**

#### 6) KV cache load time

```
kv_cache_load_time_ms = 536,870,912 / 10^12 × 1000
                      = 0.537 ms
```

**kv_cache_load_time_ms ≈ 0.537 ms**

#### 7) Total memory load time

```
total_load_time_ms = weight_load_time + kv_cache_load_time
                   = 0.134 + 0.537
                   = 0.671 ms
```

**total_memory_load_time_ms ≈ 0.671 ms**

#### 8) Analysis

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Compute | 0.000671 | 0.1% |
| Weight load | 0.134 | 20.0% |
| KV cache load | 0.537 | **80.0%** |

**KV cache loading dominates** at long context lengths!

Arithmetic intensity = FLOPs / bytes_loaded
= 671,105,024 / (134,217,728 + 536,870,912)
= 671,105,024 / 671,088,640
= **1.0 FLOPs/byte**

### Expected results

**Per-chip metrics**
* flops_per_chip: **671,105,024**
* weight_memory_per_chip: **134,217,728**
* kv_cache_per_chip: **536,870,912**

**Timing metrics**
* compute_time_ms: **≈0.000671**
* load_time_ms: **≈0.671** (weights 0.134 + KV cache 0.537)
* wall_clock_time_ms: **≈0.671** (memory-bound: max(0.000671, 0.671) = 0.671)

**Key insight:** At 32K context, KV cache is 4× larger than weights and dominates decode time.

---

## Summary Table

| Test | Phase | TP | Context | Compute (ms) | Load (ms) | Comm (ms) | Wall Clock (ms) | Bottleneck |
|------|-------|----|---------|--------------|--------------------|-----------|-----------------|------------|
| ATT-TIME-1 | Prefill | 1 | 2048 | 1.374 | 0.134 | 0 | 1.374 | Compute |
| ATT-TIME-2 | Decode | 1 | 2048 | 0.000168 | 0.168 | 0 | 0.168 | Memory |
| ATT-TIME-3 | Decode | 4 | 2048 | 0.000042 | 0.042 | 0.00108 | 0.043 | Memory |
| ATT-TIME-4 | Prefill | 4 | 1024 | 0.0773 | 0.0336 | 0.169 | 0.246 | Communication |
| ATT-TIME-5 | Decode | 1 | 32K | 0.000671 | 0.671 | 0 | 0.671 | KV Cache (80%) |

---

## Implementation Note

The implementation correctly calculates timing as:

**Load time** (`load_time_ms`):
- **Prefill**: weights only (KV cache is written, not read)
- **Decode**: weights + KV cache (KV cache must be read for attention)

```python
load_bytes = weight_mem_per_package
if phase == Phase.DECODE and kv_cache_per_package > 0:
    load_bytes += kv_cache_per_package  # Add KV cache read during decode
load_time = (load_bytes / mem_bw) * 1000
```

**Wall clock time** (`wall_clock_time_ms`):
- Memory load and compute can overlap (prefetching), so take max
- Communication happens after compute (all-reduce patterns)

```python
effective_compute = max(compute_time, load_time)
if supports_overlap:
    wall_clock = max(effective_compute, comm_time)
else:
    wall_clock = effective_compute + comm_time
```

This accurately models memory-bound decode workloads where KV cache loading dominates.

---

## Verification

```python
from backend.layers import AttentionLayer
from backend.layers.base import Phase

# Hardware config matching the test specs
test_hardware = {
    "compute_per_package_PFlops": {"bf16": 1.0},  # 1 PFLOP = 1000 TFLOPS
    "memory_bandwidth_gb_s": 1000,
    "interconnect_bandwidth_gb_s": 100,
    "interconnect_latency_us": 1.0,
    "supports_overlap": False
}

# Test ATT-TIME-1: Prefill
layer = AttentionLayer(
    layer_idx=0,
    hidden_size=4096, num_heads=32, head_dim=128,
    dtype='bf16', parallelism={'tensor_parallel': 1}
)
metrics = layer.compute_metrics(
    batch_size=4, seq_len=2048, phase='prefill',
    hardware=test_hardware
)
print(f"ATT-TIME-1 (Prefill): compute={metrics.compute_time_ms:.6f} ms, "
      f"load={metrics.load_time_ms:.6f} ms (weights only), "
      f"kv_cache={metrics.kv_cache_per_package:,} bytes")

# Test ATT-TIME-2: Decode (load_time now includes KV cache!)
metrics = layer.compute_metrics(
    batch_size=1, seq_len=2048, phase='decode',
    hardware=test_hardware
)
print(f"ATT-TIME-2 (Decode): compute={metrics.compute_time_ms:.6f} ms, "
      f"load={metrics.load_time_ms:.6f} ms (weights + KV cache), "
      f"kv_cache={metrics.kv_cache_per_package:,} bytes")

# Test ATT-TIME-5: Long context decode
metrics = layer.compute_metrics(
    batch_size=1, seq_len=32768, phase='decode',
    hardware=test_hardware
)
print(f"ATT-TIME-5 (32K Decode): compute={metrics.compute_time_ms:.6f} ms, "
      f"load={metrics.load_time_ms:.6f} ms (weights + KV cache), "
      f"kv_cache={metrics.kv_cache_per_package:,} bytes")
```
