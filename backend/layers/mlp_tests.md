
# MLP Layer Tests

This document contains test cases for the MLP (MLPLayer) implementation with detailed calculations and expected results.

## Overview

These tests verify the correctness of:
- **FLOPs computation**: Matrix multiply operations accounting for parallelism
- **Weight memory**: Parameter storage per chip and in aggregate
- **Activation memory**: Intermediate tensor buffers during forward pass
- **Communication**: Inter-chip data transfer requirements (TP, SP)

All tests are verified against the current implementation and can be run using the code in the [Verification](#verification) section.

## Test Summary

| Test | Config | Hidden | Intermediate | Batch | Seq Len | FLOPs/chip | Weight/chip | Activation/chip |
|------|--------|--------|--------------|-------|---------|------------|-------------|-----------------|
| 0 | No parallelism | 16 | 64 | 4 | 8 | 131,072 | 4,096 | 9,216 |
| 1 | TP=4 | 1024 | 4096 | 2 | 128 | 1,073,741,824 | 16,777,216 | 1,572,864 |
| 2 | SP=4 | 1024 | 4096 | 2 | 128 | 1,073,741,824 | 16,777,216 | 1,179,648 |
| 3 | TP=4, SP=2 | 1024 | 4096 | 2 | 128 | 536,870,912 | 16,777,216 | 786,432 |

## Conventions Used in All Tests

**Model:** Standard 2-projection FFN/MLP block
```
h = activation(x @ W1)
y = h @ W2
```

**Shapes:**
- Input: `x ∈ ℝ^(M × d)` where `M = batch_size × seq_len`
- Weight matrices: `W1 ∈ ℝ^(d × d_ff)`, `W2 ∈ ℝ^(d_ff × d)`
- Intermediate: `h ∈ ℝ^(M × d_ff)`
- Output: `y ∈ ℝ^(M × d)`

**FLOPs Calculation:** 
Matrix multiply `(M × K) @ (K × N)` = `2MKN` FLOPs

**Memory Conventions:**
- `bytes_per_elem = 2` (BF16/FP16)
- `weight_memory_per_chip`: Total weights on one chip (currently NOT sharded by TP in implementation)
- `activation_memory_per_chip`: Peak activation buffers on one chip
- `kv_cache_per_chip = 0` for MLP layers

**Activation Memory Calculation (Implementation):**
The implementation counts intermediate buffers conservatively:
- `up_output`: `B × (S/sp) × (I/tp)` elements (output of first projection)
- `down_input`: `B × (S/sp) × (I/tp)` elements (buffer for second projection input)  
- `down_output`: `B × (S/sp) × H` elements (output before all-reduce/reduce-scatter)

This counts `h` twice (up_output + down_input), accounting for peak memory with less aggressive buffer reuse.

**Communication:**
- **Tensor Parallel (TP)**: All-reduce on output `y` of shape `(M × d)` or `(M_local × d)` if SP enabled
- **Sequence Parallel (token-parallel, inference)**: Tokens are partitioned across chips. Each chip independently executes FFN on its local tokens. **No communication is required for FFN layers.**
- `communication_bytes`: Payload bytes (logical tensor size) transferred per chip
  
**Current Implementation Notes:**
1. **Weight sharding**: Weights are NOT currently sharded by TP (full weights on each chip)
2. **Communication**: Communication bytes shown reflect ground-truth payload for TP all-reduce; overlap/timing modeling is future work
3. **Token-parallel SP**: No communication needed for FFN layers (tokens processed independently)
4. These tests document expected behavior and current implementation state


# Test 0 — Vanilla FFN, single chip (inference)

### Test case parameters

* hidden_size (`d`) = **16**
* intermediate_size (`d_ff`) = **64**
* batch_size (`B`) = **4**
* seq_len (`S`) = **8**
* num_chips = **1**
* tensor_parallel = **1**
* pipeline_parallel = **1**
* bytes_per_elem = **2**

Derived:

* tokens `M = B*S = 32`

### Expected calculations (step-by-step)

**1) FLOPs**

* GEMM1: `x[M,d] @ W1[d,d_ff]`
  FLOPs = `2 * M * d * d_ff`
  = `2 * 32 * 16 * 64 = 65,536`
* GEMM2: `h[M,d_ff] @ W2[d_ff,d]`
  FLOPs = `2 * M * d_ff * d`
  = `2 * 32 * 64 * 16 = 65,536`
* Total FLOPs = **131,072**

**2) Weight memory**

* `|W1| = d*d_ff = 16*64 = 1,024 elems`
* `|W2| = d_ff*d = 64*16 = 1,024 elems`
* Total weight elems = 2,048 → bytes = `2,048 * 2 = 4,096`

**3) Activation memory (implementation counts intermediate buffers)**

* `up_output`: `B*S*(I/tp) = 4*8*64 = 2,048 elems` → `4,096 B`
* `down_input`: `B*S*(I/tp) = 4*8*64 = 2,048 elems` → `4,096 B`
* `down_output`: `B*S*H = 4*8*16 = 512 elems` → `1,024 B`
* Total activation bytes = **9,216**

**4) KV cache**

* Not applicable for FFN-only: **0**

**5) Communication**

* Single chip: **0**

### Expected results

**Per-chip metrics**

* flops_per_chip: **131,072**
* weight_memory_per_chip: **4,096**
* activation_memory_per_chip: **9,216**
* kv_cache_per_chip: **0**

**Aggregate metrics**

* flops_total: **131,072**
* weight_memory_total: **4,096**
* activation_memory_total: **9,216**
* kv_cache_total: **0**

**Hardware-dependent (optional)**

* communication_bytes: **0**

---

# Test 1 — Tensor Parallel FFN (TP = 4), shard on intermediate size $d_{ff}$

### Test case parameters

* hidden_size `d` = **1024**
* intermediate_size `d_ff` = **4096**
* batch_size `B` = **2**
* seq_len `S` = **128**
* bytes_per_elem = **2**
* tensor_parallel `tp` = **4**
* sequence_parallel `sp` = **1**
* num_chips = **4**

Derived:

* M = B × S = 256
* d_ff,local = d_ff / tp = 1024

### Expected calculations (step-by-step)

#### 1) FLOPs

Total (same as non-parallel; just split across chips):

* GEMM1 total FLOPs = $2 × M × d × d_{ff}$

  * `= 2 × 256 × 1024 × 4096 = 2,147,483,648`
* GEMM2 total FLOPs = $2 × M × d_{ff} × d$

  * same = 2,147,483,648

**flops_total = 4,294,967,296**

Per chip (even split):

* **flops_per_chip = flops_total / 4 = 1,073,741,824**

#### 2) Weight memory

Total weights:

* `|W1| = d × d_ff = 1024 × 4096 = 4,194,304` elems
* `|W2| = d_ff × d = 4096 × 1024 = 4,194,304` elems
* Total elems = **8,388,608**
* Total bytes = `8,388,608 × 2 = 16,777,216`

**Current implementation:** Weights are NOT sharded (full weights on each chip):

* **weight_memory_per_chip = 16,777,216** (full weights)
* **weight_memory_total = 16,777,216 × 4 = 67,108,864** (replicated across TP)

#### 3) Activation memory

TP shards the intermediate activations $h$ across $d_{ff}$, but $x$ and final $y$ are typically present on each TP rank.

Per chip:

* $x$: `M × d = 256 × 1024 = 262,144` elems → **524,288 B**
* $h_{local}$: `M × d_ff,local = 256 × 1024 = 262,144` elems → **524,288 B**
* $y$: `M × d = 262,144` elems → **524,288 B**

So:

* **activation_memory_per_chip = 1,572,864**
* **activation_memory_total = 4 * 1,572,864 = 6,291,456**

#### 4) Communication

All-reduce on `y` across TP group:

* payload elems = `M × d = 256 × 1024 = 262,144`
* payload bytes = `262,144 × 2 = 524,288`

So:

* **communication_bytes = 524,288**

### Expected Results (Current Implementation)

**Per-chip metrics**

* flops_per_chip: **1,073,741,824**
* weight_memory_per_chip: **16,777,216** (full weights, not sharded)
* activation_memory_per_chip: **1,572,864**
* kv_cache_per_chip: **0**

**Aggregate metrics**

* flops_total: **4,294,967,296**
* weight_memory_total: **67,108,864** (replicated across 4 chips)
* activation_memory_total: **6,291,456**
* kv_cache_total: **0**

**Hardware-dependent**

* communication_bytes: **524,288** (TP all-reduce)

**Note:** Future implementation should have:
- `weight_memory_per_chip = 4,194,304` (sharded)
- `weight_memory_total = 16,777,216` (total unique weights)

---

# Test 2 — Sequence Parallel FFN (SP = 4)

**Token-parallel:** Each chip handles a slice of the tokens. With full SP implementation, requires all-gather on input and reduce-scatter on output.

### Test case parameters

* hidden_size `d` = **1024**
* intermediate_size `d_ff` = **4096**
* batch_size `B` = **2**
* seq_len `S` = **128**
* bytes_per_elem = **2**
* tensor_parallel `tp` = **1**
* sequence_parallel `sp` = **4**
* num_chips = **4**

Derived:

* (M = 256)
* M_local = M / sp = 64

### Expected calculations

#### 1) FLOPs

Total FLOPs unchanged:

* **flops_total = 4,294,967,296**

Per chip (token slice):

* **flops_per_chip = flops_total / 4 = 1,073,741,824**

#### 2) Weight memory

Weights are replicated across SP ranks:

* Total weights bytes = **16,777,216**
* **weight_memory_per_chip = 16,777,216**
* **weight_memory_total = 4 * 16,777,216 = 67,108,864**

#### 3) Activation memory

Per chip uses `M_local = M/sp = 256/4 = 64` tokens, but across `B=2` batches and `S_local=32`:

* `up_output`: `B × S_local × d_ff = 2 × 32 × 4096 = 262,144 elems` → **524,288 B**
* `down_input`: `B × S_local × d_ff = 262,144 elems` → **524,288 B**
* `down_output`: `B × S_local × d = 2 × 32 × 1024 = 65,536 elems` → **131,072 B**

So:

* **activation_memory_per_chip = 1,179,648**
* **activation_memory_total = 4 × 1,179,648 = 4,718,592**


#### 4) Communication

Token-parallel SP: Each chip processes its local tokens independently.

* **communication_bytes = 0** (no communication needed for FFN)

### Expected Results (Current Implementation)

**Per-chip metrics**

* flops_per_chip: **1,073,741,824**
* weight_memory_per_chip: **16,777,216** (replicated)
* activation_memory_per_chip: **1,179,648**
* kv_cache_per_chip: **0**

**Aggregate metrics**

* flops_total: **4,294,967,296**
* weight_memory_total: **67,108,864** (4 full copies)
* activation_memory_total: **4,718,592**
* kv_cache_total: **0**

**Hardware-dependent**

* communication_bytes: **0** (no communication for token-parallel SP)

---

# Test 3 — Hybrid TP×SP (TP = 4, SP = 2), total 8 chips

* TP shards $d_{ff}$ (weights + intermediate activations)
* SP shards tokens ($M$) (activations + compute)
* Communication is TP all-reduce on local (y) within each SP group

### Test case parameters

* hidden_size `d` = **1024**
* intermediate_size `d_ff` = **4096**
* batch_size `B` = **2**
* seq_len `S` = **128**
* bytes_per_elem = **2**
* tensor_parallel `tp` = **4**
* sequence_parallel `sp` = **2**
* num_chips = **8**

Derived:

* (M = 256)
* M_local = M / sp = 128
* d_ff,local = d_ff / tp = 1024

### Expected calculations

#### 1) FLOPs

Total FLOPs unchanged:

* **flops_total = 4,294,967,296**

Per chip split across 8 chips (TP×SP):

* **flops_per_chip = flops_total / 8 = 536,870,912**

#### 2) Weight memory

**Current implementation:** Weights NOT sharded (full weights on each chip):

* Total weight bytes = 16,777,216
* **weight_memory_per_chip = 16,777,216**
* **weight_memory_total = 8 × 16,777,216 = 134,217,728** (replicated across all chips)

**Expected (future):** Sharded by TP, replicated across SP:
- Per chip: 4,194,304 (1/4 of weights)
- Total: 33,554,432 (4 copies of 1/4 weights each)

#### 3) Activation memory

Per chip uses M_local = 128 and d_ff,local = 1024:

* $x_{local}$: (128 × 1024 = 131,072) elems → **262,144 B**
* $h_{local}$: (128 × 1024 = 131,072) elems → **262,144 B**
* $y_{local}$: (128 × 1024 = 131,072) elems → **262,144 B**

So:

* **activation_memory_per_chip = 786,432**
* **activation_memory_total = 8 * 786,432 = 6,291,456**

#### 4) Communication

TP all-reduce on local output `y_local` (shape `M_local × d`) within each SP group:

* payload elems = `M_local × d = 128 × 1024 = 131,072`
* payload bytes = `131,072 × 2 = 262,144`

So:

* **communication_bytes = 262,144**

### Expected Results (Current Implementation)

**Per-chip metrics**

* flops_per_chip: **536,870,912**
* weight_memory_per_chip: **16,777,216** (full weights, not sharded)
* activation_memory_per_chip: **786,432**
* kv_cache_per_chip: **0**

**Aggregate metrics**

* flops_total: **4,294,967,296**
* weight_memory_total: **134,217,728** (8 full copies)
* activation_memory_total: **6,291,456**
* kv_cache_total: **0**

**Hardware-dependent**

* communication_bytes: **262,144** (TP all-reduce on local tokens)

**Note:** Future implementation should have:
- `weight_memory_per_chip = 4,194,304` (sharded by TP)
- `weight_memory_total = 33,554,432` (replicated across SP)

---

## Verification

You can verify these tests against the actual implementation with:

```python
from backend.layers import MLPLayer

# Test 0: Vanilla FFN
layer = MLPLayer(
    name='mlp', layer_idx=0,
    hidden_size=16, intermediate_size=64,
    parallelism={'tensor_parallel': 1}
)
metrics = layer.compute_metrics(batch_size=4, seq_len=8, phase='prefill', dtype='bf16')
print(f"Test 0: FLOPs={metrics.flops_per_chip:,}, Weight={metrics.weight_memory_per_chip:,}, Activation={metrics.activation_memory_per_chip:,}")

# Test 1: TP=4
layer = MLPLayer(
    name='mlp', layer_idx=0,
    hidden_size=1024, intermediate_size=4096,
    parallelism={'tensor_parallel': 4}
)
metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill', dtype='bf16')
print(f"Test 1: FLOPs={metrics.flops_per_chip:,}, Weight={metrics.weight_memory_per_chip:,}, Activation={metrics.activation_memory_per_chip:,}")

# Test 2: SP=4  
layer = MLPLayer(
    name='mlp', layer_idx=0,
    hidden_size=1024, intermediate_size=4096,
    parallelism={'sequence_parallel': 4}
)
metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill', dtype='bf16')
print(f"Test 2: FLOPs={metrics.flops_per_chip:,}, Weight={metrics.weight_memory_per_chip:,}, Activation={metrics.activation_memory_per_chip:,}")

# Test 3: TP=4, SP=2
layer = MLPLayer(
    name='mlp', layer_idx=0,
    hidden_size=1024, intermediate_size=4096,
    parallelism={'tensor_parallel': 4, 'sequence_parallel': 2}
)
metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill', dtype='bf16')
print(f"Test 3: FLOPs={metrics.flops_per_chip:,}, Weight={metrics.weight_memory_per_chip:,}, Activation={metrics.activation_memory_per_chip:,}")
```

## Future Work

1. **Weight sharding for TP**: Implement proper weight sharding so `weight_memory_per_chip` reflects actual per-chip storage
2. **Communication implementation**: Complete `_compute_communication_bytes()` to return actual TP/SP communication costs
3. **Activation memory optimization**: Consider modeling different buffer reuse strategies
4. **Gated MLP tests**: Add tests for 3-projection (SwiGLU-style) MLPs
