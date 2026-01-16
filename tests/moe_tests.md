# MoE (Mixture of Experts) Layer Tests

This document contains test cases for MoE layer implementations with detailed calculations and expected results, focusing especially on parallelism strategies.

## Overview

These tests verify the correctness of:
- **FLOPs computation**: Router gating + expert FFN computation
- **Weight memory**: Router weights + expert weights (per chip and aggregate)
- **Activation memory**: Intermediate tensor buffers including routing tensors
- **Communication**: Expert parallel (EP) all-to-all, TP all-reduce, and hybrid patterns

## MoE Architecture

Standard MoE block with top-k routing:
```
# Router
router_logits = x @ W_router          # [M, E]
weights, indices = top_k(softmax(router_logits), k)  # top-k experts per token

# Expert computation (simplified view)
for each token:
    y = sum over selected experts: weight_i * expert_i(x)
```

## Conventions Used in All Tests

**Notation:**
- `E` = num_experts (total number of experts)
- `k` = top_k (experts selected per token)
- `d` = hidden_size
- `d_ff` = intermediate_size (per expert)
- `B` = batch_size, `S` = seq_len, `M = B*S` = total tokens
- `bytes_per_elem = 2` (BF16/FP16)

**FLOPs Calculation:**
- Router: `2 * M * d * E` (linear projection to expert scores)
- Per expert FFN (2-projection): `2 * tokens * d * d_ff + 2 * tokens * d_ff * d`
- Total expert FLOPs with top-k: `4 * k * M * d * d_ff`
- Combined: `router_flops + expert_flops`

**Weight Memory:**
- Router: `d * E` elements
- Per expert: `d * d_ff + d_ff * d = 2 * d * d_ff` elements
- Total experts: `E * 2 * d * d_ff` elements

**Activation Memory (minimal resident):**
- Input `x`: `M * d`
- Router logits: `M * E`
- Expert intermediate (per expert batch): varies by parallelism
- Expert outputs (accumulated): `M * d`

**Communication:**
- **Expert Parallel (EP)**: All-to-all dispatch (tokens to experts) and combine (outputs back)
- **TP within experts**: All-reduce on expert FFN outputs
- **Hybrid EP x TP**: Both patterns combined

**Expert Parallel Assumptions:**
- With EP=`ep`, each chip holds `E_local = E/ep` experts
- Tokens must be routed to the correct chip via all-to-all
- All-to-all payload per direction: `M * d * bytes_per_elem` (assuming uniform distribution)
- Total EP communication: 2x payload (dispatch + combine)
- Capacity factor `C` = 1.0 (no token dropping, perfect load balance assumed for these tests)

---

## Test Summary

| Test | Config | Experts | Top-k | Hidden | d_ff | Batch | Seq | FLOPs/chip | Weight/chip | Comm |
|------|--------|---------|-------|--------|------|-------|-----|------------|-------------|------|
| MOE-1 | Single chip | 8 | 2 | 1024 | 4096 | 2 | 128 | 8,594,128,896 | 134,234,112 | 0 |
| MOE-2 | EP=4 | 8 | 2 | 1024 | 4096 | 2 | 128 | 2,151,677,952 | 33,570,816 | 1,048,576 |
| MOE-3 | EP=8 | 8 | 2 | 1024 | 4096 | 2 | 128 | 1,077,936,128 | 16,793,600 | 1,048,576 |
| MOE-4 | TP=4 | 8 | 2 | 1024 | 4096 | 2 | 128 | 2,151,677,952 | 33,570,816 | 524,288 |
| MOE-5 | EP=4, TP=2 | 8 | 2 | 1024 | 4096 | 2 | 128 | 1,077,936,128 | 16,793,600 | 1,310,720 |
| MOE-6 | EP=8, TP=4, shared=2 | 8+2 | 2 | 1024 | 4096 | 2 | 128 | 541,065,216 | 12,599,296 | 1,245,184 |
| MOE-7 | EP=4, CP=2 | 8 | 2 | 1024 | 4096 | 2 | 128 | 1,075,838,976 | 33,570,816 | 524,288 |
| MOE-8 | EP=8, TP=4, CP=2 | 8 | 2 | 1024 | 4096 | 2 | 128 | 136,314,880 | 4,210,688 | 589,824 |

---

# Test MOE-1 — MoE Single Chip (baseline)

### Test case parameters

* hidden_size `d` = **1024**
* intermediate_size `d_ff` = **4096**
* num_experts `E` = **8**
* top_k `k` = **2**
* batch_size `B` = **2**
* seq_len `S` = **128**
* bytes_per_elem = **2**
* num_chips = **1**
* expert_parallel `ep` = **1**
* tensor_parallel `tp` = **1**

Derived:

* `M = B*S = 256` tokens
* Total expert invocations = `k * M = 512`

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Router**
Linear projection `x[M,d] @ W_router[d,E]`:

* FLOPs = `2 * M * d * E`
* = `2 * 256 * 1024 * 8`
* = `4,194,304`

**(b) Expert FFN computation**
Each token is processed by `k=2` experts. Using the formula `4 * k * M * d * d_ff`:

* Expert FLOPs = `4 * k * M * d * d_ff`
* = `4 * 2 * 256 * 1024 * 4096`
* Step-by-step: `4 * 2 = 8`, `8 * 256 = 2,048`, `2,048 * 1024 = 2,097,152`, `2,097,152 * 4096 = 8,589,934,592`
* = `8,589,934,592`

**(c) Total FLOPs**

* Router + Experts = `4,194,304 + 8,589,934,592`
* **flops_total = 8,594,128,896**
* **flops_per_chip = 8,594,128,896**

#### 2) Weight memory

**(a) Router weights**

* Elements = `d * E = 1024 * 8 = 8,192`
* Bytes = `8,192 * 2 = 16,384`

**(b) Expert weights**
Each expert has W1[d, d_ff] and W2[d_ff, d]:

* Elements per expert = `2 * d * d_ff = 2 * 1024 * 4096 = 8,388,608`
* Total for E=8 experts = `8 * 8,388,608 = 67,108,864`
* Bytes = `67,108,864 * 2 = 134,217,728`

**(c) Total weights**

* Total bytes = `16,384 + 134,217,728 = 134,234,112`

So:

* **weight_memory_per_chip = 134,234,112**
* **weight_memory_total = 134,234,112**

#### 3) Activation memory (resident, minimal)

For inference, minimal resident buffers:

* Input `x`: `M * d = 256 * 1024 = 262,144` elems -> `524,288` bytes
* Router logits: `M * E = 256 * 8 = 2,048` elems -> `4,096` bytes
* Expert intermediate (peak, one expert batch at a time): `M * d_ff = 256 * 4096 = 1,048,576` elems -> `2,097,152` bytes
* Output `y`: `M * d = 262,144` elems -> `524,288` bytes

Total:

* **activation_memory_per_chip = 524,288 + 4,096 + 2,097,152 + 524,288 = 3,149,824**
* **activation_memory_total = 3,149,824**

#### 4) KV cache

* Not applicable for MoE FFN layers: **0**

#### 5) Communication

* Single chip, no parallelism: **0**

### Expected results

**Per-chip metrics**

* flops_per_chip: **8594128896**
* weight_memory_per_chip: **134234112**
* activation_memory_per_chip: **3149824**
* kv_cache_per_chip: **0**

**Aggregate metrics**

* flops_total: **8594128896**
* weight_memory_total: **134234112**
* activation_memory_total: **3149824**
* kv_cache_total: **0**

**Hardware-dependent (optional)**

* communication_bytes: **0**

---

# Test MOE-2 — Expert Parallel (EP=4)

### Assumptions (explicit)

* **Expert Parallel (EP)** shards experts across chips
* Each chip holds `E_local = E/ep = 8/4 = 2` experts
* Router is replicated on all chips (small overhead)
* All-to-all communication dispatches tokens to correct expert chips
* All-to-all combines outputs back to original token positions
* Assuming uniform token distribution: each chip processes `M * k / ep` token-expert pairs

### Test case parameters

Same as MOE-1, plus:

* expert_parallel `ep` = **4**
* num_chips = **4**

Derived:

* `E_local = 8/4 = 2` experts per chip
* Token-expert pairs per chip = `k * M / ep = 2 * 256 / 4 = 128`

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Router (replicated)**
Each chip computes full router:

* Router FLOPs per chip = `2 * M * d * E = 4,194,304`

**(b) Expert computation per chip**
Each chip processes its share of token-expert pairs:

* Token-expert pairs per chip = `k * M / ep = 512 / 4 = 128`
* Expert FLOPs per chip = `128 * 4 * d * d_ff`
* = `128 * 4 * 1024 * 4096`
* = `2,147,483,648`

**(c) Total**

Per chip computes router (replicated) + local experts (unique):

* **flops_per_chip = 4,194,304 + 2,147,483,648 = 2,151,677,952**
* **flops_total = 4 * 2,151,677,952 = 8,606,711,808** (includes replicated router)

#### 2) Weight memory

**(a) Router (replicated)**

* Bytes per chip = `16,384`

**(b) Expert weights (sharded)**

* Total expert weight bytes = `134,217,728`
* Per chip with EP=4 = `134,217,728 / 4 = 33,554,432`

**(c) Total**

* **weight_memory_per_chip = 16,384 + 33,554,432 = 33,570,816**
* **weight_memory_total = 4 * 33,570,816 = 134,283,264** (router replicated, experts sharded)

#### 3) Activation memory

Per chip processes `M` tokens through router, but only local expert activations:

* Input x: `M * d` = 524,288 bytes
* Router logits: `M * E` = 4,096 bytes
* Expert intermediate (local token-expert pairs): `(k*M/ep) * d_ff = 128 * 4096 = 524,288` elems -> 1,048,576 bytes
* Output y: `M * d` = 524,288 bytes

Total:

* **activation_memory_per_chip = 524,288 + 4,096 + 1,048,576 + 524,288 = 2,101,248**
* **activation_memory_total = 4 * 2,101,248 = 8,404,992**

#### 4) KV cache

* **kv_cache_per_chip = 0**
* **kv_cache_total = 0**

#### 5) Communication

EP requires all-to-all communication for:
1. **Dispatch**: Send tokens to expert chips. Payload = `M * d * bytes = 256 * 1024 * 2 = 524,288` bytes
2. **Combine**: Return expert outputs. Payload = `M * d * bytes = 524,288` bytes

Total all-to-all payload:

* **communication_bytes = 2 * 524,288 = 1,048,576**

### Expected results

**Per-chip metrics**

* flops_per_chip: **2151677952**
* weight_memory_per_chip: **33570816**
* activation_memory_per_chip: **2101248**
* kv_cache_per_chip: **0**

**Aggregate metrics**

* flops_total: **8606711808**
* weight_memory_total: **134283264**
* activation_memory_total: **8404992**
* kv_cache_total: **0**

**Hardware-dependent (optional)**

* communication_bytes: **1048576**

---

# Test MOE-3 — Expert Parallel (EP=8, one expert per chip)

### Assumptions (explicit)

* Maximum expert parallelism: each chip holds exactly 1 expert
* `E_local = E/ep = 8/8 = 1`
* Router replicated on all chips
* This maximizes parallelism but also maximizes all-to-all communication

### Test case parameters

Same as MOE-1, plus:

* expert_parallel `ep` = **8**
* num_chips = **8**

Derived:

* `E_local = 1` expert per chip
* Token-expert pairs per chip = `k * M / ep = 512 / 8 = 64`

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Router (replicated)**

* Router FLOPs per chip = `4,194,304`

**(b) Expert computation per chip**

* Token-expert pairs per chip = `64`
* Expert FLOPs per chip = `64 * 4 * d * d_ff`
* = `64 * 4 * 1024 * 4096`
* = `1,073,741,824`

**(c) Total**

* **flops_per_chip = 4,194,304 + 1,073,741,824 = 1,077,936,128**
* **flops_total = 8 * 1,077,936,128 = 8,623,489,024** (includes replicated router 8x)

#### 2) Weight memory

**(a) Router (replicated)**

* Bytes per chip = `16,384`

**(b) Expert weights (sharded)**

* Per chip = `134,217,728 / 8 = 16,777,216`

**(c) Total**

* **weight_memory_per_chip = 16,384 + 16,777,216 = 16,793,600**
* **weight_memory_total = 8 * 16,793,600 = 134,348,800**

#### 3) Activation memory

* Input x: 524,288 bytes
* Router logits: 4,096 bytes
* Expert intermediate: `64 * 4096 = 262,144` elems -> 524,288 bytes
* Output y: 524,288 bytes

Total:

* **activation_memory_per_chip = 524,288 + 4,096 + 524,288 + 524,288 = 1,576,960**
* **activation_memory_total = 8 * 1,576,960 = 12,615,680**

#### 4) KV cache

* **kv_cache_per_chip = 0**
* **kv_cache_total = 0**

#### 5) Communication

All-to-all dispatch + combine:

* **communication_bytes = 2 * M * d * bytes = 2 * 256 * 1024 * 2 = 1,048,576**

(Same as EP=4; all-to-all payload scales with total tokens, not EP degree)

### Expected results

**Per-chip metrics**

* flops_per_chip: **1077936128**
* weight_memory_per_chip: **16793600**
* activation_memory_per_chip: **1576960**
* kv_cache_per_chip: **0**

**Aggregate metrics**

* flops_total: **8623489024**
* weight_memory_total: **134348800**
* activation_memory_total: **12615680**
* kv_cache_total: **0**

**Hardware-dependent (optional)**

* communication_bytes: **1048576**

---

# Test MOE-4 — Tensor Parallel within Experts (TP=4)

### Assumptions (explicit)

* **TP shards each expert's FFN** across the intermediate dimension
* Each chip holds all E=8 experts, but each expert is TP-sharded
* `d_ff_local = d_ff / tp = 4096 / 4 = 1024`
* Router is replicated (not sharded by TP)
* Requires all-reduce after each expert's down projection

### Test case parameters

Same as MOE-1, plus:

* tensor_parallel `tp` = **4**
* expert_parallel `ep` = **1**
* num_chips = **4**

Derived:

* `d_ff_local = d_ff / tp = 1024`
* All experts on all chips (but sharded)

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Router (replicated)**

* Router FLOPs per chip = `4,194,304`

**(b) Expert computation**
Total expert FLOPs are split across TP ranks:

* Total expert FLOPs = `8,589,934,592`
* Per chip with TP=4 = `8,589,934,592 / 4 = 2,147,483,648`

**(c) Total**

* **flops_per_chip = 4,194,304 + 2,147,483,648 = 2,151,677,952**
* **flops_total = 4 * 2,151,677,952 = 8,606,711,808**

#### 2) Weight memory

**(a) Router (replicated)**

* Bytes per chip = `16,384`

**(b) Expert weights (TP-sharded)**
Each expert's W1 and W2 are sharded:
* W1[d, d_ff] -> W1_shard[d, d_ff/tp] = [1024, 1024]
* W2[d_ff, d] -> W2_shard[d_ff/tp, d] = [1024, 1024]
* Elements per expert shard = `d * d_ff/tp + d_ff/tp * d = 2 * d * d_ff/tp`
* = `2 * 1024 * 1024 = 2,097,152`
* For all E=8 experts = `8 * 2,097,152 = 16,777,216` elements
* Bytes = `16,777,216 * 2 = 33,554,432`

**(c) Total**

Router is replicated, expert weights are TP-sharded:

* **weight_memory_per_chip = 16,384 + 33,554,432 = 33,570,816**
* **weight_memory_total = 4 * 16,384 + 134,217,728 = 134,283,264**

#### 3) Activation memory

Per chip processes all M tokens through router and TP-sharded experts:

* Input x (replicated): `M * d` = 524,288 bytes
* Router logits: `M * E` = 4,096 bytes
* Expert intermediate (TP-local): `M * d_ff_local = 256 * 1024 = 262,144` elems -> 524,288 bytes
* Output y (after all-reduce): `M * d` = 524,288 bytes

Total:

* **activation_memory_per_chip = 524,288 + 4,096 + 524,288 + 524,288 = 1,576,960**
* **activation_memory_total = 4 * 1,576,960 = 6,307,840**

#### 4) KV cache

* **kv_cache_per_chip = 0**
* **kv_cache_total = 0**

#### 5) Communication

TP requires all-reduce on expert outputs. For MoE, need to all-reduce after processing each token's selected experts:

* All-reduce payload = `M * d * bytes_per_elem = 256 * 1024 * 2 = 524,288`

But with k=2 experts per token and sparse routing, communication may be more complex. Simplified assumption: one all-reduce on final accumulated output.

* **communication_bytes = 524,288**

### Expected results

**Per-chip metrics**

* flops_per_chip: **2151677952**
* weight_memory_per_chip: **33570816**
* activation_memory_per_chip: **1576960**
* kv_cache_per_chip: **0**

**Aggregate metrics**

* flops_total: **8606711808**
* weight_memory_total: **134283264**
* activation_memory_total: **6307840**
* kv_cache_total: **0**

**Hardware-dependent (optional)**

* communication_bytes: **524288**

---

# Test MOE-5 — Hybrid Expert Parallel + Tensor Parallel (EP=4, TP=2)

### Assumptions (explicit)

* **EP shards experts**: each EP group holds `E/ep = 8/4 = 2` experts
* **TP shards each expert**: within each EP group, experts are TP-sharded
* Total chips = `ep * tp = 4 * 2 = 8`
* Topology: 4 EP groups, each with 2 TP ranks
* Communication: EP all-to-all + TP all-reduce within each EP group

### Test case parameters

Same as MOE-1, plus:

* expert_parallel `ep` = **4**
* tensor_parallel `tp` = **2**
* num_chips = **8**

Derived:

* `E_local = E / ep = 2` experts per EP group
* `d_ff_local = d_ff / tp = 2048` per TP rank
* Token-expert pairs per EP group = `k * M / ep = 128`
* Token-expert pairs per chip = `128` (each TP rank processes all, but sharded)

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Router (replicated)**

* Router FLOPs per chip = `4,194,304`

**(b) Expert computation**
Total expert FLOPs split across EP*TP:

* Total expert FLOPs = `8,589,934,592`
* Per chip = `8,589,934,592 / 8 = 1,073,741,824`

**(c) Total**

* **flops_per_chip = 4,194,304 + 1,073,741,824 = 1,077,936,128**
* **flops_total = 8 * 1,077,936,128 = 8,623,489,024**

#### 2) Weight memory

**(a) Router (replicated)**

* Bytes per chip = `16,384`

**(b) Expert weights (EP + TP sharded)**
Each chip holds `E_local=2` experts, each TP-sharded:
* Elements per expert per TP rank = `2 * d * d_ff / tp = 2 * 1024 * 4096 / 2 = 4,194,304`
* Elements for 2 experts = `2 * 4,194,304 = 8,388,608`
* Bytes = `8,388,608 * 2 = 16,777,216`

**(c) Total**

* **weight_memory_per_chip = 16,384 + 16,777,216 = 16,793,600**
* **weight_memory_total = 8 * 16,793,600 = 134,348,800**

#### 3) Activation memory

* Input x (replicated across EP): 524,288 bytes
* Router logits: 4,096 bytes
* Expert intermediate (EP + TP local): `(k*M/ep) * d_ff/tp = 128 * 2048 = 262,144` elems -> 524,288 bytes
* Output y: 524,288 bytes

Total:

* **activation_memory_per_chip = 524,288 + 4,096 + 524,288 + 524,288 = 1,576,960**
* **activation_memory_total = 8 * 1,576,960 = 12,615,680**

#### 4) KV cache

* **kv_cache_per_chip = 0**
* **kv_cache_total = 0**

#### 5) Communication

Two communication patterns:

**(a) EP all-to-all**

* Dispatch + combine payload = `2 * M * d * bytes = 1,048,576`

**(b) TP all-reduce (within each EP group)**

* All-reduce payload per EP group = `(k*M/ep) * d * bytes`
* = `128 * 1024 * 2 = 262,144`

Total communication per chip:

* **communication_bytes = 1,048,576 + 262,144 = 1,310,720**

### Expected results

**Per-chip metrics**

* flops_per_chip: **1077936128**
* weight_memory_per_chip: **16793600**
* activation_memory_per_chip: **1576960**
* kv_cache_per_chip: **0**

**Aggregate metrics**

* flops_total: **8623489024**
* weight_memory_total: **134348800**
* activation_memory_total: **12615680**
* kv_cache_total: **0**

**Hardware-dependent (optional)**

* communication_bytes: **1310720**

---

# Test MOE-6 — Large-Scale MoE with Shared Experts (EP=8, TP=4, shared_experts=2)

### Assumptions (explicit)

This test covers **shared expert** architectures (e.g., DeepSeek-style) where some experts are always activated for all tokens in addition to top-k routed experts.

* **Routed experts**: `E_routed = 8`, top_k = 2
* **Shared experts**: `E_shared = 2` (always activated for all tokens)
* Total experts: `E_total = E_routed + E_shared = 10`
* EP shards routed experts only (shared experts replicated or separately sharded)
* For this test: shared experts are replicated across all chips

### Test case parameters

* hidden_size `d` = **1024**
* intermediate_size `d_ff` = **4096**
* num_routed_experts `E_routed` = **8**
* num_shared_experts `E_shared` = **2**
* top_k `k` = **2**
* batch_size `B` = **2**
* seq_len `S` = **128**
* bytes_per_elem = **2**
* expert_parallel `ep` = **8**
* tensor_parallel `tp` = **4**
* num_chips = **32** (EP * TP)

Derived:

* `M = 256` tokens
* `E_routed_local = 8 / 8 = 1` routed expert per EP group
* `d_ff_local = 4096 / 4 = 1024` per TP rank
* Routed token-expert pairs per chip = `(k * M) / (ep * tp) = 512 / 32 = 16`
* Shared expert tokens per chip = `M / tp = 64` (shared across EP, TP-sharded)

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Router**

* Router FLOPs per chip = `2 * M * d * E_routed = 2 * 256 * 1024 * 8 = 4,194,304`

**(b) Routed expert computation**
Total routed expert FLOPs split across EP*TP:

* Routed FLOPs total = `4 * k * M * d * d_ff = 8,589,934,592`
* Per chip = `8,589,934,592 / 32 = 268,435,456`

**(c) Shared expert computation**
Each token goes through both shared experts. Shared experts are computed once and sharded across all chips:

* Shared expert FLOPs total = `4 * E_shared * M * d * d_ff`
* = `4 * 2 * 256 * 1024 * 4096 = 8,589,934,592`
* Per chip = `8,589,934,592 / 32 = 268,435,456`

**(d) Total**

* Router: 4,194,304 (replicated)
* Routed experts: 268,435,456 (1/32 of routed)
* Shared experts: 268,435,456 (1/32 of shared)
* **flops_per_chip = 4,194,304 + 268,435,456 + 268,435,456 = 541,065,216**
* **flops_total = 32 * 541,065,216 = 17,314,086,912**

#### 2) Weight memory

**(a) Router (replicated)**

* Bytes per chip = `16,384`

**(b) Routed expert weights (EP + TP sharded)**

* Per chip = `(E_routed / ep) * 2 * d * (d_ff / tp) * bytes`
* = `1 * 2 * 1024 * 1024 * 2 = 4,194,304`

**(c) Shared expert weights (replicated across EP, TP-sharded)**

* Per chip (within TP group) = `E_shared * 2 * d * (d_ff / tp) * bytes`
* = `2 * 2 * 1024 * 1024 * 2 = 8,388,608`

Shared experts are replicated across EP groups:
* Total shared weight bytes = `E_shared * 2 * d * d_ff * bytes = 33,554,432`
* Per TP rank within any EP group = `33,554,432 / tp = 8,388,608`

**(d) Total**

* **weight_memory_per_chip = 16,384 + 4,194,304 + 8,388,608 = 12,599,296**
* **weight_memory_total = 32 * 12,599,296 = 403,177,472**

#### 3) Activation memory

* Input x: 524,288 bytes
* Router logits: 4,096 bytes
* Routed expert intermediate: `16 * d_ff_local = 16 * 1024 = 16,384` elems -> 32,768 bytes
* Shared expert intermediate: `64 * d_ff_local = 65,536` elems -> 131,072 bytes
* Output y: 524,288 bytes

Total:

* **activation_memory_per_chip = 524,288 + 4,096 + 32,768 + 131,072 + 524,288 = 1,216,512**
* **activation_memory_total = 32 * 1,216,512 = 38,928,384**

#### 4) KV cache

* **kv_cache_per_chip = 0**
* **kv_cache_total = 0**

#### 5) Communication

**(a) EP all-to-all (for routed experts)**

* Dispatch + combine = `2 * M * d * bytes = 1,048,576`

**(b) TP all-reduce (for both routed and shared within each EP group)**

* Routed output all-reduce: `(k*M/ep) * d * bytes = 64 * 1024 * 2 = 131,072`
* Shared output all-reduce: `(M/ep) * d * bytes = 32 * 1024 * 2 = 65,536`
* Total TP all-reduce = `131,072 + 65,536 = 196,608`

**(c) Shared expert broadcast across EP (if computed centrally)**
If shared experts are computed in one place and broadcast:
* Broadcast payload = `M * d * bytes = 524,288`

Assuming shared computed in parallel (TP-sharded) and no EP broadcast needed:

* **communication_bytes = 1,048,576 + 196,608 = 1,245,184**

### Expected results

**Per-chip metrics**

* flops_per_chip: **541065216**
* weight_memory_per_chip: **12599296**
* activation_memory_per_chip: **1216512**
* kv_cache_per_chip: **0**

**Aggregate metrics**

* flops_total: **17314086912**
* weight_memory_total: **403177472**
* activation_memory_total: **38928384**
* kv_cache_total: **0**

**Hardware-dependent (optional)**

* communication_bytes: **1245184**

---

# Test MOE-7 — MoE with Context Parallel (EP=4, CP=2)

### Assumptions (explicit)

* **Context Parallel (CP)** shards the sequence across chips
* Combined with EP, tokens are first partitioned by CP, then routed to experts by EP
* Each chip processes `S_local = S/cp` tokens
* All-to-all happens within each CP group (tokens stay in their CP partition)

### Test case parameters

* hidden_size `d` = **1024**
* intermediate_size `d_ff` = **4096**
* num_experts `E` = **8**
* top_k `k` = **2**
* batch_size `B` = **2**
* seq_len `S` = **128**
* bytes_per_elem = **2**
* expert_parallel `ep` = **4**
* context_parallel `cp` = **2**
* tensor_parallel `tp` = **1**
* num_chips = **8** (EP * CP)

Derived:

* `M = 256` tokens
* `M_local = M / cp = 128` tokens per CP group
* `E_local = E / ep = 2` experts per EP rank
* Token-expert pairs per chip = `k * M_local / ep = 2 * 128 / 4 = 64`

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Router (per CP group)**
Each CP group processes M_local tokens:

* Router FLOPs per chip = `2 * M_local * d * E`
* = `2 * 128 * 1024 * 8 = 2,097,152`

**(b) Expert computation per chip**

* Total expert FLOPs = `4 * k * M * d * d_ff = 8,589,934,592`
* Per chip (split across EP * CP) = `8,589,934,592 / 8 = 1,073,741,824`

**(c) Total**

* **flops_per_chip = 2,097,152 + 1,073,741,824 = 1,075,838,976**
* **flops_total = 8 * 1,075,838,976 = 8,606,711,808**

#### 2) Weight memory

**(a) Router (replicated across all chips)**

* Bytes per chip = `16,384`

**(b) Expert weights (EP-sharded, replicated across CP)**

* Per EP rank (2 experts) = `2 * 2 * d * d_ff * bytes = 2 * 2 * 1024 * 4096 * 2 = 33,554,432`
* Each expert shard replicated across cp=2 CP ranks
* Per chip = `33,554,432`

**(c) Total**

Expert weights are unique across EP but replicated across CP:

* **weight_memory_per_chip = 16,384 + 33,554,432 = 33,570,816**
* **weight_memory_total = 8 * 33,570,816 = 268,566,528**

#### 3) Activation memory

Each chip processes M_local tokens:

* Input x_local: `M_local * d = 128 * 1024 = 131,072` elems -> 262,144 bytes
* Router logits: `M_local * E = 128 * 8 = 1,024` elems -> 2,048 bytes
* Expert intermediate: `64 * d_ff = 262,144` elems -> 524,288 bytes
* Output y_local: `M_local * d` = 262,144 bytes

Total:

* **activation_memory_per_chip = 262,144 + 2,048 + 524,288 + 262,144 = 1,050,624**
* **activation_memory_total = 8 * 1,050,624 = 8,404,992**

#### 4) KV cache

* **kv_cache_per_chip = 0**
* **kv_cache_total = 0**

#### 5) Communication

**(a) EP all-to-all (within each CP group)**
Each CP group has ep=4 chips. All-to-all on M_local tokens:

* Dispatch + combine = `2 * M_local * d * bytes = 2 * 128 * 1024 * 2 = 524,288`

**(b) No CP communication for MoE layer**
CP partitions are independent for feed-forward layers.

Total:

* **communication_bytes = 524,288**

### Expected results

**Per-chip metrics**

* flops_per_chip: **1075838976**
* weight_memory_per_chip: **33570816**
* activation_memory_per_chip: **1050624**
* kv_cache_per_chip: **0**

**Aggregate metrics**

* flops_total: **8606711808**
* weight_memory_total: **268566528**
* activation_memory_total: **8404992**
* kv_cache_total: **0**

**Hardware-dependent (optional)**

* communication_bytes: **524288**

---

# Test MOE-8 — Maximum Parallelism (EP=8, TP=4, CP=2)

### Assumptions (explicit)

* All three parallelism dimensions active
* Total chips = EP * TP * CP = 8 * 4 * 2 = 64
* Each chip holds: 1 expert (EP=8), TP-sharded (TP=4), processing local sequence (CP=2)

### Test case parameters

* hidden_size `d` = **1024**
* intermediate_size `d_ff` = **4096**
* num_experts `E` = **8**
* top_k `k` = **2**
* batch_size `B` = **2**
* seq_len `S` = **128**
* bytes_per_elem = **2**
* expert_parallel `ep` = **8**
* tensor_parallel `tp` = **4**
* context_parallel `cp` = **2**
* num_chips = **64**

Derived:

* `M = 256`, `M_local = M / cp = 128`
* `E_local = E / ep = 1`
* `d_ff_local = d_ff / tp = 1024`
* Token-expert pairs per chip = `k * M_local / ep = 2 * 128 / 8 = 32`

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Router**

* Router FLOPs per chip = `2 * M_local * d * E = 2 * 128 * 1024 * 8 = 2,097,152`

**(b) Expert computation**

* Total expert FLOPs = `8,589,934,592`
* Per chip = `8,589,934,592 / 64 = 134,217,728`

**(c) Total**

* **flops_per_chip = 2,097,152 + 134,217,728 = 136,314,880**
* **flops_total = 64 * 136,314,880 = 8,724,152,320**

#### 2) Weight memory

**(a) Router (replicated)**

* Per chip = `16,384`

**(b) Expert weights (EP + TP sharded, replicated across CP)**

* Per chip = `E_local * 2 * d * d_ff_local * bytes`
* = `1 * 2 * 1024 * 1024 * 2 = 4,194,304`

**(c) Total**

* **weight_memory_per_chip = 16,384 + 4,194,304 = 4,210,688**
* **weight_memory_total = 64 * 4,210,688 = 269,484,032**

#### 3) Activation memory

* Input x_local: 262,144 bytes
* Router logits: 2,048 bytes
* Expert intermediate: `32 * d_ff_local = 32,768` elems -> 65,536 bytes
* Output y_local: 262,144 bytes

Total:

* **activation_memory_per_chip = 262,144 + 2,048 + 65,536 + 262,144 = 591,872**
* **activation_memory_total = 64 * 591,872 = 37,879,808**

#### 4) KV cache

* **kv_cache_per_chip = 0**
* **kv_cache_total = 0**

#### 5) Communication

**(a) EP all-to-all (within each CP*TP group)**

* Per CP group, EP=8 chips do all-to-all
* Payload = `2 * M_local * d * bytes = 524,288`

**(b) TP all-reduce (within each EP*CP group)**

* Local output size = `32 * d * bytes = 65,536`
* All-reduce payload = `65,536`

Total:

* **communication_bytes = 524,288 + 65,536 = 589,824**

### Expected results

**Per-chip metrics**

* flops_per_chip: **136314880**
* weight_memory_per_chip: **4210688**
* activation_memory_per_chip: **591872**
* kv_cache_per_chip: **0**

**Aggregate metrics**

* flops_total: **8724152320**
* weight_memory_total: **269484032**
* activation_memory_total: **37879808**
* kv_cache_total: **0**

**Hardware-dependent (optional)**

* communication_bytes: **589824**

---

## Notes / Edge Cases

### Token Dropping and Capacity Factor

These tests assume perfect load balancing (capacity factor C=1.0). In practice:
- Tokens may be dropped if expert capacity is exceeded
- Capacity factor C > 1.0 provides buffer for imbalanced routing
- FLOPs and memory scale with actual tokens processed, not theoretical maximum

### Auxiliary Loss Computation

Router typically includes auxiliary losses (load balancing, entropy). These add small FLOPs:
- Load balancing loss: O(E) per token
- Not included in these tests as they're negligible compared to expert compute

### Gated Experts (SwiGLU-style)

Modern MoE often uses gated FFN (3 projections instead of 2):
- Up projection + gate projection + down projection
- FLOPs = `6 * tokens * d * d_ff` instead of `4 * tokens * d * d_ff`
- Weight memory increases by 50%

Future test: MOE with gated experts could be added.

### Communication Overlap

All-to-all communication can often be overlapped with computation:
- Dispatch overlapped with router computation
- Expert compute overlapped with other experts' dispatch
- These optimizations don't change total bytes, but affect latency

---

## Verification

Tests can be verified against implementation with:

```python
from backend.layers import MoELayer

# Test MOE-1: Single chip
layer = MoELayer(
    name='moe', layer_idx=0,
    hidden_size=1024, intermediate_size=4096,
    num_experts=8, top_k=2,
    parallelism={'expert_parallel': 1, 'tensor_parallel': 1}
)
metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill', dtype='bf16')
print(f"MOE-1: FLOPs={metrics.flops_per_chip:,}, Weight={metrics.weight_memory_per_chip:,}")

# Test MOE-2: EP=4
layer = MoELayer(
    name='moe', layer_idx=0,
    hidden_size=1024, intermediate_size=4096,
    num_experts=8, top_k=2,
    parallelism={'expert_parallel': 4, 'tensor_parallel': 1}
)
metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill', dtype='bf16')
print(f"MOE-2: FLOPs={metrics.flops_per_chip:,}, Weight={metrics.weight_memory_per_chip:,}")

# Additional tests follow same pattern...
```

## Future Work

1. **Gated MoE tests**: Add tests for 3-projection (SwiGLU-style) experts
2. **Capacity factor tests**: Tests with C > 1.0 and token dropping
3. **Decode phase tests**: MoE behavior during autoregressive decode
4. **Auxiliary loss FLOPs**: Include router loss computation
5. **Pipeline parallel MoE**: PP combined with EP/TP
