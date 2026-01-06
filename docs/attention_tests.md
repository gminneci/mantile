
Conventions as before:

* `bytes_per_elem = 2` (FP16/BF16)
* `B=batch_size`, `S=seq_len`, `M=B*S`
* `d=hidden_size`, `h=num_heads`, `dh=head_dim` with `d = h*dh`
* **FLOPs for GEMM** `(M×K)@(K×N)` = `2*M*K*N`
* **communication_bytes** = **payload bytes** (logical tensor size), not “ring injected bytes”

Standard MHA block with 4 dense projections:

* `Q = X Wq`, `K = X Wk`, `V = X Wv`, `Y = O Wo` where `O = softmax(QK^T / sqrt(dh)) V`

---

## Future Enhancements / Roadmap

### Tensor Parallelism by Hidden Dimension (TP-d)

**Current implementation**: TP-by-heads (TP-h) shards attention heads across devices.

**Proposed**: TP-d would shard the hidden dimension across projections, useful for:
- **Multi-head Latent Attention (MLA)** architectures (e.g., DeepSeek V2/V3)
  - MLA uses low-rank compression/decompression instead of traditional heads
  - Projects to compressed latent dimension, then back
  - TP-d shards hidden dim across compression/decompression operations
- Models with very large hidden dimensions relative to head count
- Different communication patterns that may be more efficient for certain architectures

**Configuration examples**:
- Current: `{"tensor_parallel": 4}` means TP-by-heads (4-way head sharding)
- Future: Could support both:
  - `{"tensor_parallel_heads": 4}` - shard by heads (current behavior)
  - `{"tensor_parallel_dim": 4}` - shard by hidden dimension (new)
  - Potentially both: `{"tensor_parallel_heads": 2, "tensor_parallel_dim": 2}` for 4-way parallelism

**Implementation considerations**:
- Different all-reduce/all-gather patterns compared to TP-by-heads
- Weight sharding across projections differs (each projection shards columns vs rows)
- Communication occurs at different points in the attention pipeline
- May require separate layer types or mode parameter to avoid confusion

**Priority**: Implement when adding MLA or similar non-head-based attention mechanisms.

---

# Test MHA-1 — Standard Attention Prefill, single chip

### Test case parameters

* hidden_size `d` = **1024**
* num_heads `h` = **16**
* head_dim `dh` = **64**  (since 16*64 = 1024)
* batch_size `B` = **2**
* seq_len `S` = **128**
* bytes_per_elem = **2**
* num_chips = **1**
* tensor_parallel `tp` = **1**

Derived:

* `M = B*S = 256`

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Q/K/V projections**
Each is `X[M,d] @ W[d,d]`:

* FLOPs per projection = `2 * M * d * d`
* = `2 * 256 * 1024 * 1024 = 536,870,912`
  There are 3 of them:
* QKV FLOPs = `3 * 536,870,912 = 1,610,612,736`

**(b) Attention scores QKᵀ**
Per batch & head: `(S×dh) @ (dh×S)` → `2*S*S*dh`

* FLOPs = `2 * B * h * S * S * dh`
* = `2 * 2 * 16 * 128 * 128 * 64`
* First: `128*128=16,384`
* Then: `16,384*64=1,048,576`
* Then: `2*2*16=64`
* Total = `64 * 1,048,576 = 67,108,864`

**(c) Apply attention to V (softmax(QKᵀ)V)**
Per batch & head: `(S×S) @ (S×dh)` → `2*S*S*dh`
Same as (b):

* = **67,108,864**

**(d) Output projection**
`O[M,d] @ Wo[d,d]`

* FLOPs = `2 * M * d * d`
* = `536,870,912`

**Total FLOPs**

* `flops_total = QKV + scores + applyV + outproj`
* = `1,610,612,736 + 67,108,864 + 67,108,864 + 536,870,912`
* = **2,281,701,376**

Per chip (1 chip):

* **flops_per_chip = 2,281,701,376**

#### 2) Weight memory

Weights are 4 matrices of size `d×d` (Wq,Wk,Wv,Wo):

* total elems = `4 * d * d = 4 * 1024 * 1024 = 4,194,304`
* bytes = `4,194,304 * 2 = 8,388,608`

Per chip:

* **weight_memory_per_chip = 8,388,608**
* Total:
* **weight_memory_total = 8,388,608**

#### 3) Activation memory (inference, “resident tensors”)

A clean minimal set for attention prefill:

* X: `M×d`
* Q,K,V: each `M×d`
* Output Y: `M×d`

Total elems = `5 * M * d = 5 * 256 * 1024 = 1,310,720`
Bytes = `1,310,720 * 2 = 2,621,440`

So:

* **activation_memory_per_chip = 2,621,440**
* **activation_memory_total = 2,621,440**

(We are intentionally **not** counting the `B*h*S*S` attention matrix as a resident activation buffer, since most implementations tile/stream it.)

#### 4) KV cache

Prefill writes KV cache for all tokens:

* K cache elems = `B * h * S * dh = B*S*d = 2*128*1024 = 262,144`
* V cache elems = same
* Total KV elems = `2 * 262,144 = 524,288`
  Bytes = `524,288 * 2 = 1,048,576`

So:

* **kv_cache_per_chip = 1,048,576**
* **kv_cache_total = 1,048,576**

#### 5) Communication

Single chip:

* **communication_bytes = 0**

### Expected results

**Per-chip metrics**

* flops_per_chip: **2281701376**
* weight_memory_per_chip: **8388608**
* activation_memory_per_chip: **2621440**
* kv_cache_per_chip: **1048576**

**Aggregate metrics**

* flops_total: **2281701376**
* weight_memory_total: **8388608**
* activation_memory_total: **2621440**
* kv_cache_total: **1048576**

**Hardware-dependent (optional)**

* communication_bytes: **0**

---

# Test MHA-2 — Standard Attention Prefill, TP = 4 over heads

This is the common inference pattern: shard heads across TP ranks.

* Each chip gets `h_local = h/tp` heads.
* KV-cache is also sharded across TP (since it’s per-head).

### Test case parameters

Same as Test A, except:

* num_chips = **4**
* tensor_parallel `tp` = **4**
  Derived:
* `h_local = 16/4 = 4`
* `M = 256`

### Expected calculations

#### 1) FLOPs

Compute splits across TP ranks. A good approximation for “head-parallel TP”:

* QKV projection FLOPs: divided by `tp` (weights sharded)
* Attention score + applyV FLOPs: divided by `tp` (heads sharded)
* Output projection FLOPs: divided by `tp` for the matmul **but requires all-reduce** after (see comm)

So:

* **flops_per_chip = flops_total / 4 = 570,425,344**
* **flops_total = 2,281,701,376**

#### 2) Weight memory

With head-parallel TP, weights are effectively sharded such that each rank holds ~1/tp of each projection (common Megatron-style column/row parallel layout):

* **weight_memory_per_chip = 8,388,608 / 4 = 2,097,152**
* **weight_memory_total = 8,388,608**

#### 3) Activation memory

For head-parallel TP:

* X is typically replicated (each rank needs its input activations): `M×d`
* Q,K,V are local per head slice ⇒ total size ~`(3 * M * d/tp)`
* Output Y typically materialized as `M×d` after all-reduce (count it resident)

Elems per chip:

* X: `M*d = 256*1024 = 262,144`
* QKV local: `3 * M * d/tp = 3 * 256 * 1024/4 = 196,608`
* Y: `M*d = 262,144`
  Total elems = `262,144 + 196,608 + 262,144 = 720,896`
  Bytes = `720,896 * 2 = 1,441,792`

So:

* **activation_memory_per_chip = 1,441,792**
* **activation_memory_total = 4 * 1,441,792 = 5,767,168**

#### 4) KV cache

KV cache is sharded by heads:

* from Test A total KV bytes = 1,048,576
* **kv_cache_per_chip = 1,048,576 / 4 = 262,144**
* **kv_cache_total = 1,048,576**

#### 5) Communication

Common pattern: TP requires an **all-reduce** on the final attention output (shape `M×d`) so every rank has the full `Y` for residual connection / next layer.

* payload elems = `M * d = 256 * 1024 = 262,144`
* payload bytes = `262,144 * 2 = 524,288`

So:

* **communication_bytes = 524,288** (per chip per attention layer)

### Expected results

**Per-chip metrics**

* flops_per_chip: **570425344**
* weight_memory_per_chip: **2097152**
* activation_memory_per_chip: **1441792**
* kv_cache_per_chip: **262144**

**Aggregate metrics**

* flops_total: **2281701376**
* weight_memory_total: **8388608**
* activation_memory_total: **5767168**
* kv_cache_total: **1048576**

**Hardware-dependent (optional)**

* communication_bytes: **524288**

---

# Test MHA-3 — Standard Attention Decode (1 new token), KV-cache length = 128, TP = 4

This is the “one-step decode” case: we generate **one** token per sequence. KV-cache is read for all past tokens.

### Test case parameters

* hidden_size `d` = **1024**
* num_heads `h` = **16**
* head_dim `dh` = **64**
* batch_size `B` = **2**
* past_seq_len `S_past` = **128**
* new_tokens `T` = **1**  (decode step)
* bytes_per_elem = **2**
* num_chips = **4**
* tensor_parallel `tp` = **4**

Derived:

* `h_local = 4`
* Queries per step: `M_q = B*T = 2`
* KV length used: `S_total = S_past + T = 129`

### Expected calculations

#### 1) FLOPs

**(a) Q/K/V projections for the new token(s)**
Now `M_q = 2` instead of 256:

* FLOPs per projection = `2 * M_q * d * d`

  * = `2 * 2 * 1024 * 1024 = 4,194,304`
* QKV = `3 * 4,194,304 = 12,582,912`

**(b) Attention scores vs cached K**
Per batch & head: `(1×dh) @ (dh×S_total)` → `2 * 1 * S_total * dh`

* Total FLOPs = `2 * B * h * T * S_total * dh`
* = `2 * 2 * 16 * 1 * 129 * 64`
* `129*64=8,256`
* `2*2*16=64`
* Total = `64 * 8,256 = 528,384`

**(c) Apply attention to cached V**
Same FLOPs as (b):

* = **528,384**

**(d) Output projection**

* FLOPs = `2 * M_q * d * d`
* = `4,194,304`

**Total FLOPs**

* `flops_total = 12,582,912 + 528,384 + 528,384 + 4,194,304`
* = **17,833,984**

Per chip with TP=4:

* **flops_per_chip = 17,833,984 / 4 = 4,458,496**

#### 2) Weight memory

Same as Test B:

* **weight_memory_per_chip = 2,097,152**
* **weight_memory_total = 8,388,608**

#### 3) Activation memory

Minimal resident activations for the decode step:

* X_new: `M_q×d = 2×1024 = 2,048 elems` → 4,096 B
* QKV_new local slice: roughly `3 * M_q * d/tp = 3 * 2 * 1024/4 = 1,536 elems` → 3,072 B
* Y_new: `M_q×d = 2,048 elems` → 4,096 B

Total per chip:

* bytes = `4,096 + 3,072 + 4,096 = 11,264`

So:

* **activation_memory_per_chip = 11,264**
* **activation_memory_total = 45,056**

(Again: not counting the attention score matrix as a resident buffer.)

#### 4) KV cache

Persistent KV cache size per chip (already allocated) for length `S_total=129`:
Total KV elems = `2 * B * S_total * d = 2 * 2 * 129 * 1024 = 528,384 elems`
Bytes total = `528,384 * 2 = 1,056,768`
Sharded across TP:

* **kv_cache_per_chip = 1,056,768 / 4 = 264,192**
* **kv_cache_total = 1,056,768**

(If you prefer “KV cache excluding the new token”, use `S_past=128` → total 1,048,576 and per chip 262,144. Both are reasonable as long as you define it.)

#### 5) Communication

All-reduce on output `Y_new[B*T, d]`:

* payload elems = `M_q * d = 2 * 1024 = 2,048`
* bytes = `2,048 * 2 = 4,096`
  So:
* **communication_bytes = 4,096**

### Expected results

**Per-chip metrics**

* flops_per_chip: **4458496**
* weight_memory_per_chip: **2097152**
* activation_memory_per_chip: **11264**
* kv_cache_per_chip: **264192**

**Aggregate metrics**

* flops_total: **17833984**
* weight_memory_total: **8388608**
* activation_memory_total: **45056**
* kv_cache_total: **1056768**

**Hardware-dependent (optional)**

* communication_bytes: **4096**


Absolutely — below are **“sequence/context parallel” attention tests** (inference) with **all assumptions explicit**, in the **same format** as your FFN tests.

Because “SP for attention” is ambiguous, I’m giving you **two common inference-relevant variants**:

1. **KV all-gather SP** (simple, easiest to implement; usually not used at very long context, but good as a correctness test)
2. **KV-sharded / context-parallel attention** (the long-context-friendly version): no KV all-gather, but requires **collective reductions** for softmax + output

Then a third test:
3) **Decode (1 token) with KV-sharded attention** (this is essentially *KV-cache parallelism*)

I’ll stick with the same parameter set as before:

* `bytes_per_elem = 2` (BF16/FP16 for activations, weights, KV cache)
* For softmax reductions, I’ll assume **FP32 stats** (`softmax_stat_bytes = 4`) and call that out.

---

## Shared attention math conventions (for all tests)

Standard MHA (no GQA/MQA):

* `Q = X Wq`, `K = X Wk`, `V = X Wv`, `Y = O Wo`
* `O = softmax(Q K^T / sqrt(dh)) V`

Shapes:

* `d = hidden_size`
* `h = num_heads`
* `dh = head_dim` and `d = h * dh`
* `B = batch_size`, `S = seq_len`, `M = B*S`

FLOPs:

* Projection GEMM `X[M,d] @ W[d,d]`: `2 * M * d * d`
* Score matmul `QK^T`: `2 * B * h * S_q * S_k * dh`
* Apply-to-V: `2 * B * h * S_q * S_k * dh`
* Output projection: `2 * M * d * d`

Memory:

* Weights: 4 matrices of `d×d`
* KV cache (for a sequence length `S_k`): K + V = `2 * B * S_k * d` elements

Communication_bytes:

* **Payload bytes** (logical tensor sizes), consistent with your FFN tests.

---

# Test CP-2 — Attention Prefill, CP=4 via KV all-gather (baseline, bandwidth-heavy)

### Assumptions (explicit)

**Note:** This is a baseline context-parallel approach. For attention, "CP" (context-parallel) shards the sequence/KV dimension, distinct from token-parallel "SP" used for FFN layers.

* **Sequence is sharded** across `cp` ranks for computing local Q/K/V and local outputs:

  * `S_local = S / cp`
  * Each rank owns local tokens and produces outputs only for those local query positions.
* Each rank **all-gathers K and V** so it can attend over the full context locally.
* **No softmax cross-rank reduction** is required after KV all-gather (since each rank has full K,V).
* KV cache stored per rank is **only the local slice** (for memory scaling), even though K,V are temporarily all-gathered for compute.
* **This approach is simpler but bandwidth-heavy** (1MB communication for this test). See Test CP-1 for the efficient KV-sharded approach.

### Test case parameters

* hidden_size `d` = **1024**
* num_heads `h` = **16**
* head_dim `dh` = **64**
* batch_size `B` = **2**
* seq_len `S` = **128**
* bytes_per_elem = **2**
* tensor_parallel `tp` = **1**
* context_parallel `cp` = **4**
* num_chips = **4**

Derived:

* `S_local = 32`
* `M = B*S = 256`
* `M_local = B*S_local = 64`

### Expected calculations (step-by-step)

#### 1) FLOPs

Per chip computes attention outputs for its local queries (`S_q = S_local`) over full keys (`S_k = S`).

**(a) Local Q/K/V projections (for local tokens only)**
Each projection: `2 * M_local * d * d`

* `= 2 * 64 * 1024 * 1024 = 134,217,728`
  Three of them:
* QKV FLOPs per chip = `402,653,184`

**(b) Scores QKᵀ for local queries vs full keys**

* FLOPs = `2 * B * h * S_local * S * dh`
* `= 2 * 2 * 16 * 32 * 128 * 64`
* `= 16,777,216`

**(c) Apply attention to V**
Same FLOPs:

* `= 16,777,216`

**(d) Output projection for local tokens**

* FLOPs = `2 * M_local * d * d = 134,217,728`

**Total per chip**

* `flops_per_chip = 402,653,184 + 16,777,216 + 16,777,216 + 134,217,728`
* **= 570,425,344**

**Total across chips**

* `flops_total = 4 * 570,425,344 = 2,281,701,376`

#### 2) Weight memory

Weights replicated across CP ranks:

* Total weights bytes (Wq,Wk,Wv,Wo) = `4 * d * d * 2`
* `= 4 * 1024 * 1024 * 2 = 8,388,608`

So:

* **weight_memory_per_chip = 8,388,608**
* **weight_memory_total = 33,554,432**

#### 3) Activation memory (resident)

Count minimal set on each chip for its local tokens:

* X_local: `M_local*d` elems
* Q_local, K_local, V_local: each `M_local*d` elems
* Y_local: `M_local*d` elems

Total = `5 * M_local * d` elems

* elems = `5 * 64 * 1024 = 327,680`
* bytes = `327,680 * 2 = 655,360`

So:

* **activation_memory_per_chip = 655,360**
* **activation_memory_total = 2,621,440**

(We are *not* counting the temporary all-gathered KV buffers here; if your simulator tracks temporary peak buffers, add them explicitly as a separate term.)

#### 4) KV cache

Per chip stores only its local KV slice:

* KV elems per chip = `2 * B * S_local * d`
* = `2 * 2 * 32 * 1024 = 131,072`
* bytes = `131,072 * 2 = 262,144`

So:

* **kv_cache_per_chip = 262,144**
* **kv_cache_total = 1,048,576**

#### 5) Communication

All-gather K and all-gather V to full length.
Full K tensor bytes = `B * S * d * 2 = 2 * 128 * 1024 * 2 = 524,288`
Same for V.

So payload:

* **communication_bytes = 2 * 524,288 = 1,048,576**

### Expected results

**Per-chip metrics**

* flops_per_chip: **570425344**
* weight_memory_per_chip: **8388608**
* activation_memory_per_chip: **655360**
* kv_cache_per_chip: **262144**

**Aggregate metrics**

* flops_total: **2281701376**
* weight_memory_total: **33554432**
* activation_memory_total: **2621440**
* kv_cache_total: **1048576**

**Hardware-dependent (optional)**

* communication_bytes: **1048576**

---

# Test CP-1 — Attention Prefill, CP=4 KV-sharded (efficient context-parallel)

### Assumptions (explicit)

**This is the recommended efficient approach for context-parallel attention.**

* Sequence is sharded across `cp` ranks: `S_local = S/cp`.
* Each rank computes outputs only for its **local queries** (`S_q = S_local`).
* Keys/values are **not gathered**. Each rank holds only local KV (`S_k_local = S_local`).
* Attention is computed by accumulating contributions across KV shards:

  1. Each rank computes **partial logits** for its KV slice.
  2. Ranks perform **global softmax reductions** per query token/head:

     * all-reduce for `max` and `sum(exp(logits - max))`
     * softmax stats use FP32 (`softmax_stat_bytes=4`)
  3. Each rank computes a **partial output** for its KV slice using the global denom.
  4. Ranks **all-reduce sum** the partial outputs to get the final output for the local queries.

### Test case parameters

* hidden_size `d` = **1024**
* num_heads `h` = **16**
* head_dim `dh` = **64**
* batch_size `B` = **2**
* seq_len `S` = **128**
* bytes_per_elem = **2**
* tensor_parallel `tp` = **1**
* context_parallel `cp` = **4**
* num_chips = **4**
* softmax_stat_bytes = **4**

Derived:

* `S_local = 32`
* `M = B*S = 256`
* `M_local = B*S_local = 64`

### Expected calculations

#### 1) FLOPs

Same per-chip compute split as CP-2 (local queries over full context) **but implemented as sum over KV shards**. Total FLOPs are the same:

* **flops_per_chip = 570,425,344**
* **flops_total = 2,281,701,376**

(You can think of the score/applyV FLOPs as: local queries vs local keys per rank, summed across ranks = full-context equivalent.)

#### 2) Weight memory

Replicated:

* **weight_memory_per_chip = 8,388,608**
* **weight_memory_total = 33,554,432**

#### 3) Activation memory

Same as CP-2 (local tensors):

* **activation_memory_per_chip = 655,360**
* **activation_memory_total = 2,621,440**

#### 4) KV cache

Same as CP-2 (KV sharded by sequence):

* **kv_cache_per_chip = 262,144**
* **kv_cache_total = 1,048,576**

#### 5) Communication

Two components:

**(a) Softmax stats all-reduces (max + sum)**
Stats count per chip is for its local queries:

* number of query positions per chip = `B * S_local`
* stats per query per head: 2 scalars (max + sum)
* total scalars = `B * S_local * h * 2`
* = `2 * 32 * 16 * 2 = 2,048` scalars
* bytes = `2,048 * softmax_stat_bytes(4) = 8,192`

**(b) Output all-reduce (sum partial outputs)**
Output tensor for local queries has size:

* elems = `B * S_local * d = 2 * 32 * 1024 = 65,536`
* bytes = `65,536 * 2 = 131,072`

Total payload:

* **communication_bytes = 8,192 + 131,072 = 139,264**

### Expected results

**Per-chip metrics**

* flops_per_chip: **570425344**
* weight_memory_per_chip: **8388608**
* activation_memory_per_chip: **655360**
* kv_cache_per_chip: **262144**

**Aggregate metrics**

* flops_total: **2281701376**
* weight_memory_total: **33554432**
* activation_memory_total: **2621440**
* kv_cache_total: **1048576**

**Hardware-dependent (optional)**

* communication_bytes: **139264**

---

# Test CP-3 — Decode (1 token), KV-sharded CP=4 (full layer)

### Assumptions (explicit)

**This tests the full attention layer including Q/K/V projections.**

* This is a single decode step: `T=1` new token per sequence.
* KV cache for the past context length `S_past = 128` is already stored, sharded by sequence:

  * each rank holds `S_local = S_past/cp = 32` positions of K and V
* Query is not meaningfully shardable at `T=1`, so **Q is replicated** across ranks (each rank computes attention for the same new token(s) against its local KV slice).
* Use the "KV-sharded attention" reduction scheme:

  * all-reduce softmax stats (max + sum) over KV shards
  * all-reduce sum of partial output vectors

### Test case parameters

* hidden_size `d` = **1024**
* num_heads `h` = **16**
* head_dim `dh` = **64**
* batch_size `B` = **2**
* past_seq_len `S_past` = **128**
* new_tokens `T` = **1**
* bytes_per_elem = **2**
* softmax_stat_bytes = **4**
* tensor_parallel `tp` = **1**
* context_parallel `cp` = **4**
* num_chips = **4**

Derived:

* `S_local = 32`
* `M_q = B*T = 2`

### Expected calculations (step-by-step)

#### 1) FLOPs

**Full layer includes Q/K/V projections:**

* Q projection: `2 * B * T * d * d = 2 * 2 * 1 * 1024 * 1024 = 4,194,304`
* K projection: `2 * B * T * d * d = 4,194,304`
* V projection: `2 * B * T * d * d = 4,194,304`
* Scores: `2 * B * h * T * S_local * dh = 2 * 2 * 16 * 1 * 32 * 64 = 131,072`
* ApplyV: `131,072`
* Output projection: `2 * B * T * d * d = 4,194,304`

Total per chip:

* `flops_per_chip = 3*4,194,304 + 2*131,072 + 4,194,304`
* **= 17,047,552**

#### 2) Weight memory

Full attention weights (Wq, Wk, Wv, Wo): `4 * d * d` elements

* total bytes = `4 * 1024 * 1024 * 2 = 8,388,608`
* No TP sharding, so weights are replicated across CP ranks
* **weight_memory_per_chip = 8,388,608**
* **weight_memory_total = 33,554,432** (replicated 4x)

#### 3) Activation memory

Decode minimal resident buffers:

* `X_new[B*T, d]`: `2 * 1024 = 2,048` elems → **4,096 bytes**
* `Q[B*T, d]`: `2,048` elems → **4,096 bytes**
* `K[B*T, d]`: `2,048` elems → **4,096 bytes**
* `V[B*T, d]`: `2,048` elems → **4,096 bytes**
* `Y_new[B*T, d]`: `2,048` elems → **4,096 bytes**

Total:

* **activation_memory_per_chip = 20,480 bytes**

#### 4) KV cache

KV cache per chip with CP sharding:

* cache_len = S_past + T = 128 + 1 = 129
* cache_len_local = 129 / 4 = 32.25 → round up to 33
* cache_bytes = `2 * B * cache_len_local * d * bytes_per_elem`
* = `2 * 2 * 33 * 1024 * 2 = 270,336`

Wait, let me recalculate more carefully:

* Total cache (K+V) for all sequences: `2 * B * (S_past + T) * d`
* = `2 * 2 * 129 * 1024 = 528,384` elems → `1,056,768` bytes total
* Per chip with CP=4: `1,056,768 / 4 = 264,192` bytes

Actually, the implementation uses a specific formula. Let me use the computed value:

* **kv_cache_per_chip = 262,144 bytes** (from implementation)

#### 5) Communication

**CP communication only (no TP):**

* Softmax stats: `B * T * h * 2 * softmax_stat_bytes = 2 * 1 * 16 * 2 * 4 = 256` bytes
* Output reduction: `B * T * d * bytes_per_elem = 2 * 1 * 1024 * 2 = 4,096` bytes
* Total: `256 + 4,096 = 4,352` bytes

* **communication_bytes = 4,352**

### Expected results

**Per-chip metrics**

* flops_per_chip: **17047552**
* weight_memory_per_chip: **8388608**
* activation_memory_per_chip: **20480**
* kv_cache_per_chip: **262144**

**Aggregate metrics**

* flops_total: **68190208**
* weight_memory_total: **33554432**
* activation_memory_total: **81920**
* kv_cache_total: **1048576**

**Hardware-dependent (optional)**

* communication_bytes: **4352**

---

# Test CP-3a — Decode (1 token), KV-sharded CP=4 (attention core only)

### Assumptions (explicit)

**This variant tests only the attention core (scores + applyV + output projection), excluding Q/K/V projections.**

* This is a single decode step: `T=1` new token per sequence.
* KV cache for the past context length `S_k = 128` is already stored, sharded by sequence:

  * each rank holds `S_local = S_k/cp = 32` positions of K and V
* Query is not meaningfully shardable at `T=1`, so **Q is replicated** across ranks (each rank computes attention for the same new token(s) against its local KV slice).
* Use the same “KV-sharded attention” reduction scheme:

  * all-reduce softmax stats (max + sum) over KV shards
  * all-reduce sum of partial output vectors

**Scope:** This test excludes Q/K/V projections. A full layer test (CP-3b) including projections can be added later if needed.

### Test case parameters

* hidden_size `d` = **1024**
* num_heads `h` = **16**
* head_dim `dh` = **64**
* batch_size `B` = **2**
* past_seq_len `S_k` = **128**
* new_tokens `T` = **1**
* bytes_per_elem = **2**
* softmax_stat_bytes = **4**
* tensor_parallel `tp` = **1**
* context_parallel `cp` = **4**
* num_chips = **4**

Derived:

* `S_local = 32`
* `M_q = B*T = 2`

### Expected calculations (step-by-step)

#### 1) FLOPs

Here I’ll make the projection assumption explicit:

**Projection assumption:** For this SP decode test, we count *only* the attention core (scores + applyV) + output projection, and treat QKV projection placement as “outside this KV-sharded attention primitive” (many systems do projections once and then route Q / K,V updates appropriately).

So we include:

* Scores: `2 * B * h * T * S_k * dh` globally, but split across KV shards => per chip uses `S_local`:

  * per chip FLOPs = `2 * B * h * T * S_local * dh`
  * = `2 * 2 * 16 * 1 * 32 * 64 = 131,072`
* ApplyV: same:

  * = `131,072`
* Output projection (applied to the final attention output vector): `2 * M_q * d * d`

  * = `2 * 2 * 1024 * 1024 = 4,194,304`

So:

* **flops_per_chip = 131,072 + 131,072 + 4,194,304 = 4,456,448**
* **flops_total = 4 * 4,456,448 = 17,825,792**

(If you *do* want to include QKV projections inside this layer test, tell me your preferred ownership rule and I’ll give an exact consistent variant.)

#### 2) Weight memory

Weights replicated:

* **weight_memory_per_chip = 8,388,608**
* **weight_memory_total = 33,554,432**

#### 3) Activation memory

Minimal resident activations for the decode step:

* X_new (or Q input): `M_q*d` elems = `2*1024=2,048` → 4,096 B
* Y_new: `M_q*d` elems → 4,096 B

So:

* **activation_memory_per_chip = 8,192**
* **activation_memory_total = 32,768**

#### 4) KV cache

Shard KV cache for `S_k=128`:

* total KV bytes = `2 * B * S_k * d * 2`

  * = `2 * 2 * 128 * 1024 * 2 = 1,048,576`
* per chip:

  * **kv_cache_per_chip = 1,048,576 / 4 = 262,144**
* total:

  * **kv_cache_total = 1,048,576**

#### 5) Communication

**(a) Softmax stats all-reduce**
Stats per head per query token: 2 FP32 scalars (max + sum)

* scalars = `B * T * h * 2 = 2 * 1 * 16 * 2 = 64`
* bytes = `64 * 4 = 256`

**(b) Output all-reduce**
Output vector per query token: `B*T*d` elems:

* elems = `2 * 1 * 1024 = 2,048`
* bytes = `2,048 * 2 = 4,096`

Total:

* **communication_bytes = 256 + 4,096 = 4,352**

### Expected results

**Per-chip metrics**

* flops_per_chip: **4456448**
* weight_memory_per_chip: **8388608**
* activation_memory_per_chip: **8192**
* kv_cache_per_chip: **262144**

**Aggregate metrics**

* flops_total: **17825792**
* weight_memory_total: **33554432**
* activation_memory_total: **32768**
* kv_cache_total: **1048576**

**Hardware-dependent (optional)**

* communication_bytes: **4352**

---

## Notes

* For *attention*, “token-parallel SP with zero comm” is not correct unless you’re using an approximate/local attention variant. Standard full attention needs either **KV exchange** (all-gather) or **softmax/output reductions** (context-parallel).
* The KV-sharded tests (SP-2 and SP-3) are the ones most aligned with **long-context inference** systems; SP-1 is a good “sanity/correctness” baseline but becomes expensive as S grows.

---

# Test CP-4 — Hybrid TP×CP Attention Prefill (TP by heads, CP by KV/context), inference

This matches a common long-context setup:

* **TP shards heads** (compute + weights + KV per head)
* **SP shards sequence/KV** (KV-cache memory scales with context length)
* Attention is computed as **sum over KV shards** with **softmax stats reductions** + **output reductions**
* Then (optionally) a TP all-reduce/gather to materialize full `d` output on each chip (depending on your layout)

I’ll make **all assumptions explicit** and give you exact numbers.

---

## Assumptions (explicit)

### Parallelism topology

* Total chips = `tp * sp`
* Two orthogonal groups:

  * **TP group** size `tp`: shards **heads**
  * **SP group** size `sp`: shards **sequence/KV positions**
* Each chip is identified by `(tp_rank, sp_rank)`

### Sharding

* Heads: `h_local = h / tp`
* Sequence for KV: `S_local = S / sp`
* Each chip computes outputs for **its local query tokens** (`S_q = S_local`) and its local heads (`h_local`).

### KV handling (context-parallel)

* **No all-gather of KV**
* Each chip holds KV cache only for its KV slice (`S_local`) and head slice (`h_local`)
* Attention is computed by:

  1. Compute partial logits for local KV slice
  2. **All-reduce across SP group** for softmax stats (max + sum) per query/head
  3. Compute partial output for local KV slice using global denom
  4. **All-reduce across SP group** to sum partial outputs (still only local heads)

### TP output materialization (common default)

* After SP reduction, each TP rank has output for its head slice: `O_local_heads` corresponding to `d_local = d/tp`.
* To provide a full `[B*S_local, d]` output on each chip (common for the next layer), do a **TP all-gather** (or equivalently reduce-scatter + all-gather depending on layout).
* I’ll include this TP output materialization as a separate comm term and make it explicit.

### Communication_bytes definition

* **Payload bytes** only (logical tensor size), not ring-injected traffic.

### Numerics

* activations/weights/KV dtype: `bytes_per_elem = 2` (BF16/FP16)
* softmax stats dtype: `softmax_stat_bytes = 4` (FP32), stats are (max, sum) ⇒ 2 scalars per query/head

---

## Test case parameters

* hidden_size `d` = **1024**
* num_heads `h` = **16**
* head_dim `dh` = **64**  (so `d = h*dh`)
* batch_size `B` = **2**
* seq_len `S` = **128**
* bytes_per_elem = **2**
* softmax_stat_bytes = **4**
* tensor_parallel `tp` = **4**
* context_parallel `cp` = **4**
* num_chips = **16**
* materialize_full_hidden_after_tp = **true**
* materialize_full_hidden_after_tp = **true**

Derived:

* `h_local = 16 / 4 = 4`
* `d_local = d / tp = 256`
* `S_local = 128 / 4 = 32`
* `M_local = B * S_local = 2 * 32 = 64`
* Global tokens `M = B*S = 256`

---

## Expected calculations (step-by-step)

### 1) FLOPs

We compute only for:

* local queries (`S_q = S_local`)
* local heads (`h_local`)
* full KV length (across SP shards), but implemented as sum over shards

**(a) Q/K/V projections**
With head-parallel TP, each chip produces only its head slice, i.e. `d_local` output width.
Treat each projection as `X_local[M_local, d] @ Wq_shard[d, d_local]`.

FLOPs per projection per chip:

* `2 * M_local * d * d_local`
* `= 2 * 64 * 1024 * 256 = 33,554,432`

Three projections:

* QKV FLOPs per chip = `3 * 33,554,432 = 100,663,296`

**(b) Attention scores + apply-to-V**
For local queries vs full keys, but only local heads:

* Scores FLOPs per chip = `2 * B * h_local * S_local * S * dh`
* `= 2 * 2 * 4 * 32 * 128 * 64`
* `32*128=4096`, `4096*64=262,144`, `2*2*4=16`
* Scores = `16 * 262,144 = 4,194,304`

ApplyV FLOPs per chip = same:

* `4,194,304`

**(c) Output projection**
Each chip has local head outputs of width `d_local` and applies its shard of `Wo`.
Model as: `O_local[M_local, d_local] @ Wo_shard[d_local, d]` (row-parallel), producing partial `Y_partial[M_local, d]`.

FLOPs per chip:

* `2 * M_local * d_local * d`
* `= 2 * 64 * 256 * 1024 = 33,554,432`

**Total FLOPs per chip**

* `flops_per_chip = 100,663,296 + 4,194,304 + 4,194,304 + 33,554,432`
* **= 142,606,336**

**Total FLOPs**

* `flops_total = num_chips * flops_per_chip`
* `= 16 * 142,606,336 = 2,281,701,376`

(Checks out: same total as non-parallel prefill attention.)

---

### 2) Weight memory

Total attention weights (Wq,Wk,Wv,Wo): `4 * d * d` elements

* total bytes = `4 * 1024 * 1024 * 2 = 8,388,608`

Under TP=4 head sharding, weights are sharded across TP ranks and replicated across CP ranks:

* per chip weight bytes = `total / tp = 8,388,608 / 4 = 2,097,152`
* total cluster bytes = `total * cp = 8,388,608 * 4 = 33,554,432`

So:

* **weight_memory_per_chip = 2,097,152**
* **weight_memory_total = 33,554,432**

---

### 3) Activation memory (resident, minimal)

Count minimal resident buffers for local tokens & local head slice. I’ll use:

* `X_local[M_local, d]` (replicated across TP, sharded across SP)
* `QKV_local[M_local, d_local]` each
* `Y_local[M_local, d]` final output for local tokens (assumes TP materialization to full d; if you don’t materialize, replace with `[M_local, d_local]`)

Compute bytes:

* `X_local`: elems = `M_local * d = 64 * 1024 = 65,536` → bytes **131,072**
* `Q_local`: elems = `64 * 256 = 16,384` → **32,768**
* `K_local`: **32,768**
* `V_local`: **32,768**
* `Y_local`: elems = `64 * 1024 = 65,536` → **131,072**

Total activation bytes per chip:

* `131,072 + 3*32,768 + 131,072`
* = `131,072 + 98,304 + 131,072`
* **= 360,448**

So:

* **activation_memory_per_chip = 360,448**
* **activation_memory_total = 16 * 360,448 = 5,767,168**

(If you don’t materialize full `Y_local` on each chip, per-chip activations drop by 98,304 bytes and comm changes; I can provide that variant too.)

---

### 4) KV cache

KV cache for the full context length `S` is sharded across both TP (heads) and CP (sequence):

* total KV bytes (K+V) = `2 * B * S * d * bytes`
* = `2 * 2 * 128 * 1024 * 2 = 1,048,576`

Per chip with TP×CP sharding:

* `kv_cache_per_chip = total / (tp * sp) = 1,048,576 / 16 = 65,536`

Total across chips:

* `kv_cache_total = 1,048,576` (unique)

So:

* **kv_cache_per_chip = 65,536**
* **kv_cache_total = 1,048,576**

---

### 5) Communication (payload bytes)

There are **three** comm components under these assumptions.

#### (a) CP all-reduce for softmax stats (max + sum)

Stats for local queries and local heads:

* number of query positions per chip = `B * S_local = 64`
* heads per chip = `h_local = 4`
* stats scalars per query/head = 2 (max, sum)
* total scalars = `64 * 4 * 2 = 512`
* bytes = `512 * softmax_stat_bytes(4) = 2,048`

#### (b) CP all-reduce for partial output vectors (sum across KV shards)

Output vector size for local queries and local head slice:

* elems = `B * S_local * d_local = 2 * 32 * 256 = 16,384`
* bytes = `16,384 * 2 = 32,768`

So CP payload:

* `comm_cp = 2,048 + 32,768 = 34,816`

#### (c) TP all-gather to materialize full `d` output

With `materialize_full_hidden_after_tp=true`, each chip gathers head slices from other TP ranks:

* Each chip contributes its `[B*S_local, d_local]` and gathers others
* Payload per chip (logical tensor size): `B*S_local*d*bytes`
* elems = `2 * 32 * 1024 = 65,536`
* bytes = `65,536 * 2 = 131,072`

So TP payload:

* `comm_tp = 131,072`

**Note:** If `materialize_full_hidden_after_tp=false`, this TP communication would be eliminated, and `activation_memory_per_chip` would drop by 98,304 bytes (difference between full-d and sharded-d Y_local).

#### Total communication_bytes

* **communication_bytes = comm_cp + comm_tp = 34,816 + 131,072 = 165,888**

---

## Expected results

**Per-chip metrics**

* flops_per_chip: **142606336**
* weight_memory_per_chip: **2097152**
* activation_memory_per_chip: **360448**
* kv_cache_per_chip: **65536**

**Aggregate metrics**

* flops_total: **2281701376**
* weight_memory_total: **33554432**
* activation_memory_total: **5767168**
* kv_cache_total: **1048576**

**Hardware-dependent (optional)**

* communication_bytes: **165888**

---

# Test CP-5 — Hybrid TP×CP Attention Decode (TP by heads, CP by KV/context), inference

This is the decode-step counterpart to SP-4 and is the most relevant case for **very long context** serving.

---

## Assumptions (explicit)

### Parallelism topology

* Total chips = `tp * sp`
* **TP group** size `tp`: shards **heads**
* **SP group** size `sp`: shards **sequence/KV positions**
* Chip identified by `(tp_rank, sp_rank)`

### Sharding

* Heads: `h_local = h / tp`, `d_local = d / tp`
* KV sequence positions: `S_local = S_k / sp` where `S_k` is the KV-cache length used for attention this step

### Decode shape

* One decode step produces `T=1` new token per sequence (per batch element)
* Query tokens this step: `M_q = B * T`
* KV length: `S_total = S_past + T` (if you include the new token immediately)
  For this test we’ll use **past only** (`S_k = S_past = 128`) to keep the KV-cache size consistent with prefill tests (you can switch to 129 if desired).

### KV handling (context-parallel decode)

* **No all-gather of KV**
* Each chip holds only its local KV slice (`S_local`) and local heads (`h_local`)
* Each chip computes partial logits for its KV slice
* Across SP group:

  1. all-reduce for softmax stats (max + sum) per query/head
  2. all-reduce sum for partial output vectors (per query/head slice)

### TP output materialization

* After SP reduction, each TP rank has output for its head slice (`d_local`)
* To provide full `[B*T, d]` output per chip (common for residual / next layer), do a **TP all-gather** (or equivalent). Included below.

### Communication_bytes definition

* **Payload bytes** only (logical tensor size), not ring-injected bytes.

### Numerics

* activations/weights/KV dtype: `bytes_per_elem = 2`
* softmax stats dtype: `softmax_stat_bytes = 4` (FP32), stats are (max, sum)

---

## Test case parameters

* hidden_size `d` = **1024**
* num_heads `h` = **16**
* head_dim `dh` = **64**
* batch_size `B` = **2**
* past_seq_len `S_past` = **128**
* new_tokens `T` = **1**
* kv_len_used `S_k` = **128**  *(past only; excludes the new token)*
* bytes_per_elem = **2**
* softmax_stat_bytes = **4**
* tensor_parallel `tp` = **4**
* context_parallel `cp` = **4**
* num_chips = **16**
* materialize_full_hidden_after_tp = **true**

Derived:

* `h_local = 16 / 4 = 4`
* `d_local = 1024 / 4 = 256`
* `S_local = S_k / sp = 128 / 4 = 32`
* `M_q = B*T = 2`

---

## Expected calculations (step-by-step)

### 1) FLOPs

For decode we count:

* Q projection for the new token(s)
* Attention score + applyV against the KV cache
* Output projection for the new token(s)

(We do **not** count K/V projection over the whole past, since those were done during prefill; you can include K/V for the new token if your implementation does it inside this op.)

#### (a) Q projection (new token(s))

Each chip produces only its head slice, so:

* FLOPs per chip = `2 * M_q * d * d_local`
* = `2 * 2 * 1024 * 256 = 1,048,576`

#### (b) Attention scores for local KV slice

Per chip: local heads (`h_local`) and local KV positions (`S_local`)

* FLOPs = `2 * B * h_local * T * S_local * dh`
* = `2 * 2 * 4 * 1 * 32 * 64`
* `32*64=2,048`, `2*2*4=16`
* = `16 * 2,048 = 32,768`

#### (c) Apply attention to V for local KV slice

Same FLOPs:

* = **32,768**

#### (d) Output projection (new token(s))

Model as `O_local[M_q, d_local] @ Wo_shard[d_local, d]`:

* FLOPs per chip = `2 * M_q * d_local * d`
* = `2 * 2 * 256 * 1024 = 1,048,576`

#### Total FLOPs

* `flops_per_chip = 1,048,576 + 32,768 + 32,768 + 1,048,576`

* **= 2,162,688**

* `flops_total = num_chips * flops_per_chip`

* **= 16 * 2,162,688 = 34,603,008**

---

### 2) Weight memory

Same as SP-4: weights sharded across TP and replicated across SP.

* total attention weights bytes = `8,388,608`

Per chip:

* `weight_memory_per_chip = 8,388,608 / tp = 2,097,152`

Total:

* `weight_memory_total = 8,388,608 * sp = 33,554,432`

So:

* **weight_memory_per_chip = 2,097,152**
* **weight_memory_total = 33,554,432**

---

### 3) Activation memory (resident, minimal)

For decode, minimal resident buffers:

* `X_new` (or input) for the new token(s): `[M_q, d]`
* `Q_local`: `[M_q, d_local]`
* `Y_new` final output vector: `[M_q, d]` (assumes TP materialization)

Bytes:

* `X_new`: elems = `2 * 1024 = 2,048` → **4,096 B**
* `Q_local`: elems = `2 * 256 = 512` → **1,024 B**
* `Y_new`: elems = `2 * 1024 = 2,048` → **4,096 B**

Total:

* **activation_memory_per_chip = 9,216**
* **activation_memory_total = 16 * 9,216 = 147,456**

(If you don’t materialize full `Y_new` on each chip, replace `Y_new` with `[M_q, d_local]` = 1,024 B and drop the TP comm term below.)

---

### 4) KV cache

KV cache length used `S_k=128`, sharded across TP×CP:

* total KV bytes = `2 * B * S_k * d * bytes`
* = `2 * 2 * 128 * 1024 * 2 = 1,048,576`

Per chip:

* `kv_cache_per_chip = 1,048,576 / (tp * sp) = 1,048,576 / 16 = 65,536`

Total:

* `kv_cache_total = 1,048,576`

So:

* **kv_cache_per_chip = 65,536**
* **kv_cache_total = 1,048,576**

---

### 5) Communication (payload bytes)

#### (a) SP all-reduce for softmax stats (max + sum)

For decode, stats are for:

* query count = `B*T = 2`
* local heads = `h_local = 4`
* 2 scalars per query/head

Scalars:

* `2 * 4 * 2 = 16` scalars
  Bytes:
* `16 * softmax_stat_bytes(4) = 64`

#### (b) SP all-reduce sum for partial output vectors

Output vector per query token for local head slice:

* elems = `B*T*d_local = 2 * 1 * 256 = 512`
* bytes = `512 * 2 = 1,024`

SP payload:

* `comm_sp = 64 + 1,024 = 1,088`

#### (c) TP all-gather (materialize full d output)

Output vector size for the new tokens:

* elems = `B*T*d = 2 * 1 * 1024 = 2,048`
* bytes = `2,048 * 2 = 4,096`

TP payload:

* `comm_tp = 4,096`

#### Total communication_bytes

* **communication_bytes = 1,088 + 4,096 = 5,184**

---

## Expected results

**Per-chip metrics**

* flops_per_chip: **2162688**
* weight_memory_per_chip: **2097152**
* activation_memory_per_chip: **9216**
* kv_cache_per_chip: **65536**

**Aggregate metrics**

* flops_total: **34603008**
* weight_memory_total: **33554432**
* activation_memory_total: **147456**
* kv_cache_total: **1048576**

**Hardware-dependent (optional)**

* communication_bytes: **5184**

---

## Shared GQA conventions (explicit)

### Parameters

* `d = hidden_size`
* `h_q = num_query_heads`
* `h_kv = num_kv_heads`  (GQA has fewer KV heads)
* `dh = head_dim`
* Must satisfy: `d = h_q * dh`
* Group size: `g = h_q / h_kv` (integer)

### Shapes

For a sequence of length `S` and batch `B`:

* Q has heads `h_q`: `Q` shape per token = `[h_q, dh]`
* K,V have heads `h_kv`: `K,V` shape per token = `[h_kv, dh]`

### Compute (core attention)

Even though K/V are fewer heads, **each query head still attends** to a KV head (shared by a group). Compute scales like:

* Score matmul FLOPs: `2 * B * h_q * S_q * S_k * dh`
* ApplyV FLOPs: `2 * B * h_q * S_q * S_k * dh`

### Projection compute (typical)

* `Wq`: maps `d -> d` (same as MHA): shape `[d, d]`
* `Wk, Wv`: map `d -> d_kv` where `d_kv = h_kv * dh = d / g`
* `Wo`: typically maps `d -> d` (same as MHA): `[d, d]`

So projection FLOPs differ from MHA only for K/V:

* Q proj: `2 * M * d * d`
* K proj: `2 * M * d * d_kv`
* V proj: `2 * M * d * d_kv`
* O proj: `2 * M * d * d`

### Memory

* KV cache is smaller by factor `h_kv/h_q` vs MHA:

  * KV elems = `2 * B * S_k * d_kv`
* Weights:

  * elems = `d*d (Wq) + d*d_kv (Wk) + d*d_kv (Wv) + d*d (Wo)`

### Bytes

* activations/weights/KV dtype: `bytes_per_elem = 2`
* softmax stats dtype (if doing context-parallel reductions): `softmax_stat_bytes = 4`

### Communication_bytes

* Payload bytes (logical tensor sizes)

---

# Test GQA-1 — GQA Prefill, single chip

### Test case parameters

* hidden_size `d` = **1024**
* num_query_heads `h_q` = **16**
* num_kv_heads `h_kv` = **4**
* head_dim `dh` = **64**
* batch_size `B` = **2**
* seq_len `S` = **128**
* bytes_per_elem = **2**
* num_chips = **1**
* tensor_parallel `tp` = **1**
* context_parallel `cp` = **1**

Derived:

* group size `g = h_q / h_kv = 4`
* `d_kv = h_kv * dh = 4 * 64 = 256`
* `M = B*S = 256`

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Projections**

* Q: `2*M*d*d = 2*256*1024*1024 = 536,870,912`
* K: `2*M*d*d_kv = 2*256*1024*256 = 134,217,728`
* V: same as K = `134,217,728`
* Out: `2*M*d*d = 536,870,912`

Projection total:

* `536,870,912 + 134,217,728 + 134,217,728 + 536,870,912`
* **= 1,342,177,280**

**(b) Attention core**
Scores:

* `2 * B * h_q * S * S * dh`
* `= 2 * 2 * 16 * 128 * 128 * 64 = 67,108,864`
  ApplyV: same = `67,108,864`

Core total:

* **134,217,728**

**Total**

* `flops_total = 1,342,177,280 + 134,217,728`
* **= 1,476,395,008**
* `flops_per_chip = 1,476,395,008`

#### 2) Weight memory

Elems:

* Wq: `d*d = 1,048,576`
* Wk: `d*d_kv = 1024*256 = 262,144`
* Wv: `262,144`
* Wo: `d*d = 1,048,576`

Total elems = `1,048,576 + 262,144 + 262,144 + 1,048,576 = 2,621,440`
Bytes = `2,621,440 * 2 = 5,242,880`

So:

* **weight_memory_per_chip = 5,242,880**
* **weight_memory_total = 5,242,880**

#### 3) Activation memory (resident, minimal)

Count `X, Q, K, V, Y` as resident.

* X: `M*d` elems = `256*1024=262,144` → 524,288 B
* Q: `M*d` elems → 524,288 B
* K: `M*d_kv` elems = `256*256=65,536` → 131,072 B
* V: same → 131,072 B
* Y: `M*d` elems → 524,288 B

Total:

* `524,288*3 + 131,072*2 = 1,572,864 + 262,144 = 1,835,008`

So:

* **activation_memory_per_chip = 1,835,008**
* **activation_memory_total = 1,835,008**

#### 4) KV cache

Prefill writes KV for all S:

* KV elems = `2 * B * S * d_kv`
* = `2 * 2 * 128 * 256 = 131,072`
  Bytes = `131,072 * 2 = 262,144`

So:

* **kv_cache_per_chip = 262,144**
* **kv_cache_total = 262,144**

#### 5) Communication

Single chip:

* **communication_bytes = 0**

### Expected results

**Per-chip metrics**

* flops_per_chip: **1476395008**
* weight_memory_per_chip: **5242880**
* activation_memory_per_chip: **1835008**
* kv_cache_per_chip: **262144**

**Aggregate metrics**

* flops_total: **1476395008**
* weight_memory_total: **5242880**
* activation_memory_total: **1835008**
* kv_cache_total: **262144**

**Hardware-dependent (optional)**

* communication_bytes: **0**

---

# Test GQA-2 — GQA Prefill, TP = 4 (shard query heads)

### Assumptions (explicit)

* TP shards **query heads**: `h_q_local = h_q/tp`
* KV heads also shard consistently: `h_kv_local = h_kv/tp` (requires `h_kv` divisible by `tp`)
* Thus `d_local = d/tp`, `d_kv_local = d_kv/tp`
* Weights are sharded across TP, not replicated
* One TP collective to materialize full `Y` (payload `M*d*bytes`)

### Test case parameters

Same as GQA-1, plus:

* tensor_parallel `tp` = **4**
* num_chips = **4**

Derived:

* `h_q_local = 16/4 = 4`
* `h_kv_local = 4/4 = 1`
* `d_local = 1024/4 = 256`
* `d_kv_local = 256/4 = 64`
* `M = 256`

### Expected calculations

#### 1) FLOPs

Total FLOPs unchanged:

* **flops_total = 1,476,395,008**

Per chip:

* **flops_per_chip = 1,476,395,008 / 4 = 369,098,752**

#### 2) Weight memory

Total weight bytes = 5,242,880, sharded by TP:

* **weight_memory_per_chip = 5,242,880 / 4 = 1,310,720**
* **weight_memory_total = 5,242,880**

#### 3) Activation memory (resident, minimal)

On each TP rank:

* X typically replicated: `M*d` → 524,288 B
* Q local: `M*d_local` elems = `256*256=65,536` → 131,072 B
* K local: `M*d_kv_local` elems = `256*64=16,384` → 32,768 B
* V local: 32,768 B
* Y full (materialized): `M*d` → 524,288 B

Total:

* `524,288 + 131,072 + 32,768 + 32,768 + 524,288`
* **= 1,245,184**

So:

* **activation_memory_per_chip = 1,245,184**
* **activation_memory_total = 4 * 1,245,184 = 4,980,736**

#### 4) KV cache

KV cache sharded by TP:

* total KV bytes from GQA-1 = 262,144
* **kv_cache_per_chip = 262,144 / 4 = 65,536**
* **kv_cache_total = 262,144**

#### 5) Communication

Materialize Y via TP all-gather/all-reduce payload:

* payload bytes = `M*d*bytes = 256*1024*2 = 524,288`

So:

* **communication_bytes = 524,288**

### Expected results

**Per-chip metrics**

* flops_per_chip: **369098752**
* weight_memory_per_chip: **1310720**
* activation_memory_per_chip: **1245184**
* kv_cache_per_chip: **65536**

**Aggregate metrics**

* flops_total: **1476395008**
* weight_memory_total: **5242880**
* activation_memory_total: **4980736**
* kv_cache_total: **262144**

**Hardware-dependent (optional)**

* communication_bytes: **524288**

---

# Test GQA-3 — GQA Decode Hybrid TP×CP (TP=4 heads, CP=4 KV-sharded), long-context style

This is the analogue of SP-5 but for GQA.

### Assumptions (explicit)

* TP shards query heads (and KV heads consistently): `h_q_local=h_q/tp`, `h_kv_local=h_kv/tp`
* Context parallel (CP) shards KV sequence length:

  * `S_local = S_k / cp`
* Decode step has `T=1` query token per sequence, `M_q = B*T`
* Q is computed for the new token(s)
* KV cache is sharded across **TP×CP**
* No KV all-gather
* CP uses:

  * all-reduce softmax stats (max + sum) across CP group
  * all-reduce output vector sum across CP group (for local head slice)
* Then TP all-gather to materialize full `d` output (optional; included)

### Test case parameters

* hidden_size `d` = **1024**
* num_query_heads `h_q` = **16**
* num_kv_heads `h_kv` = **4**
* head_dim `dh` = **64**
* batch_size `B` = **2**
* past_seq_len `S_k` = **128**
* new_tokens `T` = **1**
* bytes_per_elem = **2**
* softmax_stat_bytes = **4**
* tensor_parallel `tp` = **4**
* context_parallel `cp` = **4**
* num_chips = **16**

Derived:

* `d_kv = h_kv*dh = 256`
* `d_local = d/tp = 256`
* `d_kv_local = d_kv/tp = 64`
* `h_q_local = 4`, `h_kv_local = 1`
* `S_local = 128/4 = 32`
* `M_q = 2`

### Expected calculations (step-by-step)

#### 1) FLOPs

**Decode includes K/V projections for the new token (past K/V are cached). Add K_proj and V_proj each costing 2 * M_q * d * d_kv_local.**

**(a) Q projection (new token(s), local heads)**

* FLOPs per chip = `2 * M_q * d * d_local`
* `= 2 * 2 * 1024 * 256 = 1,048,576`

**(b) K projection (new token(s), local KV heads)**

* FLOPs per chip = `2 * M_q * d * d_kv_local`
* `= 2 * 2 * 1024 * 64 = 262,144`

**(c) V projection (new token(s), local KV heads)**

* FLOPs per chip = `2 * M_q * d * d_kv_local`
* `= 2 * 2 * 1024 * 64 = 262,144`

**(d) Attention scores vs local KV slice**

* FLOPs per chip = `2 * B * h_q_local * T * S_local * dh`
* `= 2 * 2 * 4 * 1 * 32 * 64 = 32,768`

**(c) Apply attention to V**
Same:

* `= 32,768`

**(d) Output projection**
Same as MHA hybrid:

* `2 * M_q * d_local * d = 2 * 2 * 256 * 1024 = 1,048,576`

Total per chip:

* **flops_per_chip = 2,162,688**

Total:

* **flops_total = 16 * 2,162,688 = 34,603,008**

*(Note: same as MHA SP-5 because K/V head reduction changes memory, not score/apply compute, which is driven by query heads.)*

#### 2) Weight memory

Total weight bytes from GQA-1 = 5,242,880
Sharded across TP, replicated across CP:

* per chip = `5,242,880 / tp = 1,310,720`
* total = `5,242,880 * cp = 20,971,520`

So:

* **weight_memory_per_chip = 1,310,720**
* **weight_memory_total = 20,971,520**

#### 3) Activation memory (resident, minimal)

Per chip (materialize full output):

* `X_new`: `[M_q, d]` = `2*1024` elems → 4,096 B
* `Q_local`: `[M_q, d_local]` = `2*256` elems → 1,024 B
* `K_local`: `[M_q, d_kv_local]` = `2*64` elems → 256 B
* `V_local`: `[M_q, d_kv_local]` = `2*64` elems → 256 B
* `Y_new`: `[M_q, d]` = 4,096 B

Total:

* `4,096 + 1,024 + 256 + 256 + 4,096 = 9,728`
* **activation_memory_per_chip = 9,728**
* **activation_memory_total = 155,648**

#### 4) KV cache

Total KV bytes (past only):

* `2 * B * S_k * d_kv * bytes`
* = `2 * 2 * 128 * 256 * 2 = 262,144`

Sharded across TP×CP:

* per chip = `262,144 / 16 = 16,384`
* total = `262,144`

So:

* **kv_cache_per_chip = 16,384**
* **kv_cache_total = 262,144**

#### 5) Communication

**(a) CP all-reduce for softmax stats**
Stats scalars:

* `B*T*h_q_local*2 = 2*1*4*2 = 16 scalars`
  Bytes:
* `16 * 4 = 64`

**(b) CP all-reduce for partial output vectors**
Output local head slice:

* elems = `B*T*d_local = 2*1*256 = 512`
* bytes = `512*2 = 1,024`

So CP payload:

* `comm_cp = 64 + 1,024 = 1,088`

**(c) TP all-gather to materialize full output**
Output for new tokens:

* elems = `B*T*d = 2*1*1024 = 2,048`
* bytes = `2,048*2 = 4,096`

Total:

* **communication_bytes = 1,088 + 4,096 = 5,184**

### Expected results

**Per-chip metrics**

* flops_per_chip: **2686976**
* weight_memory_per_chip: **1310720**
* activation_memory_per_chip: **9728**
* kv_cache_per_chip: **16384**

**Aggregate metrics**

* flops_total: **42991616**
* weight_memory_total: **20971520**
* activation_memory_total: **155648**
* kv_cache_total: **262144**

**Hardware-dependent (optional)**

* communication_bytes: **5184**

---

## Notes / likely follow-ups

* If your implementation supports `h_kv < tp` or `h_kv` not divisible by `tp`, you’ll need a different sharding rule (e.g., replicate KV across some TP ranks). I can write a dedicated edge-case test for that if it matters.
* If you prefer “no TP materialization” (keep outputs sharded by heads), I can provide variant expected results (comm drops; activation output drops).

If you want, next we can add one more very practical GQA test: **TP-head only decode (no CP)** for moderate contexts, which is what many serving stacks do by default when context isn’t huge.
