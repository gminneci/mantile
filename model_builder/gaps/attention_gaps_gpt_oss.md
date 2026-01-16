# Layer Gap Report: GPT-OSS Attention (SWA, RoPE YaRN, Biases)

## 1. Model Configuration Summary

Based on the GPT-OSS-120B architecture:

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_size (`d`) | 2880 | Non-standard (not power of 2) |
| num_query_heads (`h_q`) | 32 | Standard MHA head count |
| num_kv_heads (`h_kv`) | 4 | GQA with 8:1 ratio |
| head_dim (`dh`) | 128 | Standard |
| sliding_window | 128 | Limits KV cache and attention span |
| rope_type | yarn | YaRN scaling for extended context |
| rope_scaling_factor | 32.0 | 32x context extension |
| attention_bias | true | All projections have bias terms |

**Derived Values:**
- Q projection output: `h_q * dh = 32 * 128 = 4096`
- KV projection output: `h_kv * dh = 4 * 128 = 512`
- GQA group size: `h_q / h_kv = 8`

## 2. Example Tensors (from gpt-oss-120b dry run)

### Projections with Biases
```
  model.layers.0.self_attn.q_proj.weight: (4096, 2880) (torch.float32)
  model.layers.0.self_attn.q_proj.bias: (4096,) (torch.float32)
  model.layers.0.self_attn.k_proj.weight: (512, 2880) (torch.float32)
  model.layers.0.self_attn.k_proj.bias: (512,) (torch.float32)
  model.layers.0.self_attn.v_proj.weight: (512, 2880) (torch.float32)
  model.layers.0.self_attn.v_proj.bias: (512,) (torch.float32)
  model.layers.0.self_attn.o_proj.weight: (2880, 4096) (torch.float32)
  model.layers.0.self_attn.o_proj.bias: (2880,) (torch.float32)
```

### Non-Standard Parameters
```
  model.layers.0.self_attn.sinks: (64,) (torch.float32)
```

The `sinks` tensor suggests **attention sink** functionality - keeping initial tokens always in the attention window for streaming/infinite context scenarios.

## 3. Identified Gaps

### Gap 1: Sliding Window Attention (SWA)

**Config**: `sliding_window: 128`

**Description**: Each query token only attends to the most recent 128 key tokens (plus any sink tokens), rather than full causal attention over the entire sequence.

**Impact on Mantile Calculations**:

| Metric | Standard Attention | With SWA (window=W) |
|--------|-------------------|---------------------|
| KV Cache Size | `2 * B * S * d_kv` | `2 * B * min(S, W) * d_kv` |
| Prefill Score FLOPs | `2 * B * h_q * S * S * dh` | `2 * B * h_q * S * min(S, W) * dh` |
| Decode Score FLOPs | `2 * B * h_q * 1 * S_past * dh` | `2 * B * h_q * 1 * min(S_past, W) * dh` |

**Implementation Requirements**:
1. Circular buffer for KV cache management
2. Windowed attention mask during prefill
3. KV cache eviction policy (FIFO with optional sink preservation)

**Reference**: Mistral-7B uses SWA with window=4096.

### Gap 2: RoPE YaRN Scaling

**Config**:
```yaml
rope_scaling:
  rope_type: yarn
  factor: 32.0
  original_max_position_embeddings: 2048
  attention_factor: null  # computed dynamically
  beta_fast: 32
  beta_slow: 1
```

**Description**: YaRN (Yet Another RoPE Extension) modifies RoPE frequencies to enable context extension beyond training length. It uses:
- High-frequency extrapolation (no modification)
- Low-frequency interpolation (standard NTK-aware)
- Smooth transition between the two regimes

**Impact on Mantile Calculations**:

| Metric | Impact |
|--------|--------|
| FLOPs | Negligible (RoPE application is O(S * d), dominated by attention O(S² * d)) |
| Memory | None (frequencies can be precomputed or computed on-the-fly) |
| Compute Time | Minor increase for frequency computation |

**Implementation Requirements**:
1. Multi-frequency scaling logic based on `beta_fast`, `beta_slow`
2. Dynamic `attention_factor` computation (typically `0.1 * ln(factor) + 1`)
3. Precomputed frequency tables for efficiency

**Reference**: [YaRN Paper](https://arxiv.org/abs/2309.00071)

### Gap 3: Attention Biases

**Config**: All Q/K/V/O projections include bias terms.

**Impact on Mantile Calculations**:

| Metric | Without Bias | With Bias | Delta |
|--------|-------------|-----------|-------|
| Weight Memory | `d*d_q + d*d_kv + d*d_kv + d_q*d` | Same + `d_q + d_kv + d_kv + d` | +`d_q + 2*d_kv + d` elements |
| FLOPs (per projection) | `2*M*d_in*d_out` | Same + `M*d_out` (bias add) | +`M*d_out` per projection |

For GPT-OSS with `d=2880`, `d_q=4096`, `d_kv=512`:
- Extra weight memory: `4096 + 512 + 512 + 2880 = 8000` elements = 16,000 bytes (BF16)
- Extra FLOPs per token: `4096 + 512 + 512 + 2880 = 8000` (negligible vs projection FLOPs)

**Implementation Requirements**:
1. Add `has_bias` parameter to attention layer
2. Include bias memory in weight calculations
3. Include bias FLOPs in compute calculations (typically negligible)

### Gap 4: Attention Projection Dimension Mismatch

**Observation**: `hidden_size (2880) != num_heads * head_dim (32 * 128 = 4096)`

**Description**: The model uses a projection from hidden dimension to a larger attention dimension:
- Input: `[B, S, 2880]`
- Q projection: `[2880, 4096]` → Q: `[B, S, 4096]` → `[B, S, 32, 128]`
- O projection: `[4096, 2880]` → Output: `[B, S, 2880]`

This is different from standard transformers where `d = h * dh`.

**Impact on Mantile Calculations**:

| Projection | Standard Shape | GPT-OSS Shape |
|------------|---------------|---------------|
| W_q | `[d, d]` | `[d, h_q * dh]` = `[2880, 4096]` |
| W_k | `[d, d_kv]` | `[d, h_kv * dh]` = `[2880, 512]` |
| W_v | `[d, d_kv]` | `[d, h_kv * dh]` = `[2880, 512]` |
| W_o | `[d, d]` | `[h_q * dh, d]` = `[4096, 2880]` |

**Implementation Requirements**:
1. Separate `hidden_size` from `attention_hidden_size` (= `h_q * dh`)
2. Update projection weight calculations to use correct dimensions
3. Ensure output projection maps back to model hidden size

### Gap 5: Attention Sinks

**Config**: `sinks` tensor of shape `(64,)`

**Description**: Attention sinks preserve the first N tokens in the KV cache regardless of sliding window eviction. This prevents degradation in streaming/infinite context scenarios.

**Impact on Mantile Calculations**:

| Metric | Impact |
|--------|--------|
| KV Cache Size | `2 * B * (num_sinks + window) * d_kv` instead of `2 * B * window * d_kv` |
| Score FLOPs | Attend to `num_sinks + min(S, window)` positions |

**Implementation Requirements**:
1. `num_sinks` parameter (here: 64)
2. Modified KV cache eviction: preserve first `num_sinks` positions
3. Attention mask includes both sinks and sliding window

## 4. Documentation References

### PyTorch
- `torch.nn.functional.scaled_dot_product_attention` with custom masks
- `torch.nn.MultiheadAttention` (does not natively support SWA)

### vLLM
- `vllm/attention/backends/flash_attn.py` - Flash attention with sliding window
- `vllm/model_executor/layers/rotary_embedding.py` - YaRN implementation

### SGLang
- Sliding window support in FlashInfer backend
- YaRN RoPE scaling in model implementations

### FlashAttention
- `flash_attn_func` with `window_size` parameter for SWA
- Efficient masking for windowed attention

## 5. Parallelism Considerations

### Tensor Parallel (TP)

With TP, the attention projections are sharded:

| Projection | Sharding | TP Communication |
|------------|----------|------------------|
| Q | Column parallel (`d` → `d_q/tp`) | None |
| K | Column parallel (`d` → `d_kv/tp`) | None |
| V | Column parallel (`d` → `d_kv/tp`) | None |
| O | Row parallel (`d_q/tp` → `d`) | All-reduce on output |

**Note**: With GQA (h_kv < h_q), KV projections may not divide evenly by large TP. GPT-OSS has h_kv=4, so maximum efficient TP=4 for KV heads.

### Context Parallel (CP)

With SWA, CP has modified behavior:
- Each CP rank holds a sequence chunk
- KV cache exchange only needed at chunk boundaries (within window)
- Reduced communication compared to full attention

---

# Test Suite: GPT-OSS Attention

## Conventions

* `bytes_per_elem = 2` (BF16)
* `B = batch_size`, `S = seq_len`, `M = B*S`
* `d = hidden_size = 2880`
* `d_q = h_q * dh = 32 * 128 = 4096` (attention hidden size)
* `d_kv = h_kv * dh = 4 * 128 = 512`
* `W = sliding_window = 128`
* FLOPs for GEMM `(M×K)@(K×N)` = `2*M*K*N`
* Bias add FLOPs included but typically negligible

---

# Test SWA-1 — Sliding Window Attention Prefill, Single Chip, seq_len < window

When sequence length is less than sliding window, behavior matches standard attention.

### Test case parameters

* hidden_size `d` = **2880**
* num_query_heads `h_q` = **32**
* num_kv_heads `h_kv` = **4**
* head_dim `dh` = **128**
* sliding_window `W` = **128**
* batch_size `B` = **2**
* seq_len `S` = **64** (less than window)
* bytes_per_elem = **2**
* has_bias = **true**
* num_chips = **1**
* tensor_parallel `tp` = **1**

Derived:

* `M = B*S = 128`
* `d_q = h_q * dh = 4096`
* `d_kv = h_kv * dh = 512`
* Effective attention span = `min(S, W) = 64` (full sequence)

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Q projection**
`X[M, d] @ Wq[d, d_q]` + bias:

* GEMM FLOPs = `2 * M * d * d_q = 2 * 128 * 2880 * 4096 = 3,019,898,880`
* Bias FLOPs = `M * d_q = 128 * 4096 = 524,288`
* Q total = `3,020,423,168`

**(b) K projection**
`X[M, d] @ Wk[d, d_kv]` + bias:

* GEMM FLOPs = `2 * M * d * d_kv = 2 * 128 * 2880 * 512 = 377,487,360`
* Bias FLOPs = `M * d_kv = 65,536`
* K total = `377,552,896`

**(c) V projection**
Same as K:

* V total = `377,552,896`

**(d) Attention scores (QK^T)**
Per batch & head: `(S × dh) @ (dh × S)` → `2 * S * S * dh`

* FLOPs = `2 * B * h_q * S * S * dh`
* = `2 * 2 * 32 * 64 * 64 * 128`
* = `67,108,864`

**(e) Apply attention to V**
Same shape as scores:

* = `67,108,864`

**(f) Output projection**
`O[M, d_q] @ Wo[d_q, d]` + bias:

* GEMM FLOPs = `2 * M * d_q * d = 2 * 128 * 4096 * 2880 = 3,019,898,880`
* Bias FLOPs = `M * d = 128 * 2880 = 368,640`
* O total = `3,020,267,520`

**(g) Total FLOPs**

* Q + K + V + Scores + ApplyV + O
* = `3,020,423,168 + 377,552,896 + 377,552,896 + 67,108,864 + 67,108,864 + 3,020,267,520`
* **flops_total = 6,930,014,208**
* **flops_per_chip = 6,930,014,208**

#### 2) Weight memory

**(a) Projection weights**

* Wq: `d * d_q = 2880 * 4096 = 11,796,480` elements
* Wk: `d * d_kv = 2880 * 512 = 1,474,560` elements
* Wv: `d * d_kv = 1,474,560` elements
* Wo: `d_q * d = 4096 * 2880 = 11,796,480` elements
* Total projection = `26,542,080` elements

**(b) Bias weights**

* bq: `d_q = 4096` elements
* bk: `d_kv = 512` elements
* bv: `512` elements
* bo: `d = 2880` elements
* Total bias = `8,000` elements

**(c) Total weights**

* Total elements = `26,542,080 + 8,000 = 26,550,080`
* Bytes = `26,550,080 * 2 = 53,100,160`
* **weight_memory_per_chip = 53,100,160**
* **weight_memory_total = 53,100,160**

#### 3) Activation memory (resident, minimal)

* Input X: `M * d = 128 * 2880 = 368,640` elems → `737,280` bytes
* Q: `M * d_q = 128 * 4096 = 524,288` elems → `1,048,576` bytes
* K: `M * d_kv = 65,536` elems → `131,072` bytes
* V: `65,536` elems → `131,072` bytes
* Output Y: `M * d = 368,640` elems → `737,280` bytes

Total:

* **activation_memory_per_chip = 737,280 + 1,048,576 + 131,072 + 131,072 + 737,280 = 2,785,280**
* **activation_memory_total = 2,785,280**

#### 4) KV cache

Prefill writes KV cache. With S < W, full sequence is cached:

* K cache: `B * h_kv * S * dh = 2 * 4 * 64 * 128 = 65,536` elems
* V cache: same = `65,536` elems
* Total = `131,072` elems → `262,144` bytes

* **kv_cache_per_chip = 262,144**
* **kv_cache_total = 262,144**

#### 5) Communication

Single chip:

* **communication_bytes = 0**

### Expected results

**Per-chip metrics**

* flops_per_chip: **6930014208**
* weight_memory_per_chip: **53100160**
* activation_memory_per_chip: **2785280**
* kv_cache_per_chip: **262144**

**Aggregate metrics**

* flops_total: **6930014208**
* weight_memory_total: **53100160**
* activation_memory_total: **2785280**
* kv_cache_total: **262144**

**Hardware-dependent (optional)**

* communication_bytes: **0**

---

# Test SWA-2 — Sliding Window Attention Prefill, Single Chip, seq_len > window

When sequence length exceeds sliding window, attention is limited to window size.

### Test case parameters

* hidden_size `d` = **2880**
* num_query_heads `h_q` = **32**
* num_kv_heads `h_kv` = **4**
* head_dim `dh` = **128**
* sliding_window `W` = **128**
* batch_size `B` = **2**
* seq_len `S` = **512** (4x window)
* bytes_per_elem = **2**
* has_bias = **true**
* num_chips = **1**
* tensor_parallel `tp` = **1**

Derived:

* `M = B*S = 1024`
* Effective attention span = `min(S, W) = 128` (limited by window)

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Q projection**

* GEMM = `2 * M * d * d_q = 2 * 1024 * 2880 * 4096 = 24,159,191,040`
* Bias = `M * d_q = 4,194,304`
* Q total = `24,163,385,344`

**(b) K projection**

* GEMM = `2 * M * d * d_kv = 2 * 1024 * 2880 * 512 = 3,019,898,880`
* Bias = `M * d_kv = 524,288`
* K total = `3,020,423,168`

**(c) V projection**

* V total = `3,020,423,168`

**(d) Attention scores with sliding window**
Each query attends to at most W keys. For prefill with causal mask + SWA:
- Average attention span ≈ `W/2` for early tokens, `W` for later tokens
- Simplified: use average span of `W` for later tokens (conservative)

Actual formula for SWA prefill with S > W:
* FLOPs ≈ `2 * B * h_q * S * W * dh` (each query attends to ~W keys)
* = `2 * 2 * 32 * 512 * 128 * 128`
* = `1,073,741,824`

**(e) Apply attention to V**

* = `1,073,741,824`

**(f) Output projection**

* GEMM = `2 * M * d_q * d = 24,159,191,040`
* Bias = `M * d = 2,949,120`
* O total = `24,162,140,160`

**(g) Total FLOPs**

* = `24,163,385,344 + 3,020,423,168 + 3,020,423,168 + 1,073,741,824 + 1,073,741,824 + 24,162,140,160`
* **flops_total = 56,513,855,488**
* **flops_per_chip = 56,513,855,488**

Note: Without SWA (full attention), score FLOPs would be:
* `2 * 2 * 32 * 512 * 512 * 128 = 4,294,967,296` (4x higher!)

#### 2) Weight memory

Same as SWA-1 (weights don't depend on sequence length):

* **weight_memory_per_chip = 53,100,160**
* **weight_memory_total = 53,100,160**

#### 3) Activation memory

* Input X: `M * d = 1024 * 2880` → `5,898,240` bytes
* Q: `M * d_q` → `8,388,608` bytes
* K: `M * d_kv` → `1,048,576` bytes
* V: → `1,048,576` bytes
* Output Y: → `5,898,240` bytes

Total:

* **activation_memory_per_chip = 22,282,240**
* **activation_memory_total = 22,282,240**

#### 4) KV cache

With SWA, KV cache is limited to window size (or uses circular buffer):

* K cache: `B * h_kv * W * dh = 2 * 4 * 128 * 128 = 131,072` elems
* V cache: same = `131,072` elems
* Total = `262,144` elems → `524,288` bytes

* **kv_cache_per_chip = 524,288** (capped at window)
* **kv_cache_total = 524,288**

Note: Without SWA, KV cache would be `2 * 4 * 512 * 128 * 2 = 2,097,152` bytes (4x larger).

#### 5) Communication

* **communication_bytes = 0**

### Expected results

**Per-chip metrics**

* flops_per_chip: **56513855488**
* weight_memory_per_chip: **53100160**
* activation_memory_per_chip: **22282240**
* kv_cache_per_chip: **524288**

**Aggregate metrics**

* flops_total: **56513855488**
* weight_memory_total: **53100160**
* activation_memory_total: **22282240**
* kv_cache_total: **524288**

**Hardware-dependent (optional)**

* communication_bytes: **0**

---

# Test SWA-3 — Sliding Window Attention Decode, Single Chip

Decode step with sliding window attention.

### Test case parameters

* hidden_size `d` = **2880**
* num_query_heads `h_q` = **32**
* num_kv_heads `h_kv` = **4**
* head_dim `dh` = **128**
* sliding_window `W` = **128**
* batch_size `B` = **2**
* past_seq_len `S_past` = **256** (exceeds window)
* new_tokens `T` = **1**
* bytes_per_elem = **2**
* has_bias = **true**
* num_chips = **1**

Derived:

* `M_q = B * T = 2`
* Effective KV length = `min(S_past, W) = 128`

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Q projection (new token)**

* GEMM = `2 * M_q * d * d_q = 2 * 2 * 2880 * 4096 = 47,185,920`
* Bias = `M_q * d_q = 8,192`
* Q total = `47,194,112`

**(b) K projection (new token)**

* GEMM = `2 * M_q * d * d_kv = 2 * 2 * 2880 * 512 = 5,898,240`
* Bias = `M_q * d_kv = 1,024`
* K total = `5,899,264`

**(c) V projection**

* V total = `5,899,264`

**(d) Attention scores**
Query attends to W cached keys (window limit):

* FLOPs = `2 * B * h_q * T * W * dh`
* = `2 * 2 * 32 * 1 * 128 * 128`
* = `2,097,152`

**(e) Apply attention to V**

* = `2,097,152`

**(f) Output projection**

* GEMM = `2 * M_q * d_q * d = 47,185,920`
* Bias = `M_q * d = 5,760`
* O total = `47,191,680`

**(g) Total FLOPs**

* = `47,194,112 + 5,899,264 + 5,899,264 + 2,097,152 + 2,097,152 + 47,191,680`
* **flops_total = 110,378,624**
* **flops_per_chip = 110,378,624**

#### 2) Weight memory

Same as SWA-1:

* **weight_memory_per_chip = 53,100,160**
* **weight_memory_total = 53,100,160**

#### 3) Activation memory

Decode step minimal buffers:

* X_new: `M_q * d = 2 * 2880` → `11,520` bytes
* Q: `M_q * d_q = 8,192` → `16,384` bytes
* K_new: `M_q * d_kv = 1,024` → `2,048` bytes
* V_new: → `2,048` bytes
* Y: `M_q * d` → `11,520` bytes

Total:

* **activation_memory_per_chip = 43,520**
* **activation_memory_total = 43,520**

#### 4) KV cache

With SWA, cache is limited to window (circular buffer):

* KV cache = `2 * B * h_kv * W * dh * bytes`
* = `2 * 2 * 4 * 128 * 128 * 2 = 524,288`

* **kv_cache_per_chip = 524,288**
* **kv_cache_total = 524,288**

#### 5) Communication

* **communication_bytes = 0**

### Expected results

**Per-chip metrics**

* flops_per_chip: **110378624**
* weight_memory_per_chip: **53100160**
* activation_memory_per_chip: **43520**
* kv_cache_per_chip: **524288**

**Aggregate metrics**

* flops_total: **110378624**
* weight_memory_total: **53100160**
* activation_memory_total: **43520**
* kv_cache_total: **524288**

**Hardware-dependent (optional)**

* communication_bytes: **0**

---

# Test SWA-4 — Sliding Window Attention with TP=4

Tensor parallel sliding window attention.

### Test case parameters

* hidden_size `d` = **2880**
* num_query_heads `h_q` = **32**
* num_kv_heads `h_kv` = **4**
* head_dim `dh` = **128**
* sliding_window `W` = **128**
* batch_size `B` = **2**
* seq_len `S` = **256**
* bytes_per_elem = **2**
* has_bias = **true**
* tensor_parallel `tp` = **4**
* num_chips = **4**

Derived:

* `M = B*S = 512`
* `h_q_local = h_q / tp = 8`
* `h_kv_local = h_kv / tp = 1`
* `d_q_local = h_q_local * dh = 1024`
* `d_kv_local = h_kv_local * dh = 128`
* Effective attention span = `min(S, W) = 128`

### Expected calculations (step-by-step)

#### 1) FLOPs

Total FLOPs (same as single chip with S=256, W=128):

**(a) Projections (full sequence)**

* Q: `2 * M * d * d_q + M * d_q = 2 * 512 * 2880 * 4096 + 2,097,152 = 12,081,692,672`
* K: `2 * M * d * d_kv + M * d_kv = 1,510,211,584`
* V: `1,510,211,584`
* O: `2 * M * d_q * d + M * d = 12,080,218,112`

**(b) Attention with SWA**

* Scores: `2 * B * h_q * S * W * dh = 2 * 2 * 32 * 256 * 128 * 128 = 536,870,912`
* ApplyV: `536,870,912`

**(c) Total**

* flops_total = `12,081,692,672 + 1,510,211,584 + 1,510,211,584 + 536,870,912 + 536,870,912 + 12,080,218,112`
* = `28,256,075,776`

Per chip with TP=4:

* **flops_per_chip = 28,256,075,776 / 4 = 7,064,018,944**
* **flops_total = 28,256,075,776**

#### 2) Weight memory

With TP, weights are sharded by heads:

* Wq: `d * d_q_local = 2880 * 1024 = 2,949,120` elems per chip
* Wk: `d * d_kv_local = 2880 * 128 = 368,640` elems per chip
* Wv: `368,640` elems per chip
* Wo: `d_q_local * d = 1024 * 2880 = 2,949,120` elems per chip
* Biases: `d_q_local + d_kv_local + d_kv_local + d = 1024 + 128 + 128 + 2880 = 4,160` elems per chip

Total per chip = `2,949,120 + 368,640 + 368,640 + 2,949,120 + 4,160 = 6,639,680` elements
Bytes = `6,639,680 * 2 = 13,279,360`

* **weight_memory_per_chip = 13,279,360**
* **weight_memory_total = 53,117,440** (unique weights, not 4x)

Actually, biases for output projection (bo) are typically not sharded (full d). Let me recalculate:
- Per chip sharded: `d_q_local + d_kv_local + d_kv_local = 1280`
- Output bias replicated: `d = 2880`
- Total bias per chip: `1280 + 2880 = 4160` (as calculated)

But weight_memory_total should be unique weights = 53,100,160 (from SWA-1).

* **weight_memory_per_chip = 13,279,360**
* **weight_memory_total = 53,100,160** (unique)

#### 3) Activation memory

Per chip with TP:

* X (replicated): `M * d` → `2,949,120` bytes
* Q_local: `M * d_q_local = 512 * 1024` → `1,048,576` bytes
* K_local: `M * d_kv_local = 512 * 128` → `131,072` bytes
* V_local: → `131,072` bytes
* Y (full, after all-reduce): `M * d` → `2,949,120` bytes

Total:

* **activation_memory_per_chip = 7,208,960**
* **activation_memory_total = 4 * 7,208,960 = 28,835,840**

#### 4) KV cache

KV cache sharded by KV heads:

* Per chip: `2 * B * h_kv_local * W * dh * bytes`
* = `2 * 2 * 1 * 128 * 128 * 2 = 131,072`

* **kv_cache_per_chip = 131,072**
* **kv_cache_total = 524,288**

#### 5) Communication

TP all-reduce on output:

* payload = `M * d * bytes = 512 * 2880 * 2 = 2,949,120`

* **communication_bytes = 2,949,120**

### Expected results

**Per-chip metrics**

* flops_per_chip: **7064018944**
* weight_memory_per_chip: **13279360**
* activation_memory_per_chip: **7208960**
* kv_cache_per_chip: **131072**

**Aggregate metrics**

* flops_total: **28256075776**
* weight_memory_total: **53100160**
* activation_memory_total: **28835840**
* kv_cache_total: **524288**

**Hardware-dependent (optional)**

* communication_bytes: **2949120**

---

# Test SWA-5 — Sliding Window with Attention Sinks

Sliding window attention with sink tokens preserved.

### Test case parameters

* hidden_size `d` = **2880**
* num_query_heads `h_q` = **32**
* num_kv_heads `h_kv` = **4**
* head_dim `dh` = **128**
* sliding_window `W` = **128**
* num_sinks = **64**
* batch_size `B` = **2**
* seq_len `S` = **512**
* bytes_per_elem = **2**
* has_bias = **true**
* num_chips = **1**

Derived:

* `M = B*S = 1024`
* Effective attention span = `num_sinks + W = 64 + 128 = 192`
* Total KV positions = `num_sinks + W = 192`

### Expected calculations (step-by-step)

#### 1) FLOPs

Projections same as SWA-2 (depend on M, not attention span).

**(a-c) Q, K, V projections**

* Q total = `24,163,385,344`
* K total = `3,020,423,168`
* V total = `3,020,423,168`

**(d) Attention scores with sinks + window**
Each query attends to sinks (64) + window (up to 128):

* FLOPs = `2 * B * h_q * S * (num_sinks + W) * dh`
* = `2 * 2 * 32 * 512 * 192 * 128`
* = `1,610,612,736`

**(e) Apply attention to V**

* = `1,610,612,736`

**(f) Output projection**

* O total = `24,162,140,160`

**(g) Total FLOPs**

* = `24,163,385,344 + 3,020,423,168 + 3,020,423,168 + 1,610,612,736 + 1,610,612,736 + 24,162,140,160`
* **flops_total = 57,587,597,312**
* **flops_per_chip = 57,587,597,312**

#### 2) Weight memory

Same as before:

* **weight_memory_per_chip = 53,100,160**
* **weight_memory_total = 53,100,160**

#### 3) Activation memory

Same as SWA-2:

* **activation_memory_per_chip = 22,282,240**
* **activation_memory_total = 22,282,240**

#### 4) KV cache

With sinks, KV cache holds sinks + window:

* KV positions = `num_sinks + W = 192`
* K cache: `B * h_kv * 192 * dh = 2 * 4 * 192 * 128 = 196,608` elems
* V cache: same = `196,608` elems
* Total = `393,216` elems → `786,432` bytes

* **kv_cache_per_chip = 786,432**
* **kv_cache_total = 786,432**

#### 5) Communication

* **communication_bytes = 0**

### Expected results

**Per-chip metrics**

* flops_per_chip: **57587597312**
* weight_memory_per_chip: **53100160**
* activation_memory_per_chip: **22282240**
* kv_cache_per_chip: **786432**

**Aggregate metrics**

* flops_total: **57587597312**
* weight_memory_total: **53100160**
* activation_memory_total: **22282240**
* kv_cache_total: **786432**

**Hardware-dependent (optional)**

* communication_bytes: **0**

---

## Test Summary

| Test | Config | Seq | Window | Sinks | TP | FLOPs/chip | KV Cache/chip |
|------|--------|-----|--------|-------|----|-----------:|-------------:|
| SWA-1 | S < W | 64 | 128 | 0 | 1 | 6.93B | 256KB |
| SWA-2 | S > W | 512 | 128 | 0 | 1 | 56.5B | 512KB |
| SWA-3 | Decode | 256+ | 128 | 0 | 1 | 110M | 512KB |
| SWA-4 | TP=4 | 256 | 128 | 0 | 4 | 7.06B | 128KB |
| SWA-5 | +Sinks | 512 | 128 | 64 | 1 | 57.6B | 768KB |

## Implementation Checklist

- [ ] Sliding window mask in prefill attention
- [ ] Circular buffer KV cache management
- [ ] Sink token preservation in KV eviction
- [ ] YaRN RoPE frequency scaling
- [ ] Attention bias support in FLOPs/memory calculations
- [ ] Dimension mismatch handling (hidden_size ≠ num_heads * head_dim)
