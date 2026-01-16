# Layer Gap Report: GPT-OSS Attention (SWA, RoPE YaRN, Biases)

## 1. Example Tensors (from gpt-oss-120b dry run)

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

## 2. Identified Gaps

### Sliding Window Attention (SWA)
- **Config**: `sliding_window: 128`
- **Impact**: Limits the KV cache size to 128 tokens. Mantile's current attention layers assume full causal attention.
- **Implementation Need**: Circular buffer management for KV cache and windowed mask in prefill.

### RoPE YaRN Scaling
- **Config**: `rope_scaling: {'rope_type': 'yarn', 'factor': 32.0, ...}`
- **Impact**: Requires multi-frequency scaling logic. Mantile likely assumes standard RoPE.
- **Reference**: [YaRN: Yet Another RoPE Extension](https://arxiv.org/abs/2309.00071).

### Attention Biases
- **Impact**: All projections have bias tensors. Mantile's `GroupedQueryAttentionLayer` needs to include bias add operations in FLOP and memory calculations.

### Attention Projection Mismatch
- **Impact**: Hidden size (2880) != num_heads * head_dim (32 * 128 = 4096).
- **Implementation Need**: The `o_proj` input dimension must match the total attention width, not the model hidden size.

## 3. Test Suite Specification (Mock Step)

* hidden_size `d` = 2880
* num_heads = 32
* head_dim = 128
* sliding_window = 128

**Memory Calculation**:
- KV cache (per chip, CP=1, TP=1): `2 * batch * num_kv_heads * sliding_window * head_dim * bytes`
- Note: `sliding_window` replaces `seq_len` for the cache size limit.
