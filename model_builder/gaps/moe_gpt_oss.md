# Layer Gap Report: GPT-OSS MoE (128 Experts)

## 1. Example Tensors (from gpt-oss-120b dry run)

```
  model.layers.0.mlp.router.weight: (128, 2880) (torch.float32)
  model.layers.0.mlp.router.bias: (128,) (torch.float32)
  model.layers.0.mlp.experts.gate_up_proj: (128, 2880, 5760) (torch.float32)
  model.layers.0.mlp.experts.gate_up_proj_bias: (128, 5760) (torch.float32)
  model.layers.0.mlp.experts.down_proj: (128, 2880, 2880) (torch.float32)
  model.layers.0.mlp.experts.down_proj_bias: (128, 2880) (torch.float32)
```

## 2. Documentation References

### PyTorch
- `torch.mm`, `torch.nn.functional.softmax`, `torch.topk`
- `torch.bmm` (Batch Matrix Multiply) for parallel expert processing.

### vLLM
- `vllm/model_executor/layers/fused_moe/`
- vLLM uses custom kernels for MoE to avoid the overhead of routing tokens individually.

### SGLang
- Implements MoE with optimized kernels for high-throughput inference.

## 3. Parallelism Strategies

| Strategy | Description | Communication Pattern |
|----------|-------------|----------------------|
| Expert Parallelism (EP) | Shards the 128 experts across devices. | All-to-All |
| Tensor Parallelism (TP) | Shards individual expert weights (gate_up/down). | All-Reduce |
| Hybrid EP+TP | Combines both for large clusters. | Both |

## 4. Test Suite Specification

### Test MOE-OSS-1: Router + Experts (Top-4)

* hidden_size `d` = **2880**
* num_experts `E` = **128**
* top_k = **4**
* intermediate_size `di` = **2880** (times 2 for gated)
* batch_size `B` = **1**
* seq_len `S` = **2048**

**Expected FLOPs (Simplified)**:
1. Router: `2 * M * d * E`
2. Experts: `top_k * (3 * 2 * M * d * di)` (for gated SwiGLU)

**Expected Memory**:
1. Router: `d * E * 2` (FP16)
2. Experts: `E * 3 * d * di * 2` (FP16)

## 5. Implementation Notes

- The `experts.gate_up_proj` is fused (gate and up projections in one tensor).
- Biases are present in all projections, which Mantile's current layers (Attention/MLP) may not expect.
- The 128-expert setup is significantly larger than typical MoE models (like Mixtral 8x7B), requiring efficient memory management.
