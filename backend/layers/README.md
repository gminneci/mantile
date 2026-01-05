# Layers Package

Modular implementation of LLM layer types with hardware-aware performance modeling.

## Structure

```
backend/layers/
├── __init__.py       # Package exports
├── base.py           # Layer, Phase, DataType, LayerMetrics
├── mlp.py            # MLPLayer, GatedMLPLayer
├── attention.py      # AttentionLayer, GroupedQueryAttentionLayer
├── norm.py           # NormLayer
└── README.md         # This file
```

## Quick Start

```python
from backend.layers import MLPLayer, Phase, DataType

# Create a layer with parallelism config
layer = MLPLayer(
    name="layer_0_mlp",
    layer_idx=0,
    hidden_size=4096,
    intermediate_size=16384,
    parallelism={"tensor_parallel": 2, "sequence_parallel": 4}
)

# Compute metrics
metrics = layer.compute_metrics(
    batch_size=1,
    seq_len=2048,
    phase="prefill",
    dtype="bf16",
    hardware=h100_config  # optional
)

print(f"FLOPs per chip: {metrics.flops_per_chip}")
print(f"Activation memory: {metrics.activation_memory_per_chip} bytes")
```

## Layer Types

### MLP Layers (`mlp.py`)
- **MLPLayer**: Standard 2-projection or 3-projection feedforward
- **GatedMLPLayer**: Convenience wrapper for 3-projection (SwiGLU-style)

Supported parallelism: TP, PP, SP

### Attention Layers (`attention.py`)
- **AttentionLayer**: Vanilla multi-head attention (MHA)
- **GroupedQueryAttentionLayer**: Grouped query attention (GQA) with shared KV heads

Supported parallelism: TP

### Normalization Layers (`norm.py`)
- **NormLayer**: LayerNorm or RMSNorm

Supported parallelism: TP, PP, EP (replicated, not sharded)

## Parallelism Strategies

- **TP (Tensor Parallel)**: Split weights/computation across devices
- **PP (Pipeline Parallel)**: Split layers across pipeline stages
- **SP (Sequence Parallel)**: Partition sequence dimension to reduce activation memory
- **EP (Expert Parallel)**: For MoE layers (future)

## Testing

Each layer type should have corresponding tests:

```
tests/
├── test_mlp.py
├── test_attention.py
└── test_norm.py
```

## Backward Compatibility

The old `backend/layers.py` file is maintained as a compatibility wrapper that re-exports from the new package structure.

## Adding New Layer Types

1. Create new file in `backend/layers/` (e.g., `moe.py`)
2. Inherit from `Layer` base class
3. Implement required methods
4. Add exports to `__init__.py`
5. Add tests
