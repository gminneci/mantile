# Layer Gap Reports

This directory contains gap reports for layer types not yet implemented in Mantile.

## Purpose

When the model builder encounters an unsupported layer type (e.g., MoE routers, 
sliding window attention, cross-attention), it generates a gap report containing:

1. **Example Tensors** - Actual tensor names/shapes from dry-run inspection
2. **Documentation References** - Links to PyTorch, vLLM, SGLang implementations
3. **Parallelism Strategies** - How the layer is typically distributed
4. **Test Suite Specification** - Minimal tests needed for implementation

## Usage

Gap reports are generated using `model_builder/utils.py`:

```python
from model_builder.utils import generate_gap_report_template, inspect_model_structure

tensors = inspect_model_structure("mistralai/Mixtral-8x7B-v0.1")
report = generate_gap_report_template(
    layer_name="Mixture of Experts Router",
    tensors=tensors,
    tensor_pattern=r"block_sparse_moe"
)
```

## Reports

| Layer Type | Model | Severity | Status | File |
|------------|-------|----------|--------|------|
| *(no open gaps)* | | | | |

When a gap report is created, **add it to this table** and link from the model config's
`gap_report` field.

## Gap Severity Levels

- **HIGH**: Core layer type unsupported (MoE, cross-attention, new architecture)
- **MEDIUM**: Variant of supported layer (sliding window, different normalization)
- **LOW**: Minor feature difference (biases, scaling factors, tied weights)
