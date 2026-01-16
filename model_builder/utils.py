"""
Utilities for inspecting HuggingFace models and extracting architecture information.

These tools enable model config generation without downloading full model weights.
For gated models, set the HF_TOKEN environment variable or pass token parameter.

Functions:
    get_model_config: Fetch HF config (no weight download)
    inspect_model_structure: Dry-run tensor inspection with empty weights
    compute_tensor_shapes_from_config: Fast tensor shape computation (no model loading)
    save_tensor_inspection: Export tensors to CSV/JSON
    analyze_layer_structure: Infer architecture from tensor names
    format_tensor_sample: Format tensors for documentation
    generate_gap_report_template: Create template for unsupported layers
    
    Validation:
    count_parameters: Calculate total parameters from tensors
    estimate_memory: Estimate memory footprint in bytes
    get_supported_layers: Dynamically list Mantile layer classes
    validate_config: Cross-check computed vs expected values

Authentication:
    For gated models (e.g., Llama, Gemma), either:
    - Set HF_TOKEN environment variable
    - Pass token parameter to functions
    - Run: huggingface-cli login
"""

from typing import Dict, List, Tuple, Any, Optional
import json
from pathlib import Path

__all__ = [
    'get_model_config',
    'inspect_model_structure',
    'compute_tensor_shapes_from_config',
    'save_tensor_inspection',
    'analyze_layer_structure',
    'format_tensor_sample',
    'generate_gap_report_template',
    # Validation utilities
    'count_parameters',
    'estimate_memory',
    'get_supported_layers',
    'validate_config',
    'validate_model_config_file',
]


def _get_hf_token(token: Optional[str] = None) -> Optional[str]:
    """
    Get HuggingFace token from argument or environment.
    
    Args:
        token: Explicit token (takes precedence)
        
    Returns:
        Token string or None if not available
    """
    import os
    return token or os.getenv("HF_TOKEN")


def get_model_config(
    model_id: str,
    trust_remote_code: bool = True,
    output_path: Optional[Path] = None,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch model configuration from HuggingFace Hub.
    
    This retrieves the high-level architecture parameters (hidden_size, num_layers, etc.)
    without downloading model weights.
    
    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.3-70B-Instruct")
        trust_remote_code: Whether to trust remote code from HF (required for some models)
        output_path: Optional path to save config JSON file
        token: HuggingFace token for gated models (or set HF_TOKEN env var)
        
    Returns:
        Dictionary containing model configuration parameters
        
    Example:
        >>> config = get_model_config("meta-llama/Llama-3.3-70B-Instruct")
        >>> config['hidden_size']
        8192
        >>> config['num_hidden_layers']
        80
    """
    from transformers import AutoConfig
    
    hf_token = _get_hf_token(token)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code, token=hf_token)
    config_dict = json.loads(config.to_json_string())
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    return config_dict


def inspect_model_structure(
    model_id: str,
    trust_remote_code: bool = True,
    use_empty_weights: bool = True,
    token: Optional[str] = None
) -> List[Tuple[str, str, Tuple[int, ...], str]]:
    """
    Inspect model structure without downloading weights (dry run).
    
    This extracts all parameters and buffers with their names, shapes, and dtypes.
    Uses accelerate's init_empty_weights() to avoid downloading large model files.
    
    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.3-70B-Instruct")
        trust_remote_code: Whether to trust remote code from HF
        use_empty_weights: If True, use empty weights (no download). If False, download full model.
        token: HuggingFace token for gated models (or set HF_TOKEN env var)
        
    Returns:
        List of tuples: (kind, name, shape, dtype)
        - kind: "param" or "buffer"
        - name: full parameter/buffer name (e.g., "model.layers.0.self_attn.q_proj.weight")
        - shape: tuple of dimensions
        - dtype: string representation of dtype
        
    Example:
        >>> tensors = inspect_model_structure("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> for kind, name, shape, dtype in tensors[:3]:
        ...     print(f"{name}: {shape} ({dtype})")
        model.embed_tokens.weight: (32000, 2048) (torch.float32)
        model.layers.0.self_attn.q_proj.weight: (2048, 2048) (torch.float32)
        model.layers.0.self_attn.k_proj.weight: (256, 2048) (torch.float32)
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    
    hf_token = _get_hf_token(token)
    
    # Get config first
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code, token=hf_token)
    
    # Load model with or without weights
    if use_empty_weights:
        try:
            from accelerate import init_empty_weights
        except ImportError:
            raise ImportError(
                "accelerate is required for empty weight inspection. "
                "Install with: pip install accelerate"
            )
        
        # Use from_config to create architecture without downloading weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=trust_remote_code
            )
    else:
        # Download full model (slow, memory-intensive)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=None,
            device_map="cpu"
        )
    
    # Extract all parameters and buffers
    tensors = []
    
    # Parameters (learnable weights)
    for name, param in model.named_parameters(recurse=True):
        tensors.append(("param", name, tuple(param.shape), str(param.dtype)))
    
    # Buffers (non-learnable tensors, e.g., position embeddings)
    for name, buffer in model.named_buffers(recurse=True):
        tensors.append(("buffer", name, tuple(buffer.shape), str(buffer.dtype)))
    
    return tensors


def compute_tensor_shapes_from_config(
    config: Dict[str, Any],
    model_type: str = None
) -> List[Tuple[str, str, Tuple[int, ...], str]]:
    """
    Compute expected tensor shapes directly from config without instantiating model.
    
    This is much faster than inspect_model_structure() as it doesn't create the model.
    It uses known architecture patterns for common model types.
    
    Args:
        config: Model config dictionary (from get_model_config)
        model_type: Override model type (default: infer from config)
        
    Returns:
        List of tuples: (kind, name, shape, dtype)
        
    Example:
        >>> config = get_model_config("mistralai/Mistral-7B-v0.1")
        >>> tensors = compute_tensor_shapes_from_config(config)
        >>> len(tensors)
        291
    """
    # Infer model type
    if model_type is None:
        model_type = config.get('model_type', 'llama')
    
    tensors = []
    dtype = "torch.float16"  # Assume fp16 for shape computation
    
    # Extract common config values
    hidden_size = config.get('hidden_size', 4096)
    num_layers = config.get('num_hidden_layers', 32)
    vocab_size = config.get('vocab_size', 32000)
    intermediate_size = config.get('intermediate_size', 11008)
    num_heads = config.get('num_attention_heads', 32)
    num_kv_heads = config.get('num_key_value_heads', num_heads)
    head_dim = hidden_size // num_heads
    
    # Embedding layer
    tensors.append(("param", "model.embed_tokens.weight", (vocab_size, hidden_size), dtype))
    
    # Transformer layers
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        
        # Attention projections
        tensors.append(("param", f"{prefix}.self_attn.q_proj.weight", (hidden_size, hidden_size), dtype))
        tensors.append(("param", f"{prefix}.self_attn.k_proj.weight", (num_kv_heads * head_dim, hidden_size), dtype))
        tensors.append(("param", f"{prefix}.self_attn.v_proj.weight", (num_kv_heads * head_dim, hidden_size), dtype))
        tensors.append(("param", f"{prefix}.self_attn.o_proj.weight", (hidden_size, hidden_size), dtype))
        
        # MLP (gated - SwiGLU style)
        tensors.append(("param", f"{prefix}.mlp.gate_proj.weight", (intermediate_size, hidden_size), dtype))
        tensors.append(("param", f"{prefix}.mlp.up_proj.weight", (intermediate_size, hidden_size), dtype))
        tensors.append(("param", f"{prefix}.mlp.down_proj.weight", (hidden_size, intermediate_size), dtype))
        
        # Layer norms (RMSNorm)
        tensors.append(("param", f"{prefix}.input_layernorm.weight", (hidden_size,), dtype))
        tensors.append(("param", f"{prefix}.post_attention_layernorm.weight", (hidden_size,), dtype))
    
    # Final norm and output
    tensors.append(("param", "model.norm.weight", (hidden_size,), dtype))
    tensors.append(("param", "lm_head.weight", (vocab_size, hidden_size), dtype))
    
    return tensors


def save_tensor_inspection(
    tensors: List[Tuple[str, str, Tuple[int, ...], str]],
    output_path: Path,
    format: str = "csv"
) -> None:
    """
    Save tensor inspection results to file.
    
    Args:
        tensors: Output from inspect_model_structure()
        output_path: Path to save file
        format: "csv" or "json"
    """
    import csv
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "csv":
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["kind", "name", "shape", "dtype"])
            writer.writerows(tensors)
    elif format == "json":
        tensor_dicts = [
            {
                "kind": kind,
                "name": name,
                "shape": list(shape),
                "dtype": dtype
            }
            for kind, name, shape, dtype in tensors
        ]
        with open(output_path, 'w') as f:
            json.dump(tensor_dicts, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'csv' or 'json'")


def format_tensor_sample(
    tensors: List[Tuple[str, str, Tuple[int, ...], str]],
    max_per_layer: int = 10,
    show_layers: List[int] = None
) -> str:
    """
    Format tensor inspection output for documentation/gap reports.
    
    Args:
        tensors: Output from inspect_model_structure()
        max_per_layer: Maximum tensors to show per layer
        show_layers: Specific layer indices to show (default: [0])
        
    Returns:
        Formatted string showing tensor examples
    """
    if show_layers is None:
        show_layers = [0]
    
    import re
    
    lines = ["```", "Tensor Inspection (Dry Run)", "=" * 40, ""]
    
    # Group tensors by category
    embedding_tensors = []
    layer_tensors = {idx: [] for idx in show_layers}
    final_tensors = []
    
    layer_pattern = re.compile(r'layers?\.(\d+)\.')
    
    for kind, name, shape, dtype in tensors:
        shape_str = str(shape)
        match = layer_pattern.search(name)
        
        if match:
            layer_idx = int(match.group(1))
            if layer_idx in show_layers:
                layer_tensors[layer_idx].append(f"{name}: {shape_str} ({dtype})")
        elif 'embed' in name.lower():
            embedding_tensors.append(f"{name}: {shape_str} ({dtype})")
        elif 'norm' in name.lower() and 'layer' not in name.lower():
            final_tensors.append(f"{name}: {shape_str} ({dtype})")
        elif 'lm_head' in name.lower() or 'output' in name.lower():
            final_tensors.append(f"{name}: {shape_str} ({dtype})")
    
    # Format output
    if embedding_tensors:
        lines.append("## Embeddings")
        lines.extend(embedding_tensors[:max_per_layer])
        lines.append("")
    
    for layer_idx in show_layers:
        if layer_tensors[layer_idx]:
            lines.append(f"## Layer {layer_idx}")
            lines.extend(layer_tensors[layer_idx][:max_per_layer])
            if len(layer_tensors[layer_idx]) > max_per_layer:
                lines.append(f"  ... ({len(layer_tensors[layer_idx]) - max_per_layer} more)")
            lines.append("")
    
    if final_tensors:
        lines.append("## Final Layers")
        lines.extend(final_tensors[:max_per_layer])
        lines.append("")
    
    lines.append("```")
    return "\n".join(lines)


def generate_gap_report_template(
    layer_name: str,
    tensors: List[Tuple[str, str, Tuple[int, ...], str]],
    tensor_pattern: str = None
) -> str:
    """
    Generate a gap report template for an unsupported layer type.
    
    Args:
        layer_name: Human-readable layer name (e.g., "Mixture of Experts Router")
        tensors: Full tensor list from inspect_model_structure()
        tensor_pattern: Regex pattern to filter relevant tensors (optional)
        
    Returns:
        Markdown template for the gap report
    """
    import re
    
    # Filter tensors if pattern provided
    if tensor_pattern:
        pattern = re.compile(tensor_pattern)
        relevant_tensors = [t for t in tensors if pattern.search(t[1])]
    else:
        relevant_tensors = tensors[:20]  # Default: first 20
    
    tensor_sample = "\n".join([
        f"  {name}: {shape} ({dtype})"
        for kind, name, shape, dtype in relevant_tensors[:15]
    ])
    
    template = f"""# Layer Gap Report: {layer_name}

## 1. Example Tensors (from dry run)

```
{tensor_sample}
```

## 2. Documentation References

### PyTorch
- Core operations: `torch.nn.Linear`, `torch.nn.functional.*`
- Relevant docs: https://pytorch.org/docs/stable/nn.html

### vLLM
- Implementation location: `vllm/model_executor/layers/`
- Docs: https://docs.vllm.ai/en/latest/

### SGLang
- Implementation patterns: TBD
- Docs: https://sgl-project.github.io/

## 3. Parallelism Strategies

| Strategy | Description | Communication Pattern |
|----------|-------------|----------------------|
| Tensor Parallelism | Split weights across devices | All-Reduce |
| Pipeline Parallelism | Split layers across devices | Point-to-Point |
| TBD | TBD | TBD |

## 4. Test Suite Specification

```python
# Tests needed for {layer_name}

def test_{layer_name.lower().replace(' ', '_')}_output_shape():
    \"\"\"Verify output shape matches expected dimensions\"\"\"
    pass

def test_{layer_name.lower().replace(' ', '_')}_flops():
    \"\"\"Verify FLOP calculation is correct\"\"\"
    pass

def test_{layer_name.lower().replace(' ', '_')}_memory():
    \"\"\"Verify memory calculation (weights + activations)\"\"\"
    pass

def test_{layer_name.lower().replace(' ', '_')}_parallelism():
    \"\"\"Verify correct behavior under tensor/pipeline parallelism\"\"\"
    pass
```

## 5. Implementation Notes

- [ ] Understand the mathematical operation
- [ ] Identify memory access patterns
- [ ] Determine optimal parallelism strategy
- [ ] Implement and test
"""
    return template


def analyze_layer_structure(tensors: List[Tuple[str, str, Tuple[int, ...], str]]) -> Dict[str, Any]:
    """
    Analyze tensor names to infer layer structure and types.
    
    This examines parameter names to identify:
    - Number of transformer layers
    - Attention types (MHA, GQA, MQA)
    - Layer components (attention, MLP, normalization)
    
    Args:
        tensors: Output from inspect_model_structure()
        
    Returns:
        Dictionary with inferred structure information
        
    Example:
        >>> tensors = inspect_model_structure("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> structure = analyze_layer_structure(tensors)
        >>> structure['num_layers']
        22
        >>> structure['attention_type']
        'GQA'
    """
    import re
    from collections import defaultdict
    
    analysis = {
        "num_layers": 0,
        "has_attention": False,
        "has_mlp": False,
        "has_normalization": False,
        "attention_projections": set(),
        "mlp_projections": set(),
        "layer_pattern": None,
    }
    
    param_names = [name for kind, name, _, _ in tensors if kind == "param"]
    
    # Find layer indices
    layer_indices = set()
    layer_pattern = re.compile(r'layers?\.(\d+)\.')
    for name in param_names:
        match = layer_pattern.search(name)
        if match:
            layer_indices.add(int(match.group(1)))
    
    if layer_indices:
        analysis['num_layers'] = max(layer_indices) + 1
        analysis['layer_pattern'] = 'model.layers.{idx}.'
    
    # Identify components
    for name in param_names:
        if 'attn' in name.lower() or 'attention' in name.lower():
            analysis['has_attention'] = True
            # Extract projection types
            if 'q_proj' in name or 'query' in name:
                analysis['attention_projections'].add('q')
            if 'k_proj' in name or 'key' in name:
                analysis['attention_projections'].add('k')
            if 'v_proj' in name or 'value' in name:
                analysis['attention_projections'].add('v')
            if 'o_proj' in name or 'out_proj' in name or 'output' in name:
                analysis['attention_projections'].add('o')
        
        if 'mlp' in name.lower() or 'ffn' in name.lower() or 'feed_forward' in name.lower():
            analysis['has_mlp'] = True
            # Extract projection types
            if 'gate_proj' in name or 'up_proj' in name or 'fc1' in name or 'w1' in name:
                analysis['mlp_projections'].add('up')
            if 'down_proj' in name or 'fc2' in name or 'w2' in name:
                analysis['mlp_projections'].add('down')
        
        if 'norm' in name.lower() or 'ln' in name.lower():
            analysis['has_normalization'] = True
    
    # Convert sets to lists for JSON serialization
    analysis['attention_projections'] = sorted(list(analysis['attention_projections']))
    analysis['mlp_projections'] = sorted(list(analysis['mlp_projections']))
    
    return analysis


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def count_parameters(
    tensors: List[Tuple[str, str, Tuple[int, ...], str]],
    include_buffers: bool = False
) -> Dict[str, Any]:
    """
    Calculate total parameter count from tensor inspection.
    
    Args:
        tensors: Output from inspect_model_structure()
        include_buffers: Whether to include buffers in count (default: False)
        
    Returns:
        Dictionary with parameter counts:
        - total: Total parameter count
        - total_formatted: Human-readable string (e.g., "70.0B")
        - by_component: Breakdown by component type
        
    Example:
        >>> tensors = inspect_model_structure("meta-llama/Llama-3.3-70B-Instruct")
        >>> counts = count_parameters(tensors)
        >>> counts['total_formatted']
        '69.5B'
    """
    import re
    from collections import defaultdict
    from functools import reduce
    import operator
    
    total = 0
    by_component = defaultdict(int)
    
    for kind, name, shape, dtype in tensors:
        if kind == "buffer" and not include_buffers:
            continue
            
        param_count = reduce(operator.mul, shape, 1)
        total += param_count
        
        # Categorize by component
        if 'embed' in name.lower():
            by_component['embedding'] += param_count
        elif 'attn' in name.lower() or 'attention' in name.lower():
            by_component['attention'] += param_count
        elif 'mlp' in name.lower() or 'ffn' in name.lower() or 'feed_forward' in name.lower():
            by_component['mlp'] += param_count
        elif 'norm' in name.lower() or 'ln' in name.lower():
            by_component['normalization'] += param_count
        elif 'lm_head' in name.lower() or 'output' in name.lower():
            by_component['output'] += param_count
        else:
            by_component['other'] += param_count
    
    # Format total
    if total >= 1e12:
        formatted = f"{total / 1e12:.1f}T"
    elif total >= 1e9:
        formatted = f"{total / 1e9:.1f}B"
    elif total >= 1e6:
        formatted = f"{total / 1e6:.1f}M"
    else:
        formatted = f"{total / 1e3:.1f}K"
    
    return {
        'total': total,
        'total_formatted': formatted,
        'by_component': dict(by_component)
    }


def estimate_memory(
    tensors: List[Tuple[str, str, Tuple[int, ...], str]],
    dtype: str = None,
    include_buffers: bool = True
) -> Dict[str, Any]:
    """
    Estimate memory footprint for model weights.
    
    Args:
        tensors: Output from inspect_model_structure()
        dtype: Override dtype for calculation (e.g., "float16", "bfloat16", "float32")
                If None, uses dtype from tensor inspection
        include_buffers: Whether to include buffers in calculation
        
    Returns:
        Dictionary with memory estimates:
        - bytes: Total memory in bytes
        - gb: Total memory in gigabytes
        - by_dtype: Breakdown by data type
        
    Example:
        >>> tensors = inspect_model_structure("meta-llama/Llama-3.3-70B-Instruct")
        >>> mem = estimate_memory(tensors, dtype="float16")
        >>> mem['gb']
        139.0
    """
    from functools import reduce
    import operator
    from collections import defaultdict
    
    # Bytes per dtype
    dtype_bytes = {
        'torch.float32': 4, 'float32': 4, 'fp32': 4,
        'torch.float16': 2, 'float16': 2, 'fp16': 2,
        'torch.bfloat16': 2, 'bfloat16': 2, 'bf16': 2,
        'torch.int8': 1, 'int8': 1,
        'torch.int4': 0.5, 'int4': 0.5,  # 4-bit quantization
    }
    
    total_bytes = 0
    by_dtype = defaultdict(int)
    
    for kind, name, shape, tensor_dtype in tensors:
        if kind == "buffer" and not include_buffers:
            continue
        
        # Use override dtype or tensor's dtype
        effective_dtype = dtype if dtype else tensor_dtype
        bytes_per_elem = dtype_bytes.get(effective_dtype.lower(), 4)  # Default to fp32
        
        param_count = reduce(operator.mul, shape, 1)
        mem = param_count * bytes_per_elem
        
        total_bytes += mem
        by_dtype[effective_dtype] += mem
    
    return {
        'bytes': total_bytes,
        'gb': total_bytes / (1024 ** 3),
        'mb': total_bytes / (1024 ** 2),
        'by_dtype': dict(by_dtype)
    }


def get_supported_layers(
    layers_dir: Path = None
) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically discover supported layer classes from backend/layers/.
    
    This inspects the actual layer implementations to determine what's supported,
    avoiding stale hardcoded lists.
    
    Args:
        layers_dir: Path to layers directory (default: auto-detect)
        
    Returns:
        Dictionary mapping class names to their info:
        - module: Source module name
        - file: Source file path
        - docstring: Class docstring (first line)
        
    Example:
        >>> layers = get_supported_layers()
        >>> list(layers.keys())
        ['EmbeddingLayer', 'GroupedQueryAttentionLayer', 'GatedMLPLayer', 'NormLayer']
    """
    import ast
    import sys
    
    # Auto-detect layers directory
    if layers_dir is None:
        # Try relative to this file first
        this_file = Path(__file__).resolve()
        layers_dir = this_file.parent.parent / "backend" / "layers"
        
        if not layers_dir.exists():
            # Try from cwd
            layers_dir = Path.cwd() / "backend" / "layers"
    
    layers_dir = Path(layers_dir)
    
    if not layers_dir.exists():
        raise FileNotFoundError(f"Layers directory not found: {layers_dir}")
    
    supported = {}
    
    # Parse each Python file in layers/
    for py_file in layers_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
            
        try:
            with open(py_file, 'r') as f:
                tree = ast.parse(f.read())
        except SyntaxError:
            continue
        
        module_name = py_file.stem
        
        # Find class definitions that end with "Layer"
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Layer"):
                # Skip base classes
                if node.name in ("Layer", "BaseLayer"):
                    continue
                
                # Extract docstring
                docstring = ""
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant)):
                    full_doc = node.body[0].value.value
                    docstring = full_doc.split('\n')[0].strip()
                
                supported[node.name] = {
                    'module': f"backend.layers.{module_name}",
                    'file': str(py_file),
                    'docstring': docstring,
                }
    
    return supported


def validate_config(
    tensors: List[Tuple[str, str, Tuple[int, ...], str]],
    expected_params: int = None,
    expected_layers: int = None,
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Validate computed values against expected values.
    
    Args:
        tensors: Output from inspect_model_structure()
        expected_params: Expected total parameter count (optional)
        expected_layers: Expected number of transformer layers (optional)
        tolerance: Acceptable relative difference (default: 1%)
        
    Returns:
        Dictionary with validation results:
        - valid: Overall validity (True if all checks pass)
        - checks: Individual check results
        - computed: Computed values
        - discrepancies: List of mismatches
        
    Example:
        >>> tensors = inspect_model_structure("meta-llama/Llama-3.3-70B-Instruct")
        >>> result = validate_config(tensors, expected_params=70_000_000_000)
        >>> result['valid']
        True
    """
    counts = count_parameters(tensors)
    structure = analyze_layer_structure(tensors)
    
    checks = {}
    discrepancies = []
    
    # Check parameter count
    if expected_params is not None:
        computed = counts['total']
        diff = abs(computed - expected_params) / expected_params
        passed = diff <= tolerance
        checks['parameter_count'] = {
            'passed': passed,
            'expected': expected_params,
            'computed': computed,
            'difference_pct': diff * 100
        }
        if not passed:
            discrepancies.append(
                f"Parameter count mismatch: expected {expected_params:,}, "
                f"got {computed:,} ({diff*100:.1f}% difference)"
            )
    
    # Check layer count
    if expected_layers is not None:
        computed = structure['num_layers']
        passed = computed == expected_layers
        checks['layer_count'] = {
            'passed': passed,
            'expected': expected_layers,
            'computed': computed
        }
        if not passed:
            discrepancies.append(
                f"Layer count mismatch: expected {expected_layers}, got {computed}"
            )
    
    return {
        'valid': all(c['passed'] for c in checks.values()) if checks else True,
        'checks': checks,
        'computed': {
            'total_params': counts['total'],
            'total_params_formatted': counts['total_formatted'],
            'num_layers': structure['num_layers'],
            'params_by_component': counts['by_component']
        },
        'discrepancies': discrepancies
    }


def validate_model_config_file(config_path: Path) -> Dict[str, Any]:
    """
    Validate a Mantile model config JSON file for consistency and correctness.
    
    Checks:
    - Layer names follow standard conventions (attention, feedforward, norm, embedding)
    - Layer order is consistent (attention, feedforward, norm, embedding)
    - No duplicate layer names
    - All required fields are present
    - model_id matches filename
    - Layer classes are valid
    
    Args:
        config_path: Path to model config JSON file
        
    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'errors': List[str],
            'warnings': List[str],
            'config': dict  # The loaded config
        }
        
    Example:
        >>> result = validate_model_config_file(Path("backend/data/model_configs/llama.json"))
        >>> if not result['valid']:
        ...     for error in result['errors']:
        ...         print(f"ERROR: {error}")
    """
    errors = []
    warnings = []
    
    # Standard layer naming conventions
    STANDARD_LAYER_NAMES = {'attention', 'feedforward', 'norm', 'embedding'}
    STANDARD_LAYER_ORDER = ['attention', 'feedforward', 'norm', 'embedding']
    
    # Valid layer classes
    VALID_LAYER_CLASSES = {
        'AttentionLayer',
        'GroupedQueryAttentionLayer',
        'MLPLayer',
        'GatedMLPLayer',
        'NormLayer',
        'EmbeddingLayer'
    }
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check model_id matches filename
        filename_without_ext = config_path.stem
        if config.get('model_id') != filename_without_ext:
            errors.append(
                f"model_id '{config.get('model_id')}' does not match filename '{filename_without_ext}'"
            )
        
        # Check required top-level fields
        required_fields = ['model_id', 'hf_model_id', 'name', 'hidden_size', 'num_layers', 'vocab_size', 'layer_types']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Check layer_types
        if 'layer_types' not in config:
            return {
                'valid': False,
                'errors': errors + ['Missing layer_types field'],
                'warnings': warnings,
                'config': config
            }
        
        layer_types = config['layer_types']
        if not isinstance(layer_types, list):
            errors.append("layer_types must be a list")
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'config': config}
        
        # Extract layer names
        layer_names = [lt.get('name') for lt in layer_types]
        
        # Check for duplicate layer names
        if len(layer_names) != len(set(layer_names)):
            duplicates = [n for n in layer_names if layer_names.count(n) > 1]
            errors.append(f"Duplicate layer names found: {set(duplicates)}")
        
        # Check layer naming conventions
        non_standard_names = set(layer_names) - STANDARD_LAYER_NAMES
        if non_standard_names:
            errors.append(
                f"Non-standard layer names found: {non_standard_names}. "
                f"Standard names are: {STANDARD_LAYER_NAMES}"
            )
        
        # Check layer order (warning only)
        present_standard_layers = [name for name in STANDARD_LAYER_ORDER if name in layer_names]
        actual_order = [name for name in layer_names if name in STANDARD_LAYER_NAMES]
        if actual_order != present_standard_layers:
            warnings.append(
                f"Layer order differs from standard: expected {present_standard_layers}, got {actual_order}"
            )
        
        # Check each layer
        for i, layer_type in enumerate(layer_types):
            layer_prefix = f"layer_types[{i}]"
            
            # Check required fields
            required_layer_fields = ['name', 'class', 'count', 'specs']
            for field in required_layer_fields:
                if field not in layer_type:
                    errors.append(f"{layer_prefix}: Missing required field '{field}'")
            
            # Check layer class is valid
            layer_class = layer_type.get('class')
            if layer_class and layer_class not in VALID_LAYER_CLASSES:
                errors.append(
                    f"{layer_prefix} ('{layer_type.get('name')}'): Unknown layer class '{layer_class}'. "
                    f"Valid classes: {VALID_LAYER_CLASSES}"
                )
            
            # Check specs is a dict
            if 'specs' in layer_type and not isinstance(layer_type['specs'], dict):
                errors.append(f"{layer_prefix}: 'specs' must be a dictionary")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'config': config
        }
        
    except json.JSONDecodeError as e:
        return {
            'valid': False,
            'errors': [f"Invalid JSON: {e}"],
            'warnings': [],
            'config': None
        }
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Error reading config: {e}"],
            'warnings': [],
            'config': None
        }


if __name__ == "__main__":
    # Example usage
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Inspect HuggingFace models and validate Mantile configs")
    subparsers = parser.add_subparsers(dest='command', help='Command to run', required=True)
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect HuggingFace model structure')
    inspect_parser.add_argument("model_id", help="HuggingFace model ID or local path")
    inspect_parser.add_argument("--config-out", help="Save config to JSON file")
    inspect_parser.add_argument("--tensors-out", help="Save tensor inspection to CSV file")
    inspect_parser.add_argument("--download", action="store_true", help="Download full weights")
    inspect_parser.add_argument("--analyze", action="store_true", help="Analyze layer structure")
    inspect_parser.add_argument("--validate", type=int, help="Expected parameter count for validation")
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate Mantile model config files')
    validate_parser.add_argument("config_paths", nargs='+', help="Path(s) to model config JSON files or directory")
    validate_parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    
    args = parser.parse_args()
    
    if args.command == 'inspect':
        # Get config
        print(f"Fetching config for {args.model_id}...")
        config = get_model_config(args.model_id, output_path=args.config_out)
        print(f"  Hidden size: {config.get('hidden_size')}")
        print(f"  Num layers: {config.get('num_hidden_layers')}")
        
        # Inspect structure
        print(f"\nInspecting model structure...")
        tensors = inspect_model_structure(args.model_id, use_empty_weights=not args.download)
        print(f"  Found {len(tensors)} tensors")
        
        if args.tensors_out:
            save_tensor_inspection(tensors, args.tensors_out)
            print(f"  Saved to {args.tensors_out}")
        
        # Count parameters
        counts = count_parameters(tensors)
        print(f"\nParameter count: {counts['total_formatted']} ({counts['total']:,})")
        for component, count in counts['by_component'].items():
            print(f"  {component}: {count:,}")
        
        # Memory estimate
        mem = estimate_memory(tensors)
        print(f"\nMemory (fp16): {mem['gb']:.1f} GB")
        
        # Analyze structure
        if args.analyze:
            print(f"\nAnalyzing layer structure...")
            structure = analyze_layer_structure(tensors)
            print(f"  Layers: {structure['num_layers']}")
            print(f"  Has attention: {structure['has_attention']}")
            print(f"  Has MLP: {structure['has_mlp']}")
            print(f"  Attention projections: {structure['attention_projections']}")
            print(f"  MLP projections: {structure['mlp_projections']}")
        
        # Validation
        if args.validate:
            print(f"\nValidating against expected {args.validate:,} parameters...")
            result = validate_config(tensors, expected_params=args.validate)
            print(f"  Valid: {result['valid']}")
            for disc in result['discrepancies']:
                print(f"  ⚠️  {disc}")
        
        # Show supported layers
        print(f"\nSupported Mantile layers:")
        try:
            layers = get_supported_layers()
            for name, info in layers.items():
                doc = info.get('docstring', 'No description')
                print(f"  {name}: {doc[:60]}..." if len(doc) > 60 else f"  {name}: {doc}")
        except FileNotFoundError as e:
            print(f"  Could not find layers directory: {e}")
    
    elif args.command == 'validate':
        # Collect all config files
        config_files = []
        for path_str in args.config_paths:
            path = Path(path_str)
            if path.is_dir():
                config_files.extend(path.glob('*.json'))
            elif path.is_file():
                config_files.append(path)
            else:
                print(f"Warning: {path_str} is not a valid file or directory")
        
        if not config_files:
            print("Error: No config files found")
            sys.exit(1)
        
        print(f"Validating {len(config_files)} config file(s)...\n")
        
        all_valid = True
        for config_file in sorted(config_files):
            result = validate_model_config_file(config_file)
            
            # Print results
            status = "✓" if result['valid'] else "✗"
            print(f"{status} {config_file.name}")
            
            if result['errors']:
                all_valid = False
                for error in result['errors']:
                    print(f"    ERROR: {error}")
            
            if result['warnings']:
                if args.strict:
                    all_valid = False
                for warning in result['warnings']:
                    prefix = "ERROR" if args.strict else "WARNING"
                    print(f"    {prefix}: {warning}")
            
            if result['valid'] and not result['warnings']:
                print(f"    All checks passed")
            
            print()
        
        # Summary
        if all_valid:
            print(f"✓ All {len(config_files)} config(s) validated successfully")
            sys.exit(0)
        else:
            print(f"✗ Validation failed for one or more configs")
            sys.exit(1)

