"""
Stateless utility functions for model configuration and analysis.
All functions are pure and require explicit context (no global state).
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from backend.models import HardwareSpecs, ModelIR, LayerSpecs
from backend.model_library import load_model_config
from backend.hardware_library import load_hardware_config, list_available_configs
from backend.layers import (
    AttentionLayer,
    GroupedQueryAttentionLayer,
    MLPLayer,
    GatedMLPLayer,
    NormLayer,
    EmbeddingLayer,
    Phase,
    DataType,
)


@dataclass
class LayerConfig:
    """Configuration for a single layer type."""
    layer_type: str  # "embedding", "attention", "mlp", "norm"
    layer_name: str
    parallelism: Dict[str, int]  # {"tensor_parallel": 4, "context_parallel": 2, etc}
    num_instances: int = 1  # How many layers of this type (e.g., 80 attention layers)
    dtype: str = "bf16"  # Numerical precision format


@dataclass
class ModelValidation:
    """Validation results for a model."""
    valid: bool
    num_layers: int
    total_params: int
    expected_params: Optional[int]
    hidden_size: int
    vocab_size: int
    attention_type: str  # "MHA" or "GQA"
    mlp_type: str  # "dense" or "gated"
    issues: List[str]


@dataclass
class SystemRequirements:
    """Minimum system requirements for the configuration."""
    min_chips: int
    total_weight_memory_gb: float
    total_activation_memory_gb: float
    total_kv_cache_gb: float
    memory_per_chip_gb: float
    fits_on_hardware: bool
    hw_capacity_gb: float


# ============================================================================
# Pure Functions - All stateless, require explicit context
# ============================================================================

def validate_model(model_ir: ModelIR) -> ModelValidation:
    """
    Validate a model and return detailed info.
    Checks parameter count, layer configs, etc.
    
    Args:
        model_ir: Model intermediate representation
        
    Returns:
        ModelValidation with details and any issues found
    """
    issues = []
    
    # Count parameters
    total_params = sum(layer.parameter_count for layer in model_ir.layers)
    
    # Determine attention type
    attention_type = "MHA"
    for layer in model_ir.layers:
        if layer.module_type == "attention" and layer.kv_heads:
            if layer.kv_heads < layer.num_heads:
                attention_type = "GQA"
                break
    
    # Determine MLP type
    mlp_type = "dense"
    for layer in model_ir.layers:
        if layer.module_type == "feedforward" and layer.hidden_dim:
            if layer.hidden_dim > layer.input_dim * 2:
                mlp_type = "gated"
                break
    
    return ModelValidation(
        valid=len(issues) == 0,
        num_layers=model_ir.num_layers,
        total_params=total_params,
        expected_params=None,
        hidden_size=model_ir.hidden_size,
        vocab_size=model_ir.vocab_size,
        attention_type=attention_type,
        mlp_type=mlp_type,
        issues=issues
    )


def get_layer_types(model_ir: ModelIR) -> List[str]:
    """
    Get all unique layer types in the model.
    
    Args:
        model_ir: Model intermediate representation
        
    Returns:
        List of unique layer type strings
    """
    return list(set(layer.module_type for layer in model_ir.layers))


def instantiate_layer(
    layer_spec: LayerSpecs,
    parallelism: Dict[str, int],
    model_ir: ModelIR,
    hardware: HardwareSpecs
):
    """
    Instantiate a layer object from specification.
    Filters parallelism types based on what each layer supports.
    
    Args:
        layer_spec: Layer specification from model IR
        parallelism: Parallelism configuration dict
        model_ir: Full model IR (for context like vocab_size)
        hardware: Hardware configuration (for validation)
        
    Returns:
        Instantiated Layer object or None if unsupported
    """
    if layer_spec.module_type == "attention":
        # Attention supports: TP, CP
        filtered_parallelism = {
            k: v for k, v in parallelism.items()
            if k in ["tensor_parallel", "context_parallel", "pipeline_parallel"]
        }
        
        if layer_spec.kv_heads and layer_spec.kv_heads < layer_spec.num_heads:
            # GQA
            return GroupedQueryAttentionLayer(
                layer_idx=layer_spec.layer_idx,
                hidden_size=layer_spec.input_dim,
                num_heads=layer_spec.num_heads,
                num_kv_heads=layer_spec.kv_heads,
                head_dim=layer_spec.head_dim,
                parallelism=filtered_parallelism
            )
        else:
            # MHA
            return AttentionLayer(
                layer_idx=layer_spec.layer_idx,
                hidden_size=layer_spec.input_dim,
                num_heads=layer_spec.num_heads,
                head_dim=layer_spec.head_dim,
                parallelism=filtered_parallelism
            )
    
    elif layer_spec.module_type == "feedforward":
        # MLP supports: TP, SP (NOT CP)
        filtered_parallelism = {
            k: v for k, v in parallelism.items()
            if k in ["tensor_parallel", "sequence_parallel", "pipeline_parallel"]
        }
        
        if layer_spec.hidden_dim and layer_spec.hidden_dim > layer_spec.input_dim * 2:
            # Gated MLP
            return GatedMLPLayer(
                layer_idx=layer_spec.layer_idx,
                hidden_size=layer_spec.input_dim,
                intermediate_size=layer_spec.hidden_dim,
                parallelism=filtered_parallelism
            )
        else:
            # Regular MLP
            return MLPLayer(
                layer_idx=layer_spec.layer_idx,
                hidden_size=layer_spec.input_dim,
                intermediate_size=layer_spec.hidden_dim or layer_spec.input_dim * 4,
                parallelism=filtered_parallelism
            )
    
    elif layer_spec.module_type == "norm":
        # Norm doesn't use parallelism (replicated)
        return NormLayer(
            layer_idx=layer_spec.layer_idx,
            hidden_size=layer_spec.input_dim,
            has_bias=False,
            parallelism={}
        )
    
    elif layer_spec.module_type == "embedding":
        # Embedding doesn't use parallelism (replicated)
        return EmbeddingLayer(
            vocab_size=model_ir.vocab_size,
            hidden_size=model_ir.hidden_size,
            parallelism={}
        )
    
    else:
        return None


def calculate_minimum_system(
    model_ir: ModelIR,
    hardware: HardwareSpecs,
    layer_configs: Dict[str, LayerConfig],
    batch_size: int = 1,
    seq_length: int = 2048,
    phase: Phase = Phase.PREFILL,
    dtype: DataType = DataType.BF16
) -> SystemRequirements:
    """
    Calculate minimum system requirements given layer configurations.
    
    Args:
        model_ir: Model intermediate representation
        hardware: Hardware specification
        layer_configs: Per-layer-type parallelism configurations
        batch_size: Batch size for computation
        seq_length: Sequence length
        phase: Inference phase (prefill/decode)
        dtype: Data type precision
        
    Returns:
        SystemRequirements with memory and chip requirements
    """
    total_weight_memory = 0.0
    total_activation_memory = 0.0
    total_kv_cache = 0.0
    max_chips = 1
    
    for layer_type, config in layer_configs.items():
        # Find a representative layer of this type
        sample_layer_spec = next(
            (l for l in model_ir.layers if l.module_type == layer_type),
            None
        )
        if not sample_layer_spec:
            continue
        
        # Instantiate the actual layer to compute metrics
        layer = instantiate_layer(sample_layer_spec, config.parallelism, model_ir, hardware)
        if not layer:
            continue
        
        # Compute metrics for one instance
        metrics = layer.compute_metrics(
            batch_size=batch_size,
            seq_len=seq_length,
            phase=phase,
            dtype=dtype
        )
        
        # Get number of chips for this layer type
        max_chips = max(max_chips, metrics.num_chips)
        
        # Aggregate for all layers of this type
        total_weight_memory += metrics.weight_memory_per_chip * config.num_instances
        total_activation_memory += metrics.activation_memory_per_chip * config.num_instances
        total_kv_cache += metrics.kv_cache_per_chip * config.num_instances
    
    # Convert to GB
    total_weight_memory_gb = total_weight_memory / 1e9
    total_activation_memory_gb = total_activation_memory / 1e9
    total_kv_cache_gb = total_kv_cache / 1e9
    
    # Calculate memory per chip
    memory_per_chip_gb = (
        total_weight_memory_gb + 
        total_activation_memory_gb + 
        total_kv_cache_gb
    )
    
    # Check if it fits
    hw_capacity_gb = hardware.hbm_capacity_gb
    fits = memory_per_chip_gb <= hw_capacity_gb
    
    return SystemRequirements(
        min_chips=max_chips,
        total_weight_memory_gb=total_weight_memory_gb,
        total_activation_memory_gb=total_activation_memory_gb,
        total_kv_cache_gb=total_kv_cache_gb,
        memory_per_chip_gb=memory_per_chip_gb,
        fits_on_hardware=fits,
        hw_capacity_gb=hw_capacity_gb
    )


def compute_phase_metrics(
    model_ir: ModelIR,
    hardware: HardwareSpecs,
    layer_configs: Dict[str, LayerConfig],
    batch_size: int,
    seq_length: int,
    phase: Phase,
    dtype: DataType
) -> Dict:
    """
    Compute metrics for a single inference phase.
    
    Args:
        model_ir: Model intermediate representation
        hardware: Hardware specification
        layer_configs: Per-layer-type parallelism configurations
        batch_size: Batch size
        seq_length: Sequence length
        phase: Inference phase (prefill/decode)
        dtype: Data type precision
        
    Returns:
        Dict with aggregated metrics for this phase
    """
    total_flops = 0.0
    total_weight_memory = 0.0
    total_activation_memory = 0.0
    total_kv_cache = 0.0
    max_chips = 1
    
    for layer_type, config in layer_configs.items():
        # Find a representative layer
        sample_layer_spec = next(
            (l for l in model_ir.layers if l.module_type == layer_type),
            None
        )
        if not sample_layer_spec:
            continue
        
        # Instantiate layer
        layer = instantiate_layer(sample_layer_spec, config.parallelism, model_ir, hardware)
        if not layer:
            continue
        
        # Compute metrics
        metrics = layer.compute_metrics(
            batch_size=batch_size,
            seq_len=seq_length,
            phase=phase,
            dtype=dtype
        )
        
        # Aggregate
        total_flops += metrics.flops_total * config.num_instances
        total_weight_memory += metrics.weight_memory_per_chip * config.num_instances
        total_activation_memory += metrics.activation_memory_per_chip * config.num_instances
        total_kv_cache += metrics.kv_cache_per_chip * config.num_instances
        max_chips = max(max_chips, metrics.num_chips)
    
    memory_per_chip = (
        total_weight_memory + 
        total_activation_memory + 
        total_kv_cache
    ) / 1e9 if max_chips > 0 else 0
    
    return {
        "total_flops": total_flops,
        "flops_per_chip": total_flops / max_chips if max_chips > 0 else 0,
        "total_weight_memory": total_weight_memory,
        "total_activation_memory": total_activation_memory,
        "total_kv_cache": total_kv_cache,
        "memory_per_chip": memory_per_chip,
        "num_chips": max_chips,
    }


def compute_system_metrics(
    model_ir: ModelIR,
    hardware: HardwareSpecs,
    layer_configs: Dict[str, LayerConfig],
    batch_size: int = 1,
    input_seq: int = 2048,
    output_seq: int = 128,
    dtype: DataType = DataType.BF16
) -> Dict:
    """
    Compute full system metrics (TTFT, TPOT, throughput, etc).
    
    Args:
        model_ir: Model intermediate representation
        hardware: Hardware specification
        layer_configs: Per-layer-type parallelism configurations
        batch_size: Batch size
        input_seq: Input sequence length (prefill)
        output_seq: Output sequence length (decode tokens)
        dtype: Data type precision
        
    Returns:
        Dict with comprehensive system-level metrics
    """
    # Prefill phase metrics
    prefill_metrics = compute_phase_metrics(
        model_ir=model_ir,
        hardware=hardware,
        layer_configs=layer_configs,
        batch_size=batch_size,
        seq_length=input_seq,
        phase=Phase.PREFILL,
        dtype=dtype
    )
    
    # Decode phase metrics (per token)
    decode_metrics = compute_phase_metrics(
        model_ir=model_ir,
        hardware=hardware,
        layer_configs=layer_configs,
        batch_size=batch_size,
        seq_length=1,  # Decode is one token at a time
        phase=Phase.DECODE,
        dtype=dtype
    )
    
    # Calculate latencies
    peak_tflops_per_chip = hardware.bf16_tflops
    hbm_bw_per_chip = hardware.hbm_bandwidth_gbps
    
    # Prefill latency (TTFT)
    prefill_compute_time_ms = (prefill_metrics["total_flops"] / 1e12) / peak_tflops_per_chip * 1000
    prefill_memory_time_ms = (prefill_metrics["total_weight_memory"] / 1e9) / hbm_bw_per_chip * 1000
    ttft_ms = max(prefill_compute_time_ms, prefill_memory_time_ms)
    
    # Decode latency (TPOT)
    decode_compute_time_ms = (decode_metrics["total_flops"] / 1e12) / peak_tflops_per_chip * 1000
    decode_memory_time_ms = (decode_metrics["total_weight_memory"] / 1e9) / hbm_bw_per_chip * 1000
    tpot_ms = max(decode_compute_time_ms, decode_memory_time_ms)
    
    # Throughput (tokens/sec)
    throughput_tokens_s = 1000.0 / tpot_ms if tpot_ms > 0 else 0
    
    # Total latency for full sequence
    total_latency_ms = ttft_ms + (tpot_ms * output_seq)
    
    # Bottleneck analysis
    if prefill_compute_time_ms > prefill_memory_time_ms * 1.2:
        bottleneck = "compute"
    elif prefill_memory_time_ms > prefill_compute_time_ms * 1.2:
        bottleneck = "memory"
    else:
        bottleneck = "balanced"
    
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "throughput_tokens_s": throughput_tokens_s,
        "total_latency_ms": total_latency_ms,
        "prefill": {
            "flops_total": prefill_metrics["total_flops"],
            "flops_per_chip": prefill_metrics["flops_per_chip"],
            "compute_time_ms": prefill_compute_time_ms,
            "memory_time_ms": prefill_memory_time_ms,
        },
        "decode": {
            "flops_total": decode_metrics["total_flops"],
            "flops_per_chip": decode_metrics["flops_per_chip"],
            "compute_time_ms": decode_compute_time_ms,
            "memory_time_ms": decode_memory_time_ms,
        },
        "memory": {
            "weight_memory_gb": prefill_metrics["total_weight_memory"] / 1e9,
            "activation_memory_gb": prefill_metrics["total_activation_memory"] / 1e9,
            "kv_cache_gb": prefill_metrics["total_kv_cache"] / 1e9,
            "total_memory_gb": (
                prefill_metrics["total_weight_memory"] +
                prefill_metrics["total_activation_memory"] +
                prefill_metrics["total_kv_cache"]
            ) / 1e9,
            "memory_per_chip_gb": prefill_metrics["memory_per_chip"],
            "hw_capacity_gb": hardware.hbm_capacity_gb,
        },
        "system": {
            "num_chips": prefill_metrics["num_chips"],
            "bottleneck": bottleneck,
            "fits_on_hardware": prefill_metrics["memory_per_chip"] <= hardware.hbm_capacity_gb,
        }
    }
