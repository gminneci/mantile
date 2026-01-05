"""
Layer Modeling for LLM Performance Analysis
============================================

This package provides a hierarchical class system for modeling individual LLM layers
with hardware-aware performance characteristics.

TODO - Future Enhancements:
---------------------------
1. COLLECTIVE OPERATIONS: Model specific collective ops (All-Reduce, All-Gather, Reduce-Scatter)
   - Different hardware has different performance characteristics for each
   - Current implementation uses generic communication bytes/time
   
2. HARDWARE TOPOLOGY: Implement detailed network topology modeling
   - NVLink vs PCIe vs inter-node bandwidth
   - Switch fabric topology
   - NUMA effects
   - Multi-tier memory hierarchy (HBM, DRAM, NVMe)

PURPOSE:
--------
Each Layer class encapsulates the computational and memory characteristics of a specific
layer type (MLP, Attention, Normalization, etc.) and can compute:
  - Memory footprint (weights, activations, KV cache)
  - FLOPs (floating point operations)
  - Compute time (hardware-dependent)
  - Weight loading time (memory bandwidth-dependent)
  - Inter-chip communication requirements and time (for distributed execution)

ARCHITECTURE:
-------------
- Base classes (base.py): Layer, Phase, DataType, LayerMetrics
- MLP layers (mlp.py): MLPLayer, GatedMLPLayer
- Attention layers (attention.py): AttentionLayer, GroupedQueryAttentionLayer
- Normalization layers (norm.py): NormLayer

USAGE:
------
1. Instantiate a layer with architectural parameters AND parallelism config
2. Call compute methods with runtime parameters (batch_size, seq_len, phase, etc.)
3. Retrieve both per-chip and aggregate metrics

Example:
    from backend.layers import GroupedQueryAttentionLayer
    
    # Single-chip layer
    layer = GroupedQueryAttentionLayer(
        name="layer_0_attn",
        layer_idx=0,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        parallelism=None  # No sharding
    )
    
    # Multi-chip with tensor parallelism
    layer = GroupedQueryAttentionLayer(
        name="layer_0_attn",
        layer_idx=0,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        parallelism={"tensor_parallel": 2}  # Split across 2 chips
    )
    
    metrics = layer.compute_metrics(
        batch_size=1,
        seq_len=2048,
        phase="prefill",
        dtype="bf16",
        hardware=h100_config
    )
    
    print(f"Per-chip FLOPs: {metrics.flops_per_chip}")
    print(f"Total FLOPs: {metrics.flops_total}")
    print(f"Communication: {metrics.communication_bytes} bytes")
"""

# Base classes
from .base import Layer, Phase, DataType, LayerMetrics

# MLP layers
from .mlp import MLPLayer, GatedMLPLayer

# Attention layers
from .attention import AttentionLayer, GroupedQueryAttentionLayer

# Normalization layers
from .norm import NormLayer

__all__ = [
    # Base classes
    "Layer",
    "Phase",
    "DataType",
    "LayerMetrics",
    # MLP layers
    "MLPLayer",
    "GatedMLPLayer",
    # Attention layers
    "AttentionLayer",
    "GroupedQueryAttentionLayer",
    # Normalization layers
    "NormLayer",
]
