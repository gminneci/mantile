
# ============================================================================
# DEPRECATED: This file is deprecated and will be removed in future versions.
# ============================================================================

from pydantic import BaseModel
from typing import List, Optional, Dict

class HardwareSpecs(BaseModel):
    name: str
    description: Optional[str] = None
    
    # Compute (TFLOPS per chip/package) - Scaled to user selection (e.g. 1 chip or full rack)
    fp16_tflops: float
    bf16_tflops: float
    fp8_tflops: float
    int8_tops: float
    
    # Memory
    hbm_capacity_gb: float
    hbm_bandwidth_gbps: float
    
    # Interconnect (GB/s per direction or total, need to be consistent)
    interconnect_bandwidth_gbps: float
    interconnect_latency_us: float = 0.0
    
    # Topology details
    chips_per_node: int = 1
    nodes_per_cluster: int = 1

class LayerSpecs(BaseModel):
    name: str # e.g. "layer_0_attn"
    layer_idx: int
    module_type: str # "attention", "feedforward", "norm"
    
    # Dimensions
    input_dim: int
    output_dim: int
    
    # Params
    parameter_count: int
    
    # For Attn
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    kv_heads: Optional[int] = None
    
    # For MLP
    hidden_dim: Optional[int] = None # intermediate size

class ModelIR(BaseModel):
    name: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    layers: List[LayerSpecs]
