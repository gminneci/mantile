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

class ParallelismConfig(BaseModel):
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    sp_size: int = 1 # sequence parallelism
    
    batch_size: int = 1
    input_seq_len: int = 128
    output_seq_len: int = 128

class EstimateResult(BaseModel):
    # Time
    total_latency_ms: float
    time_to_first_token_ms: float # TTFT (Prefill)
    time_per_output_token_ms: float # TPOT (Decode)
    total_throughput_tokens_s: float
    
    # Memory (GB)
    weights_mem_gb: float
    kv_cache_mem_gb: float
    activation_mem_gb: float
    total_mem_gb: float
    max_mem_capacity_gb: float
    
    # Bottleneck Analysis
    compute_bound_percent: float # 0-100
    memory_bound_percent: float # 0-100
    comm_bound_percent: float # 0-100
    
    # Detailed Breakdown (optional, for drilldown)
    # layers: List[LayerEstimate]
