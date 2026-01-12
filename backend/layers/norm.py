"""
Normalization Layer Implementations

This module provides normalization layer implementations:
- NormLayer: LayerNorm and RMSNorm

Normalization layers are typically replicated across all devices and don't use
sharding, though they're compatible with various parallelism strategies.
"""

from typing import Optional
from .base import Layer, Phase, DataType


class NormLayer(Layer):
    """
    Layer normalization (LayerNorm or RMSNorm).
    
    Architecture:
        Single learnable scale vector (and optionally bias for LayerNorm).
        RMSNorm: x / RMS(x) * scale
        LayerNorm: (x - mean) / std * scale + bias
    
    Parameters:
        1 vector of size hidden_size (scale)
        Optionally 1 more for bias (LayerNorm only)
    
    Supported parallelism:
        - Typically replicated across all chips (no sharding)
        - For TP: replicated on each chip
    """
    
    # Norm layers accept any parallelism but are replicated (not sharded)
    SUPPORTED_PARALLELISM = {"tensor_parallel", "pipeline_parallel", "expert_parallel"}
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        has_bias: bool = False,
        dtype: DataType | str = "bf16",
        parallelism: Optional[dict] = None
    ):
        """
        Args:
            layer_idx: Layer index
            hidden_size: Model hidden dimension
            has_bias: Whether LayerNorm has bias term (False for RMSNorm)
            dtype: Numerical precision (DataType enum or string like 'bf16')
            parallelism: Parallelism config (see Layer base class)
        """
        self.hidden_size = hidden_size
        self.has_bias = has_bias
        self.param_count = hidden_size * (2 if has_bias else 1)
        super().__init__(layer_idx, dtype, parallelism)
    
    def _get_num_chips(self) -> int:
        return 1
    
    def compute_flops(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        Approximate LN/RMSNorm flops per chip: ~5 ops per element (mean/var, scale, bias).
        """
        B, S, H = batch_size, seq_len, self.hidden_size
        return int(5 * B * S * H)
    
    def compute_weight_memory(self) -> int:
        return int(self.param_count * self.dtype.bytes_per_element)
    
    def compute_activation_memory(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        Output tensor size dominates: B*S*H elements.
        """
        B, S, H = batch_size, seq_len, self.hidden_size
        return int(B * S * H * self.dtype.bytes_per_element)
