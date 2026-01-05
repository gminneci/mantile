"""
Attention Layer Implementations

This module provides attention mechanism implementations:
- AttentionLayer: Vanilla multi-head attention (MHA)
- GroupedQueryAttentionLayer: Grouped query attention (GQA) with shared KV heads

Supported parallelism strategies:
- Tensor parallelism (TP): Split by heads (requires num_heads % tp == 0)
"""

from typing import Optional
from .base import Layer, Phase, DataType


class AttentionLayer(Layer):
    """
    Vanilla multi-head attention (MHA) layer.
    
    Architecture:
        All heads have their own Q, K, V projections.
        Q, K, V: (batch, seq, hidden) -> (batch, num_heads, seq, head_dim)
        Attention: softmax(QK^T / sqrt(d)) V
        Output: (batch, seq, hidden) -> (batch, seq, hidden)
    
    Parameters:
        4 weight matrices: Q_proj, K_proj, V_proj, O_proj
    
    Supported parallelism:
        - tensor_parallel: Split by heads (requires num_heads % tp == 0)
    """
    
    SUPPORTED_PARALLELISM = {"tensor_parallel"}
    
    def __init__(
        self,
        name: str,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        parallelism: Optional[dict] = None
    ):
        """
        Args:
            name: Layer name
            layer_idx: Layer index
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head (typically hidden_size // num_heads)
            parallelism: Parallelism config (see Layer base class)
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Q, K, V, O projections (all same size for vanilla MHA)
        self.param_count = 4 * hidden_size * hidden_size
        super().__init__(name, layer_idx, parallelism)
    
    def _validate_parallelism(self) -> None:
        """MHA supports head-parallel TP with divisibility constraint"""
        # Call parent validation first
        super()._validate_parallelism()
        
        # Additional validation: num_heads must be divisible by TP degree
        if "tensor_parallel" in self.parallelism:
            tp = self.parallelism["tensor_parallel"]
            if self.num_heads % tp != 0:
                raise ValueError(f"num_heads ({self.num_heads}) must be divisible by TP degree ({tp})")
    
    def _get_num_chips(self) -> int:
        return self.parallelism.get("tensor_parallel", 1)

    def compute_flops(self, batch_size: int, seq_len: int, phase: Phase, dtype: DataType) -> int:
        """
        FLOPs per chip for MHA:
        - Projections: Q,K,V,O with per-chip output/input sized by 1/tp
        - Attention compute: uses num_heads_per_chip
        Prefill: S×S attention; Decode: 1×S
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        B, S = batch_size, seq_len
        H = self.hidden_size
        d = self.head_dim
        h_per = self.num_heads // tp

        # Projections per chip
        flops = 0
        flops += 2 * B * S * H * (H // tp)  # Q
        flops += 2 * B * S * H * (H // tp)  # K
        flops += 2 * B * S * H * (H // tp)  # V
        flops += 2 * B * S * (H // tp) * H  # O

        # Attention math
        if phase == Phase.PREFILL:
            flops += 2 * B * h_per * S * S * d  # QK^T
            flops += 2 * B * h_per * S * S * d  # Attn * V
        else:  # DECODE (1 x S)
            flops += 2 * B * h_per * S * d      # QK^T
            flops += 2 * B * h_per * S * d      # Attn * V
        return int(flops)
    
    def compute_weight_memory(self, dtype: DataType) -> int:
        return int(self.param_count * dtype.bytes_per_element)

    def compute_activation_memory(self, batch_size: int, seq_len: int, phase: Phase, dtype: DataType) -> int:
        """
        Approximate peak activations per chip: Q, K, V, scores, and output shard.
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        B, S = batch_size, seq_len
        H = self.hidden_size
        d = self.head_dim
        h_per = self.num_heads // tp

        elems = 0
        # Q, K, V shards
        elems += B * S * (H // tp)  # Q
        elems += B * S * (H // tp)  # K
        elems += B * S * (H // tp)  # V
        # Attention scores
        if phase == Phase.PREFILL:
            elems += B * h_per * S * S
        else:
            elems += B * h_per * S  # 1 x S per head
        # Output partial before all-reduce
        elems += B * S * (H // tp)
        return int(elems * dtype.bytes_per_element)

    def compute_kv_cache(self, batch_size: int, seq_len: int, dtype: DataType) -> int:
        # KV cache per chip with head-parallel sharding
        h_per = self.num_heads // self.parallelism.get("tensor_parallel", 1)
        elements = 2 * batch_size * h_per * seq_len * self.head_dim
        return int(elements * dtype.bytes_per_element)

    def _compute_communication_bytes(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase,
        dtype: DataType,
        hardware: dict
    ) -> Optional[int]:
        # All-reduce on attention output of size B*S*H
        tp = self.parallelism.get("tensor_parallel", 1)
        if tp > 1:
            return int(batch_size * seq_len * self.hidden_size * dtype.bytes_per_element)
        return None


class GroupedQueryAttentionLayer(Layer):
    """
    Grouped Query Attention (GQA) with shared KV heads (e.g., LLaMA 2/3).
    
    Architecture:
        Query heads: num_heads (each with its own projection)
        KV heads: num_kv_heads (shared across query head groups)
        Each KV head is shared by (num_heads // num_kv_heads) query heads
    
    Parameters:
        - Q_proj: hidden_size * (num_heads * head_dim)
        - K_proj: hidden_size * (num_kv_heads * head_dim)
        - V_proj: hidden_size * (num_kv_heads * head_dim)
        - O_proj: hidden_size * hidden_size
    
    Supported parallelism:
        - tensor_parallel: Split by query head groups (requires num_heads % tp == 0)
          Note: KV heads are also split, so num_kv_heads % tp == 0 required
    """
    
    SUPPORTED_PARALLELISM = {"tensor_parallel"}
    
    def __init__(
        self,
        name: str,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        parallelism: Optional[dict] = None
    ):
        """
        Args:
            name: Layer name
            layer_idx: Layer index
            hidden_size: Model hidden dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (num_heads for MHA, 1 for MQA)
            head_dim: Dimension per head
            parallelism: Parallelism config (see Layer base class)
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # Q: full size, K/V: reduced size
        q_params = hidden_size * (num_heads * head_dim)
        k_params = hidden_size * (num_kv_heads * head_dim)
        v_params = hidden_size * (num_kv_heads * head_dim)
        o_params = hidden_size * hidden_size
        self.param_count = q_params + k_params + v_params + o_params
        
        super().__init__(name, layer_idx, parallelism)
    
    def _validate_parallelism(self) -> None:
        """GQA supports head-parallel TP with divisibility constraints"""
        # Call parent validation first
        super()._validate_parallelism()
        
        # Additional validation: both num_heads and num_kv_heads must be divisible by TP degree
        if "tensor_parallel" in self.parallelism:
            tp = self.parallelism["tensor_parallel"]
            if self.num_heads % tp != 0:
                raise ValueError(f"num_heads ({self.num_heads}) must be divisible by TP degree ({tp})")
            if self.num_kv_heads % tp != 0:
                raise ValueError(f"num_kv_heads ({self.num_kv_heads}) must be divisible by TP degree ({tp})")
    
    def _get_num_chips(self) -> int:
        return self.parallelism.get("tensor_parallel", 1)

    def compute_flops(self, batch_size: int, seq_len: int, phase: Phase, dtype: DataType) -> int:
        """
        FLOPs per chip for GQA:
        - Q projection uses full hidden; per chip output is H/tp
        - K,V projections use reduced hidden_kv = num_kv_heads * d; per chip H_kv/tp
        - Attention math uses num_heads_per_chip
        - O projection per chip uses H/tp input
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        B, S = batch_size, seq_len
        H = self.hidden_size
        d = self.head_dim
        h_per = self.num_heads // tp
        kv_per = self.num_kv_heads // tp
        H_kv = self.num_kv_heads * d

        flops = 0
        # Projections
        flops += 2 * B * S * H * (H // tp)        # Q
        flops += 2 * B * S * H * (H_kv // tp)     # K
        flops += 2 * B * S * H * (H_kv // tp)     # V
        flops += 2 * B * S * (H // tp) * H        # O

        # Attention math (still scales with query heads)
        if phase == Phase.PREFILL:
            flops += 2 * B * h_per * S * S * d  # QK^T
            flops += 2 * B * h_per * S * S * d  # Attn * V
        else:
            flops += 2 * B * h_per * S * d
            flops += 2 * B * h_per * S * d
        return int(flops)
    
    def compute_weight_memory(self, dtype: DataType) -> int:
        return int(self.param_count * dtype.bytes_per_element)

    def compute_activation_memory(self, batch_size: int, seq_len: int, phase: Phase, dtype: DataType) -> int:
        tp = self.parallelism.get("tensor_parallel", 1)
        B, S = batch_size, seq_len
        H = self.hidden_size
        d = self.head_dim
        h_per = self.num_heads // tp
        H_kv = self.num_kv_heads * d

        elems = 0
        elems += B * S * (H // tp)      # Q
        elems += B * S * (H_kv // tp)   # K
        elems += B * S * (H_kv // tp)   # V
        if phase == Phase.PREFILL:
            elems += B * h_per * S * S
        else:
            elems += B * h_per * S
        elems += B * S * (H // tp)      # output partial
        return int(elems * dtype.bytes_per_element)

    def compute_kv_cache(self, batch_size: int, seq_len: int, dtype: DataType) -> int:
        tp = self.parallelism.get("tensor_parallel", 1)
        kv_per = self.num_kv_heads // tp
        elements = 2 * batch_size * kv_per * seq_len * self.head_dim
        return int(elements * dtype.bytes_per_element)

    def _compute_communication_bytes(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase,
        dtype: DataType,
        hardware: dict
    ) -> Optional[int]:
        tp = self.parallelism.get("tensor_parallel", 1)
        if tp > 1:
            return int(batch_size * seq_len * self.hidden_size * dtype.bytes_per_element)
        return None
