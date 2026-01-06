"""
Attention Layer Implementations

This module provides attention mechanism implementations:
- AttentionLayer: Vanilla multi-head attention (MHA)
- GroupedQueryAttentionLayer: Grouped query attention (GQA) with shared KV heads

Supported parallelism strategies:
- Tensor parallelism (TP): Split by heads (requires num_heads % tp == 0)
- Context parallelism (CP): Split sequence/KV (requires softmax reduction)
- Hybrid TP×CP: Combine both strategies
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
        - context_parallel: Split sequence/KV (KV-sharded attention with softmax reduction)
        - Hybrid: Both TP and CP simultaneously
    """
    
    SUPPORTED_PARALLELISM = {"tensor_parallel", "context_parallel"}
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        parallelism: Optional[dict] = None
    ):
        """
        Args:
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
        super().__init__(layer_idx, parallelism)
    
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
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        return tp * cp

    def compute_flops(self, batch_size: int, seq_len: int, phase: Phase, dtype: DataType) -> int:
        """
        FLOPs per chip for MHA:
        - Projections: Q,K,V,O with per-chip sizing by TP and CP
        - Attention: local queries × full keys (distributed across CP shards)
        Phase:
            Prefill: S×S attention on full sequence (S_local queries per chip)
            Decode: 1×S_past attention (1 new token attends to S_past cached tokens)
                    seq_len parameter represents S_past (cached context)
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        B = batch_size
        H = self.hidden_size
        d = self.head_dim
        h_per = self.num_heads // tp

        # Projections per chip
        flops = 0
        if phase == Phase.DECODE:
            # During decode: CP doesn't shard the single new token
            # Q is replicated, only KV cache is sharded
            S_new = 1
            S_past = seq_len
            flops += 2 * B * S_new * H * (H // tp)  # Q (for new token)
            flops += 2 * B * S_new * H * (H // tp)  # K (for new token)
            flops += 2 * B * S_new * H * (H // tp)  # V (for new token)
            flops += 2 * B * S_new * (H // tp) * H  # O (for new token)
            # Attention: 1×S_local (against local KV cache shard)
            S_local = S_past // cp
            flops += 2 * B * h_per * (S_local + 1) * d      # QK^T (1 × (S_local+1))
            flops += 2 * B * h_per * (S_local + 1) * d      # Attn * V
        else:  # PREFILL
            S = seq_len
            S_local = S // cp  # Local query tokens per chip
            # Projections: local tokens only
            flops += 2 * B * S_local * H * (H // tp)  # Q
            flops += 2 * B * S_local * H * (H // tp)  # K
            flops += 2 * B * S_local * H * (H // tp)  # V
            flops += 2 * B * S_local * (H // tp) * H  # O
            # Attention: local queries × full keys (S_local × S)
            # Implemented as sum over CP shards: local queries × local keys
            flops += 2 * B * h_per * S_local * S * d  # QK^T (S_local × S)
            flops += 2 * B * h_per * S_local * S * d  # Attn * V (S_local × S)
        return int(flops)
    
    def compute_weight_memory(self, dtype: DataType) -> int:
        # Weight memory per chip: TP shards weights, CP replicates them
        tp = self.parallelism.get("tensor_parallel", 1)
        params_per_chip = self.param_count // tp
        return int(params_per_chip * dtype.bytes_per_element)

    def compute_activation_memory(self, batch_size: int, seq_len: int, phase: Phase, dtype: DataType) -> int:
        """
        Approximate peak activations per chip (not counting attention score matrix).
        Most implementations tile/stream the attention scores, so we only count:
        - Input X, Q, K, V, and output Y (local tokens for CP)
        
        Phase:
            Prefill: Local tokens (M_local) with CP sharding
            Decode: Single new token (CP doesn't shard single token)
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        B = batch_size
        H = self.hidden_size

        elems = 0
        if phase == Phase.DECODE:
            # Process 1 new token (not sharded by CP)
            S_new = 1
            M_q = B * S_new
            d_local = H // tp
            # Input X_new: M_q × d
            elems += M_q * H
            # QKV local slices: 3 × M_q × d_local
            elems += 3 * M_q * d_local
            # Output Y_new: M_q × d
            elems += M_q * H
        else:  # PREFILL
            S = seq_len
            S_local = S // cp  # Local tokens per chip
            M_local = B * S_local
            # Input X_local: M_local × d
            elems += M_local * H
            # Q, K, V local: each M_local × d_local (sharded by TP)
            d_local = H // tp
            elems += 3 * M_local * d_local
            # Output Y_local: M_local × d
            elems += M_local * H
        return int(elems * dtype.bytes_per_element)

    def compute_kv_cache(self, batch_size: int, seq_len: int, dtype: DataType) -> int:
        # KV cache per chip with TP (head) and CP (sequence) sharding
        # Note: This is called by base class without phase info.
        # For decode with phase info, use _compute_kv_cache_with_phase()
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        h_per = self.num_heads // tp
        seq_local = seq_len // cp
        elements = 2 * batch_size * h_per * seq_local * self.head_dim
        return int(elements * dtype.bytes_per_element)
    
    def _compute_kv_cache_with_phase(self, batch_size: int, seq_len: int, phase: Phase, dtype: DataType) -> int:
        """
        Compute KV cache with phase awareness and CP sharding.
        - Prefill: cache size = seq_len/cp per chip (sharded by CP)
        - Decode: cache size = (seq_len + 1)/cp per chip (past + new token, sharded)
                  seq_len parameter represents past context
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        h_per = self.num_heads // tp
        
        if phase == Phase.PREFILL:
            cache_len_local = seq_len // cp
        else:  # DECODE
            # Cache includes past + new token, sharded by CP
            cache_len_local = (seq_len + 1) // cp
        
        elements = 2 * batch_size * h_per * cache_len_local * self.head_dim
        return int(elements * dtype.bytes_per_element)

    def _compute_communication_bytes(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase,
        dtype: DataType,
        hardware: dict
    ) -> Optional[int]:
        """
        Compute communication requirements for TP and/or CP.
        
        TP communication: All-reduce on attention output
        CP communication: Softmax stats reduction + output vector reduction
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        
        total_comm = 0
        
        # TP communication: All-reduce on attention output
        if tp > 1:
            tokens_this_step = 1 if phase == Phase.DECODE else seq_len // cp  # Local tokens
            tp_comm = batch_size * tokens_this_step * self.hidden_size * dtype.bytes_per_element
            total_comm += tp_comm
        
        # CP communication: Softmax stats + output reduction
        if cp > 1:
            if phase == Phase.DECODE:
                # Decode: Q is replicated, only 1 new token
                num_queries = batch_size * 1
            else:
                # Prefill: local queries
                num_queries = batch_size * (seq_len // cp)
            
            h_per = self.num_heads // tp  # Heads per chip
            
            # (a) Softmax stats: 2 FP32 scalars (max + sum) per query per head
            softmax_stat_bytes = 4  # FP32
            stats_comm = num_queries * h_per * 2 * softmax_stat_bytes
            
            # (b) Output vector reduction: sum partial outputs
            # Shape: [num_queries, d/tp] if TP, else [num_queries, d]
            output_width = self.hidden_size // tp if tp > 1 else self.hidden_size
            output_comm = num_queries * output_width * dtype.bytes_per_element
            
            cp_comm = stats_comm + output_comm
            total_comm += cp_comm
        
        return int(total_comm) if total_comm > 0 else None
    
    def compute_metrics(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase | str,
        dtype: DataType | str,
        hardware: Optional[dict] = None
    ):
        """Override to use phase-aware KV cache calculation"""
        # Convert string inputs to enums
        if isinstance(phase, str):
            phase = Phase(phase)
        if isinstance(dtype, str):
            dtype = DataType(dtype)
        
        # Call parent's compute_metrics
        metrics = super().compute_metrics(batch_size, seq_len, phase, dtype, hardware)
        
        # Replace KV cache with phase-aware version
        kv_cache_per_chip = self._compute_kv_cache_with_phase(batch_size, seq_len, phase, dtype)
        num_chips = self._get_num_chips()
        
        # Create new metrics with updated KV cache
        from dataclasses import replace
        return replace(
            metrics,
            kv_cache_per_chip=kv_cache_per_chip,
            kv_cache_total=kv_cache_per_chip * num_chips
        )


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
        - context_parallel: Split sequence/KV (KV-sharded attention with softmax reduction)
        - Hybrid: Both TP and CP simultaneously
    """
    
    SUPPORTED_PARALLELISM = {"tensor_parallel", "context_parallel"}
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        parallelism: Optional[dict] = None
    ):
        """
        Args:
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
        
        super().__init__(layer_idx, parallelism)
    
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
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        return tp * cp

    def compute_flops(self, batch_size: int, seq_len: int, phase: Phase, dtype: DataType) -> int:
        """
        FLOPs per chip for GQA:
        - Q projection uses full hidden; per chip output is H/tp
        - K,V projections use reduced hidden_kv = num_kv_heads * d; per chip H_kv/tp
        - Attention math uses num_heads_per_chip
        - O projection per chip uses H/tp input
        - CP shards sequence dimension for KV
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        B = batch_size
        H = self.hidden_size
        d = self.head_dim
        h_per = self.num_heads // tp
        H_kv = self.num_kv_heads * d

        flops = 0
        
        if phase == Phase.PREFILL:
            S = seq_len
            S_local = S // cp  # CP shards sequence
            M_local = B * S_local
            
            # Projections (local tokens)
            flops += 2 * M_local * H * (H // tp)        # Q
            flops += 2 * M_local * H * (H_kv // tp)     # K
            flops += 2 * M_local * H * (H_kv // tp)     # V
            flops += 2 * M_local * (H // tp) * H        # O

            # Attention math: local queries × full keys (full S)
            flops += 2 * B * h_per * S_local * S * d  # QK^T
            flops += 2 * B * h_per * S_local * S * d  # Attn * V
        else:  # DECODE
            # New token(s)
            T = 1
            M_q = B * T
            S_past = seq_len
            S_past_local = S_past // cp
            
            # Projections
            flops += 2 * M_q * H * (H // tp)        # Q
            flops += 2 * M_q * H * (H_kv // tp)     # K (for new token)
            flops += 2 * M_q * H * (H_kv // tp)     # V (for new token)
            flops += 2 * M_q * (H // tp) * H        # O
            
            # Attention: new queries × local past KV
            flops += 2 * B * h_per * T * S_past_local * d  # QK^T
            flops += 2 * B * h_per * T * S_past_local * d  # Attn * V
            
        return int(flops)
    
    def compute_weight_memory(self, dtype: DataType) -> int:
        """Weight memory per chip with TP sharding, CP replication"""
        tp = self.parallelism.get("tensor_parallel", 1)
        return int((self.param_count // tp) * dtype.bytes_per_element)

    def compute_activation_memory(self, batch_size: int, seq_len: int, phase: Phase, dtype: DataType) -> int:
        """
        Activation memory per chip for GQA.
        Counts: input X, Q/K/V projections, output Y (local tokens for CP).
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        B = batch_size
        H = self.hidden_size
        d = self.head_dim
        H_kv = self.num_kv_heads * d
        d_local = H // tp
        d_kv_local = H_kv // tp

        elems = 0
        
        if phase == Phase.DECODE:
            # Single new token (not sharded by CP)
            T = 1
            M_q = B * T
            # Input X_new: M_q × d (full width)
            elems += M_q * H
            # Q local: M_q × d_local
            elems += M_q * d_local
            # K, V local: M_q × d_kv_local (for new token)
            elems += M_q * d_kv_local
            elems += M_q * d_kv_local
            # Output Y_new: M_q × d (full width, materialized)
            elems += M_q * H
        else:  # PREFILL
            S = seq_len
            S_local = S // cp
            M_local = B * S_local
            # Input X_local: M_local × d
            elems += M_local * H
            # Q local: M_local × d_local
            elems += M_local * d_local
            # K, V local: M_local × d_kv_local
            elems += M_local * d_kv_local
            elems += M_local * d_kv_local
            # Output Y_local: M_local × d
            elems += M_local * H
            
        return int(elems * dtype.bytes_per_element)

    def compute_kv_cache(self, batch_size: int, seq_len: int, dtype: DataType) -> int:
        """KV cache per chip with TP (head) and CP (sequence) sharding"""
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        kv_per = self.num_kv_heads // tp
        seq_local = seq_len // cp
        elements = 2 * batch_size * kv_per * seq_local * self.head_dim
        return int(elements * dtype.bytes_per_element)
    
    def _compute_kv_cache_with_phase(self, batch_size: int, seq_len: int, phase: Phase, dtype: DataType) -> int:
        """Phase-aware KV cache calculation for GQA"""
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        kv_per = self.num_kv_heads // tp
        
        if phase == Phase.PREFILL:
            # Cache for full sequence
            cache_len = seq_len
        else:  # DECODE
            # Cache includes past + new token
            cache_len = seq_len + 1
        
        # Shard by CP
        cache_len_local = cache_len // cp
        elements = 2 * batch_size * kv_per * cache_len_local * self.head_dim
        return int(elements * dtype.bytes_per_element)

    def _compute_communication_bytes(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase,
        dtype: DataType,
        hardware: dict
    ) -> Optional[int]:
        """Communication for both TP and CP in GQA"""
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        
        total_comm = 0
        
        # TP communication: All-gather/all-reduce on attention output
        if tp > 1:
            if phase == Phase.DECODE:
                tokens_this_step = 1  # Single new token
            else:  # PREFILL
                tokens_this_step = seq_len // cp  # Local tokens
            tp_comm = batch_size * tokens_this_step * self.hidden_size * dtype.bytes_per_element
            total_comm += tp_comm
        
        # CP communication: Softmax stats + output reduction
        if cp > 1:
            if phase == Phase.DECODE:
                num_queries = batch_size * 1  # Single new token
            else:  # PREFILL
                num_queries = batch_size * (seq_len // cp)
            
            h_per = self.num_heads // tp  # Heads per chip
            
            # (a) Softmax stats: 2 FP32 scalars per query per head
            softmax_stat_bytes = 4  # FP32
            stats_comm = num_queries * h_per * 2 * softmax_stat_bytes
            
            # (b) Output vector reduction
            output_width = self.hidden_size // tp if tp > 1 else self.hidden_size
            output_comm = num_queries * output_width * dtype.bytes_per_element
            
            cp_comm = stats_comm + output_comm
            total_comm += cp_comm
        
        return int(total_comm) if total_comm > 0 else None
    
    def compute_metrics(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase | str,
        dtype: DataType | str,
        hardware: Optional[dict] = None
    ):
        """Override to use phase-aware KV cache calculation"""
        # Convert string inputs to enums
        if isinstance(phase, str):
            phase = Phase(phase)
        if isinstance(dtype, str):
            dtype = DataType(dtype)
        
        # Call parent's compute_metrics
        metrics = super().compute_metrics(batch_size, seq_len, phase, dtype, hardware)
        
        # Replace KV cache with phase-aware version
        kv_cache_per_chip = self._compute_kv_cache_with_phase(batch_size, seq_len, phase, dtype)
        num_chips = self._get_num_chips()
        
        # Create new metrics with updated KV cache
        from dataclasses import replace
        return replace(
            metrics,
            kv_cache_per_chip=kv_cache_per_chip,
            kv_cache_total=kv_cache_per_chip * num_chips
        )
