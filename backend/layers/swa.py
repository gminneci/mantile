"""
Sliding Window Attention (SWA) Layer Implementation

This module provides sliding window attention with:
- Configurable window size for limited attention span
- Attention sinks (preserved initial tokens)
- Optional biases on all projections
- Support for hidden_size ≠ num_heads * head_dim

Used by: GPT-OSS-120B, Mistral-7B, and other models with local attention.

Supported parallelism strategies:
- Tensor parallelism (TP): Split by heads
- Context parallelism (CP): Split sequence
"""

from typing import Optional
from .base import Layer, Phase, DataType


class SlidingWindowAttentionLayer(Layer):
    """
    Grouped Query Attention with Sliding Window and optional features.
    
    Architecture:
        Query heads: num_query_heads (each with own projection)
        KV heads: num_kv_heads (shared across query head groups)
        Window: Each query attends to at most `sliding_window` recent keys
        Sinks: First `num_sinks` tokens always in attention span
    
    Projection dimensions:
        - Q_proj: hidden_size → d_q (where d_q = num_query_heads * head_dim)
        - K_proj: hidden_size → d_kv (where d_kv = num_kv_heads * head_dim)
        - V_proj: hidden_size → d_kv
        - O_proj: d_q → hidden_size
        
    Note: hidden_size may differ from d_q (e.g., GPT-OSS: 2880 vs 4096)
    
    Supported parallelism:
        - tensor_parallel: Split by heads
        - context_parallel: Split sequence
    """
    
    SUPPORTED_PARALLELISM = {"tensor_parallel", "context_parallel"}
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: Optional[int] = None,
        num_sinks: int = 0,
        has_bias: bool = False,
        dtype: DataType | str = "bf16",
        parallelism: Optional[dict] = None
    ):
        """
        Args:
            layer_idx: Layer index in the model
            hidden_size: Model hidden dimension (d)
            num_query_heads: Number of query heads (h_q)
            num_kv_heads: Number of KV heads (h_kv)
            head_dim: Dimension per head (dh)
            sliding_window: Maximum attention span (None = unlimited)
            num_sinks: Number of sink tokens to preserve (default: 0)
            has_bias: Whether projections have bias terms (default: False)
            dtype: Numerical precision
            parallelism: Parallelism config
        """
        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.num_sinks = num_sinks
        self.has_bias = has_bias
        
        # Compute projection dimensions
        # Note: d_q may differ from hidden_size
        self.d_q = num_query_heads * head_dim  # Q output and O input dimension
        self.d_kv = num_kv_heads * head_dim    # K/V output dimension
        
        # Compute parameter counts
        # Q: [hidden_size, d_q]
        # K: [hidden_size, d_kv]
        # V: [hidden_size, d_kv]
        # O: [d_q, hidden_size]
        q_params = hidden_size * self.d_q
        k_params = hidden_size * self.d_kv
        v_params = hidden_size * self.d_kv
        o_params = self.d_q * hidden_size
        
        self.weight_params = q_params + k_params + v_params + o_params
        
        # Bias parameters
        self.bias_params = 0
        if has_bias:
            # bq: [d_q], bk: [d_kv], bv: [d_kv], bo: [hidden_size]
            self.bias_params = self.d_q + self.d_kv + self.d_kv + hidden_size
        
        self.param_count = self.weight_params + self.bias_params
        
        super().__init__(layer_idx, dtype, parallelism)
    
    def _validate_parallelism(self) -> None:
        """Validate parallelism constraints"""
        super()._validate_parallelism()
        
        tp = self.parallelism.get("tensor_parallel", 1)
        
        if self.num_query_heads % tp != 0:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be divisible by TP ({tp})"
            )
        if self.num_kv_heads % tp != 0:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) must be divisible by TP ({tp})"
            )
    
    def _get_num_packages(self) -> int:
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        return tp * cp
    
    def _effective_attention_span(self, seq_len: int) -> int:
        """
        Compute effective attention span with sliding window and sinks.
        
        For prefill: min(seq_len, num_sinks + sliding_window)
        For decode: num_sinks + min(past_len, sliding_window)
        
        Returns the maximum number of KV positions a query can attend to.
        """
        if self.sliding_window is None:
            return seq_len  # No window limit
        
        # With sinks, attention spans both sinks and recent window
        effective = self.num_sinks + min(seq_len, self.sliding_window)
        return min(effective, seq_len)  # Can't exceed actual sequence
    
    def _kv_cache_positions(self, seq_len: int) -> int:
        """
        Compute number of KV cache positions needed.
        
        With sliding window: min(seq_len, num_sinks + sliding_window)
        Without: full seq_len
        """
        if self.sliding_window is None:
            return seq_len
        
        # Cache holds sinks + sliding window
        max_positions = self.num_sinks + self.sliding_window
        return min(seq_len, max_positions)
    
    def compute_flops(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        Compute FLOPs per chip for SWA.
        
        Projections scale with tokens (M).
        Attention scores scale with M × effective_span (limited by window).
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        
        B = batch_size
        d = self.hidden_size
        d_q = self.d_q
        d_kv = self.d_kv
        dh = self.head_dim
        h_q_local = self.num_query_heads // tp
        
        d_q_local = d_q // tp
        d_kv_local = d_kv // tp
        
        flops = 0
        
        if phase == Phase.DECODE:
            # Decode: process 1 new token, attend to past KV cache
            T = 1
            M_q = B * T
            S_past = seq_len
            
            # Effective attention span (limited by window + sinks)
            attn_span = self._effective_attention_span(S_past)
            
            # Q projection: X[M_q, d] @ Wq[d, d_q] + bias
            flops += 2 * M_q * d * d_q  # GEMM
            if self.has_bias:
                flops += M_q * d_q  # bias add
            
            # K projection: X[M_q, d] @ Wk[d, d_kv] + bias
            flops += 2 * M_q * d * d_kv
            if self.has_bias:
                flops += M_q * d_kv
            
            # V projection
            flops += 2 * M_q * d * d_kv
            if self.has_bias:
                flops += M_q * d_kv
            
            # Attention scores: Q[M_q, h_q, dh] @ K[attn_span, h_kv, dh]^T
            # Per chip: h_q_local heads
            flops += 2 * B * h_q_local * T * attn_span * dh
            
            # Apply attention to V
            flops += 2 * B * h_q_local * T * attn_span * dh
            
            # O projection: O[M_q, d_q] @ Wo[d_q, d] + bias
            flops += 2 * M_q * d_q * d
            if self.has_bias:
                flops += M_q * d
            
            # Divide projections by TP (they're sharded)
            # Note: attention math already uses h_q_local
            proj_flops_total = (
                2 * M_q * d * d_q + (M_q * d_q if self.has_bias else 0) +
                2 * M_q * d * d_kv + (M_q * d_kv if self.has_bias else 0) +
                2 * M_q * d * d_kv + (M_q * d_kv if self.has_bias else 0) +
                2 * M_q * d_q * d + (M_q * d if self.has_bias else 0)
            )
            attn_flops_total = 2 * B * h_q_local * T * attn_span * dh * 2  # scores + apply
            
            # Total: projections split by TP, attention uses local heads
            flops = proj_flops_total // tp + attn_flops_total
            
        else:  # PREFILL
            S = seq_len
            S_local = S // cp
            M = B * S_local
            
            # Effective attention span
            # For prefill, each query at position i attends to min(i+1, window+sinks) positions
            # Simplified: use min(S, window+sinks) as the effective span per query
            if self.sliding_window is None:
                attn_span = S  # Full attention
            else:
                # With SWA, queries attend to ~window positions on average
                attn_span = min(S, self.num_sinks + self.sliding_window)
            
            # Projections (on local tokens M)
            q_flops = 2 * M * d * d_q + (M * d_q if self.has_bias else 0)
            k_flops = 2 * M * d * d_kv + (M * d_kv if self.has_bias else 0)
            v_flops = 2 * M * d * d_kv + (M * d_kv if self.has_bias else 0)
            o_flops = 2 * M * d_q * d + (M * d if self.has_bias else 0)
            
            proj_flops_total = q_flops + k_flops + v_flops + o_flops
            
            # Attention: S_local queries × attn_span keys per query
            # Note: With SWA in prefill, we approximate as S_local × min(S, W)
            attn_flops = 2 * B * h_q_local * S_local * attn_span * dh * 2  # scores + apply
            
            # Total: projections split by TP, attention uses local heads
            flops = proj_flops_total // tp + attn_flops
        
        return int(flops)
    
    def compute_weight_memory(self) -> int:
        """
        Compute weight memory per chip in bytes.
        
        Weights are sharded by TP along head dimension.
        Biases: Q/K/V biases sharded, O bias replicated.
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        
        bytes_per_elem = self.dtype.bytes_per_element
        
        # Weight matrices sharded by TP
        weight_elems_per_chip = self.weight_params // tp
        
        # Biases
        if self.has_bias:
            # Q, K, V biases sharded by TP
            qkv_bias_elems = (self.d_q + self.d_kv + self.d_kv) // tp
            # O bias replicated (full hidden_size)
            o_bias_elems = self.hidden_size
            bias_elems_per_chip = qkv_bias_elems + o_bias_elems
        else:
            bias_elems_per_chip = 0
        
        total_elems = weight_elems_per_chip + bias_elems_per_chip
        return int(total_elems * bytes_per_elem)
    
    def compute_activation_memory(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        Compute activation memory per chip in bytes.
        
        Resident tensors: X, Q, K, V, Y (not attention scores - streamed/tiled).
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        
        B = batch_size
        d = self.hidden_size
        d_q_local = self.d_q // tp
        d_kv_local = self.d_kv // tp
        
        bytes_per_elem = self.dtype.bytes_per_element
        
        elems = 0
        
        if phase == Phase.DECODE:
            # Single new token
            T = 1
            M_q = B * T
            
            # Input X: [M_q, d]
            elems += M_q * d
            # Q local: [M_q, d_q_local]
            elems += M_q * d_q_local
            # K local: [M_q, d_kv_local]
            elems += M_q * d_kv_local
            # V local: [M_q, d_kv_local]
            elems += M_q * d_kv_local
            # Output Y: [M_q, d]
            elems += M_q * d
            
        else:  # PREFILL
            S = seq_len
            S_local = S // cp
            M = B * S_local
            
            # Input X: [M, d]
            elems += M * d
            # Q local: [M, d_q_local]
            elems += M * d_q_local
            # K local: [M, d_kv_local]
            elems += M * d_kv_local
            # V local: [M, d_kv_local]
            elems += M * d_kv_local
            # Output Y: [M, d]
            elems += M * d
        
        return int(elems * bytes_per_elem)
    
    def compute_kv_cache(self, batch_size: int, seq_len: int) -> int:
        """
        Compute KV cache per chip in bytes.
        
        With sliding window: cache limited to (num_sinks + sliding_window) positions.
        Without: full seq_len.
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        
        kv_heads_local = self.num_kv_heads // tp
        
        # KV cache positions (limited by window)
        cache_positions = self._kv_cache_positions(seq_len)
        
        # Shard by CP
        cache_positions_local = cache_positions // cp if cp > 1 else cache_positions
        
        # K + V cache
        elements = 2 * batch_size * kv_heads_local * cache_positions_local * self.head_dim
        return int(elements * self.dtype.bytes_per_element)
    
    def _compute_kv_cache_with_phase(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """Phase-aware KV cache calculation."""
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        
        kv_heads_local = self.num_kv_heads // tp
        
        if phase == Phase.PREFILL:
            # Cache for processed sequence
            cache_len = seq_len
        else:  # DECODE
            # Cache includes past (seq_len already includes past)
            # For decode, seq_len represents past_seq_len
            cache_len = seq_len
        
        # Apply window limit
        cache_positions = self._kv_cache_positions(cache_len)
        
        # Shard by CP
        cache_positions_local = cache_positions // cp if cp > 1 else cache_positions
        
        elements = 2 * batch_size * kv_heads_local * cache_positions_local * self.head_dim
        return int(elements * self.dtype.bytes_per_element)
    
    def _compute_communication_bytes(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase,
        hardware: dict
    ) -> Optional[int]:
        """Communication for TP and CP."""
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        
        total_comm = 0
        
        # TP all-reduce on output
        if tp > 1:
            if phase == Phase.DECODE:
                tokens = batch_size * 1
            else:
                tokens = batch_size * (seq_len // cp)
            tp_comm = tokens * self.hidden_size * self.dtype.bytes_per_element
            total_comm += int(tp_comm)
        
        # CP communication (softmax stats + output reduction)
        if cp > 1:
            if phase == Phase.DECODE:
                num_queries = batch_size * 1
            else:
                num_queries = batch_size * (seq_len // cp)
            
            h_q_local = self.num_query_heads // tp
            
            # Softmax stats: 2 FP32 scalars per query per head
            stats_comm = num_queries * h_q_local * 2 * 4  # FP32
            
            # Output reduction
            output_width = self.hidden_size // tp if tp > 1 else self.hidden_size
            output_comm = num_queries * output_width * self.dtype.bytes_per_element
            
            total_comm += int(stats_comm + output_comm)
        
        return int(total_comm) if total_comm > 0 else None
    
    def compute_metrics(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase | str,
        hardware: Optional[dict] = None
    ):
        """Override to use phase-aware KV cache calculation."""
        if isinstance(phase, str):
            phase = Phase(phase)
        
        # Call parent's compute_metrics
        metrics = super().compute_metrics(batch_size, seq_len, phase, hardware)
        
        # Replace KV cache with phase-aware version
        kv_cache_per_package = self._compute_kv_cache_with_phase(batch_size, seq_len, phase)
        num_packages = self._get_num_packages()
        
        from dataclasses import replace
        return replace(
            metrics,
            kv_cache_per_package=kv_cache_per_package,
            kv_cache_total=kv_cache_per_package * num_packages
        )
