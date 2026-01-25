"""
Embedding Layer Implementation

This module provides embedding layer implementation for token embeddings.
"""

from typing import Optional
from .base import Layer, Phase, DataType

# Threshold above which vocab is sharded across TP (rather than replicated)
# Rationale: Small vocabs (<100K) fit comfortably on chip and benefit from replication
# (no all-reduce needed). Large vocabs (>100K) are memory-expensive and benefit from sharding.
VOCAB_SHARD_THRESHOLD = 100_000


class EmbeddingLayer(Layer):
    """
    Token embedding layer.
    
    Architecture:
        Maps token IDs to dense vectors: vocab_size × hidden_size lookup table
    
    Parameters:
        Single weight matrix of size [vocab_size, hidden_size]
    
    Supported parallelism:
        - tensor_parallel: Shard by vocab (row-parallel embedding)
        - Typically replicated for small vocabs, sharded for large
    
    Note:
        - FLOPs are minimal (lookup, not compute)
        - Memory dominated by weight matrix
        - For very large vocabs (>100K), may shard across TP
    """
    
    SUPPORTED_PARALLELISM = {"tensor_parallel", "pipeline_parallel"}
    default_kernel_count = 1  # Single embedding lookup kernel
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: DataType | str,
        parallelism: Optional[dict] = None
    ):
        """
        Args:
            vocab_size: Vocabulary size
            hidden_size: Model hidden dimension
            dtype: Numerical precision (DataType enum or string like 'bf16')
            parallelism: Parallelism config (see Layer base class)
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Parameter count: vocab_size × hidden_size
        self.param_count = vocab_size * hidden_size
        
        # Embeddings are typically layer_idx=-1 (before transformer layers)
        super().__init__(layer_idx=-1, dtype=dtype, parallelism=parallelism)
    
    def _get_num_packages(self) -> int:
        """Number of chips for embedding layer"""
        tp = self.parallelism.get("tensor_parallel", 1)
        return tp
    
    def compute_flops(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        FLOPs for embedding lookup.
        
        Embeddings are essentially memory lookups, not compute operations.
        FLOPs are negligible compared to attention/MLP.
        
        Return 0 to indicate no significant compute.
        """
        return 0
    
    def compute_weight_memory(self) -> int:
        """
        Weight memory per chip for embedding table.
        
        For TP: shard vocabulary across chips (row-parallel)
        For small vocabs: typically replicated (no sharding)
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        
        # For very large vocabs, shard across TP to reduce per-chip memory
        # Otherwise, replicate for simplicity (no all-reduce needed)
        if self.vocab_size > VOCAB_SHARD_THRESHOLD:
            # Shard by vocab dimension
            params_per_chip = self.param_count // tp
        else:
            # Replicate
            params_per_chip = self.param_count
        
        return int(params_per_chip * self.dtype.bytes_per_element)
    
    def compute_activation_memory(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        Activation memory for embedding layer.
        
        Output: [batch_size, seq_len, hidden_size]
        Input token IDs are negligible (int32, much smaller than embeddings)
        """
        B = batch_size
        H = self.hidden_size
        
        if phase == Phase.DECODE:
            # Single new token
            S = 1
        else:  # PREFILL
            S = seq_len
        
        # Output embeddings: [B, S, H]
        elements = B * S * H
        return int(elements * self.dtype.bytes_per_element)
    
    def compute_kv_cache(self, batch_size: int, seq_len: int) -> int:
        """Embedding layer has no KV cache"""
        return 0
    
    def _compute_communication_bytes(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase,
        hardware: dict
    ) -> Optional[int]:
        """
        Communication for embedding layer.
        
        If vocab is sharded (row-parallel), need all-reduce or all-gather.
        For small vocabs (replicated), no communication.
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        
        if tp > 1 and self.vocab_size > VOCAB_SHARD_THRESHOLD:
            # Vocab sharded, need all-reduce on output
            B = batch_size
            H = self.hidden_size
            S = 1 if phase == Phase.DECODE else seq_len
            
            # All-reduce output: [B, S, H]
            comm_bytes = B * S * H * self.dtype.bytes_per_element
            return int(comm_bytes)
        
        return 0
