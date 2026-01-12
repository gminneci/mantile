"""
MLP Layer Implementations

This module provides MLP (Multi-Layer Perceptron) layer implementations:
- MLPLayer: Standard 2-projection MLP
- GatedMLPLayer: 3-projection gated MLP (e.g., SwiGLU)

Supported parallelism strategies:
- Tensor parallelism (TP): Column-parallel up_proj, row-parallel down_proj
- Sequence parallelism (SP): Partition sequence dimension to reduce activation memory
- Pipeline parallelism (PP): Split into separate pipeline stages
"""

from typing import Optional
from .base import Layer, Phase, DataType


class MLPLayer(Layer):
    """
    Standard feedforward MLP layer with 2 projections.
    
    Architecture:
        x -> up_proj (hidden → intermediate) -> activation -> down_proj (intermediate → hidden)
    
    Parameters:
        Configurable number of projections:
          - 2: up_proj, down_proj (standard MLP)
          - 3: gate_proj, up_proj, down_proj (gated MLP / SwiGLU)
    
    Supported parallelism:
        - tensor_parallel: Column-parallel up_proj, row-parallel down_proj
        - pipeline_parallel: Can split into separate stages
        - sequence_parallel: Partition sequence dimension to reduce activation memory
    """
    
    SUPPORTED_PARALLELISM = {"tensor_parallel", "pipeline_parallel", "sequence_parallel"}
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int,
        num_projections: int = 2,
        dtype: DataType | str = "bf16",
        parallelism: Optional[dict] = None
    ):
        """
        Args:
            layer_idx: Layer index
            hidden_size: Model hidden dimension
            intermediate_size: MLP intermediate dimension (typically 4 * hidden_size)
            num_projections: 2 for standard MLP, 3 for gated/SwiGLU (default: 2)
            dtype: Numerical precision (DataType enum or string like 'bf16')
            parallelism: Parallelism config (see Layer base class)
        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_projections = num_projections
        
        # Validate projections count early
        if self.num_projections not in (2, 3):
            raise ValueError(f"MLPLayer num_projections must be 2 or 3, got {self.num_projections}")
        
        # Parameter count scales with number of projections
        # 2: up (H->I) + down (I->H) => 2 * H * I
        # 3: gate (H->I) + up (H->I) + down (I->H) => 3 * H * I
        self.param_count = hidden_size * intermediate_size * (2 if self.num_projections == 2 else 3)
        
        super().__init__(layer_idx, dtype, parallelism)
    
    def _get_num_chips(self) -> int:
        """Number of chips used for this layer's shard. TP modeled; PP TODO."""
        # TODO: Model pipeline-parallel stages explicitly with per-stage metrics
        # Note: Sequence parallelism typically shares devices with TP (doesn't add chips)
        tp = self.parallelism.get("tensor_parallel", 1)
        return tp
    
    def compute_flops(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        FLOPs per chip for MLP projections (matmuls dominate).
        - Two-projection MLP: up (H→I) + down (I→H)
        - Three-projection MLP: gate (H→I) + up (H→I) + down (I→H) + small elemwise (ignored)
        Tensor parallelism:
            Column-parallel for up/gate, row-parallel for down ⇒ per-chip flops scale by 1/tp
        Sequence parallelism:
            Sequence is partitioned ⇒ per-chip flops scale by 1/sp
        Phase:
            Prefill: process full seq_len tokens
            Decode: process 1 token (autoregressive generation), SP not effective
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        sp = self.parallelism.get("sequence_parallel", 1)
        H, I = self.hidden_size, self.intermediate_size
        B = batch_size
        
        # During decode, we only process 1 token at a time (SP not effective)
        if phase == Phase.DECODE:
            S = 1
            S_per_chip = 1  # No sequence parallelism benefit for single token
        else:
            S = seq_len
            S_per_chip = S // sp  # Distribute sequence across chips in prefill
        
        mm = 0
        # Up projection: (B, S/sp, H) @ (H, I/tp)
        mm += 2 * B * S_per_chip * H * (I // tp)
        # Optional gate projection
        if self.num_projections == 3:
            mm += 2 * B * S_per_chip * H * (I // tp)
        # Down projection: (B, S/sp, I/tp) @ (I/tp, H)
        mm += 2 * B * S_per_chip * (I // tp) * H
        return int(mm)
    
    def compute_weight_memory(self) -> int:
        """Memory = param_count * bytes_per_element"""
        return int(self.param_count * self.dtype.bytes_per_element)
    
    def compute_activation_memory(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        Approximate peak activations per chip:
        - Up (and gate) outputs: B*(S/sp)*(I/tp)
        - Down input/output buffers: B*(S/sp)*(I/tp) and B*(S/sp)*H
        Sequence parallelism reduces activation memory by factor of sp.
        Phase:
            Prefill: process full seq_len tokens
            Decode: process 1 token (autoregressive generation), SP not effective
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        sp = self.parallelism.get("sequence_parallel", 1)
        H, I = self.hidden_size, self.intermediate_size
        B = batch_size
        
        # During decode, we only process 1 token at a time (SP not effective)
        if phase == Phase.DECODE:
            S = 1
            S_per_chip = 1  # No sequence parallelism benefit for single token
        else:
            S = seq_len
            S_per_chip = S // sp  # Distribute sequence across chips in prefill
        
        elems = 0
        elems += B * S_per_chip * (I // tp)  # up output
        if self.num_projections == 3:
            elems += B * S_per_chip * (I // tp)  # gate output
        elems += B * S_per_chip * (I // tp)  # down input
        elems += B * S_per_chip * H  # down output (full H before reduce-scatter with SP)
        return int(elems * self.dtype.bytes_per_element)

    def _compute_communication_bytes(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase,
        hardware: dict
    ) -> Optional[int]:
        """
        Communication for MLP with TP and SP:
        - Tensor parallel: All-reduce on output (B*S*H bytes)
        - Sequence parallel: All-gather input + Reduce-scatter output
          Input all-gather: (B, S/sp, H) → (B, S, H) = B*S*H bytes
          Output reduce-scatter: (B, S, H) → (B, S/sp, H) = B*S*H bytes
        Pipeline-parallel comms (stage transfer) are TODO.
        Phase:
            Prefill: communicate for full seq_len
            Decode: communicate for 1 token only (SP not effective)
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        sp = self.parallelism.get("sequence_parallel", 1)
        B = batch_size
        H = self.hidden_size
        
        # During decode, we only process 1 token at a time
        S = 1 if phase == Phase.DECODE else seq_len
        
        comm_bytes = 0
        
        # Tensor-parallel all-reduce (if enabled)
        if tp > 1:
            comm_bytes += B * S * H * self.dtype.bytes_per_element
        
        # Sequence-parallel communication (if enabled)
        if sp > 1:
            # All-gather input: gather sequence shards before MLP
            comm_bytes += B * S * H * self.dtype.bytes_per_element
            # Reduce-scatter output: reduce and scatter back to sequence shards
            comm_bytes += B * S * H * self.dtype.bytes_per_element
        
        return int(comm_bytes) if comm_bytes > 0 else None


class GatedMLPLayer(MLPLayer):
    """
    Thin wrapper over `MLPLayer` configured with 3 projections (gate, up, down).
    Use when modeling LLaMA-style SwiGLU/gated MLP blocks.
    
    Supported parallelism: same as `MLPLayer` (TP, PP, and SP).
    """
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: DataType | str = "bf16",
        parallelism: Optional[dict] = None
    ):
        super().__init__(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_projections=3,
            dtype=dtype,
            parallelism=parallelism,
        )
