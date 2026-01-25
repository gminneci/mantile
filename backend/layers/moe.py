"""
Mixture of Experts (MoE) Layer Implementation

This module provides MoE layer implementations:
- MoELayer: Standard MoE with top-k routing

MoE routes each token to a subset of experts (top-k), enabling larger model
capacity without proportional compute increase.

Supported parallelism strategies:
- Expert parallelism (EP): Distribute experts across chips (all-to-all comm)
- Tensor parallelism (TP): Shard each expert's FFN (all-reduce comm)
- Context parallelism (CP): Shard sequence dimension
- Hybrid combinations: EP×TP, EP×TP×CP
"""

from typing import Optional
from .base import Layer, Phase, DataType


class MoELayer(Layer):
    """
    Mixture of Experts layer with top-k routing.
    
    Architecture:
        1. Router: Linear projection to expert scores, then top-k selection
           router_logits = x @ W_router  # [M, E]
           weights, indices = top_k(softmax(router_logits), k)
        
        2. Expert FFN: Each expert is a standard or gated MLP
           For each token, compute weighted sum of selected expert outputs
        
        3. Optional shared experts: Always-active experts (DeepSeek-style)
    
    Parameters:
        Router: W_router[d, E] (+ bias if has_bias)
        Per expert (2-proj): W1[d, d_ff], W2[d_ff, d]
        Per expert (3-proj/gated): W_gate[d, d_ff], W_up[d, d_ff], W_down[d_ff, d]
    
    Supported parallelism:
        - expert_parallel (EP): Shard experts across chips, all-to-all communication
        - tensor_parallel (TP): Shard each expert's FFN, all-reduce communication
        - context_parallel (CP): Shard sequence, reduces activation memory
    """
    
    SUPPORTED_PARALLELISM = {"expert_parallel", "tensor_parallel", "context_parallel"}
    default_kernel_count = 5  # Router (1) + scatter/gather (2) + FFN operations (2)
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        dtype: DataType | str,
        num_shared_experts: int = 0,
        num_projections: int = 2,
        has_bias: bool = False,
        parallelism: Optional[dict] = None
    ):
        """
        Args:
            layer_idx: Layer index in the model
            hidden_size: Model hidden dimension (d)
            intermediate_size: Expert FFN intermediate dimension (d_ff)
            num_experts: Total number of routed experts (E)
            top_k: Number of experts selected per token (k)
            num_shared_experts: Number of always-active shared experts (default: 0)
            num_projections: 2 for standard FFN, 3 for gated/SwiGLU (default: 2)
            has_bias: Whether projections include bias terms (default: False)
            dtype: Numerical precision
            parallelism: Parallelism config dict
        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.num_projections = num_projections
        self.has_bias = has_bias
        
        # Validate
        if num_projections not in (2, 3):
            raise ValueError(f"num_projections must be 2 or 3, got {num_projections}")
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")
        
        # Calculate parameter counts
        d, d_ff, E = hidden_size, intermediate_size, num_experts
        E_shared = num_shared_experts
        
        # Router params
        self.router_params = d * E
        if has_bias:
            self.router_params += E
        
        # Expert params (per expert)
        params_per_expert = num_projections * d * d_ff
        if has_bias:
            # Bias for each projection: d_ff for up/gate, d for down
            if num_projections == 2:
                params_per_expert += d_ff + d
            else:  # 3 projections
                params_per_expert += d_ff + d_ff + d
        
        # Total expert params
        self.expert_params = E * params_per_expert
        self.shared_expert_params = E_shared * params_per_expert
        
        # Total param count
        self.param_count = self.router_params + self.expert_params + self.shared_expert_params
        
        super().__init__(layer_idx, dtype, parallelism)
    
    def _validate_parallelism(self) -> None:
        """Validate MoE parallelism constraints."""
        super()._validate_parallelism()
        
        ep = self.parallelism.get("expert_parallel", 1)
        tp = self.parallelism.get("tensor_parallel", 1)
        
        # EP must evenly divide num_experts
        if self.num_experts % ep != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"expert_parallel ({ep})"
            )
        
        # TP must evenly divide intermediate_size
        if self.intermediate_size % tp != 0:
            raise ValueError(
                f"intermediate_size ({self.intermediate_size}) must be divisible by "
                f"tensor_parallel ({tp})"
            )
    
    def _get_num_packages(self) -> int:
        """Total chips = EP × TP × CP."""
        ep = self.parallelism.get("expert_parallel", 1)
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        return ep * tp * cp
    
    def compute_flops(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        Compute FLOPs per chip for MoE layer.
        
        Router: 2 * M * d * E (replicated across all chips)
        Routed experts: 2 * num_proj * k * M * d * d_ff / (ep * tp)
        Shared experts: 2 * num_proj * E_shared * M * d * d_ff / tp
        
        Phase:
            Prefill: process full seq_len tokens
            Decode: process 1 token
        """
        ep = self.parallelism.get("expert_parallel", 1)
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        
        d = self.hidden_size
        d_ff = self.intermediate_size
        E = self.num_experts
        k = self.top_k
        E_shared = self.num_shared_experts
        num_proj = self.num_projections
        
        B = batch_size
        S = 1 if phase == Phase.DECODE else seq_len
        S_local = S // cp  # Local tokens with CP
        M = B * S_local  # Total local tokens
        
        flops = 0
        
        # Router (replicated on all chips)
        router_flops = 2 * M * d * E
        flops += router_flops
        
        # Routed expert computation (sharded by EP and TP)
        # Total routed FLOPs = 2 * num_proj * k * M * d * d_ff
        # Split across ep * tp chips
        routed_flops_total = 2 * num_proj * k * B * S_local * d * d_ff
        routed_flops_per_chip = routed_flops_total // (ep * tp)
        flops += routed_flops_per_chip
        
        # Shared expert computation (sharded across all chips: EP × TP)
        # Shared experts process ALL tokens, not just k per token
        if E_shared > 0:
            shared_flops_total = 2 * num_proj * E_shared * B * S_local * d * d_ff
            shared_flops_per_chip = shared_flops_total // (ep * tp)
            flops += shared_flops_per_chip
        
        return int(flops)
    
    def compute_weight_memory(self) -> int:
        """
        Compute weight memory per chip in bytes (total stored).
        
        Router: replicated on all chips
        Routed experts: sharded by EP, each expert TP-sharded
        Shared experts: replicated across EP, TP-sharded
        """
        ep = self.parallelism.get("expert_parallel", 1)
        tp = self.parallelism.get("tensor_parallel", 1)
        
        bytes_per_elem = self.dtype.bytes_per_element
        
        # Router (replicated)
        router_bytes = self.router_params * bytes_per_elem
        
        # Routed expert weights (EP + TP sharded)
        expert_bytes_total = self.expert_params * bytes_per_elem
        expert_bytes_per_chip = expert_bytes_total // (ep * tp)
        
        # Shared expert weights (TP sharded only, replicated across EP)
        shared_bytes_total = self.shared_expert_params * bytes_per_elem
        shared_bytes_per_chip = shared_bytes_total // tp
        
        return int(router_bytes + expert_bytes_per_chip + shared_bytes_per_chip)
    
    def _expected_unique_experts(self, batch_size: int, seq_len: int) -> float:
        """
        Expected number of unique experts activated across a batch.
        
        Uses coupon-collector distribution: each token independently selects
        top_k experts. The probability that a specific expert is NOT selected
        by any token in the batch:
            P(not selected) = (1 - top_k/num_experts)^num_tokens
        
        Expected unique experts = num_experts × (1 - P(not selected))
        
        Args:
            batch_size: Number of sequences
            seq_len: Tokens per sequence (1 for decode, full length for prefill)
            
        Returns:
            Expected number of unique experts activated
        """
        num_tokens = batch_size * seq_len
        p_not_selected = (1 - self.top_k / self.num_experts) ** num_tokens
        return self.num_experts * (1 - p_not_selected)
    
    def compute_weight_memory_read(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        Compute weight memory actually READ during a forward pass.
        
        For MoE, only active experts are read. With batching, unique experts
        activated follows a coupon-collector distribution. Small batches read
        few experts; large batches approach reading all experts.
        
        Behavior for 128 experts, top_k=4:
            Batch=1:   ~4 experts (3.1%)
            Batch=4:   ~15 experts (12%)
            Batch=64:  ~108 experts (84%)
            Batch=128: ~126 experts (98%)
        
        Args:
            batch_size: Number of sequences
            seq_len: Tokens per sequence
            phase: PREFILL or DECODE
            
        Returns:
            Memory in bytes actually read from HBM
        """
        ep = self.parallelism.get("expert_parallel", 1)
        tp = self.parallelism.get("tensor_parallel", 1)
        bytes_per_elem = self.dtype.bytes_per_element
        
        # Router is always fully read
        router_bytes = self.router_params * bytes_per_elem
        
        # Shared experts are always fully read (TP sharded)
        shared_bytes_total = self.shared_expert_params * bytes_per_elem
        shared_bytes_read = shared_bytes_total // tp
        
        # Routed experts: probabilistic based on batch
        # For decode, seq_len=1 (one token per sequence)
        # For prefill, seq_len=full prompt length
        tokens_this_step = batch_size * seq_len
        
        # Expected unique experts activated globally
        unique_global = self._expected_unique_experts(batch_size, seq_len)
        
        # Experts per chip (EP sharding)
        experts_per_chip = self.num_experts // ep
        
        # Approximate unique experts on this chip (proportional to EP)
        unique_on_chip = unique_global / ep
        
        # Clamp to valid range [top_k/ep, experts_per_chip]
        unique_on_chip = min(max(unique_on_chip, self.top_k / ep), experts_per_chip)
        
        # Bytes per expert (TP sharded)
        params_per_expert = self.num_projections * self.hidden_size * self.intermediate_size
        if self.has_bias:
            params_per_expert += self.num_projections * self.intermediate_size
        bytes_per_expert = params_per_expert * bytes_per_elem // tp
        
        # Total routed expert bytes read
        routed_bytes_read = unique_on_chip * bytes_per_expert
        
        return int(router_bytes + shared_bytes_read + routed_bytes_read)
    
    def compute_activation_memory(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        Compute activation memory per chip in bytes.
        
        Resident tensors:
        - Input x: M * d
        - Router logits: M * E
        - Expert intermediate (peak, processing one batch at a time): M * d_ff / tp
          Note: With EP, each chip only processes M/ep tokens through local experts,
          but the intermediate buffer is sized for the local token count
        - Output y: M * d
        
        Phase:
            Prefill: process full seq_len tokens (CP shards sequence)
            Decode: process 1 token
        """
        ep = self.parallelism.get("expert_parallel", 1)
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        
        d = self.hidden_size
        d_ff = self.intermediate_size
        E = self.num_experts
        
        B = batch_size
        S = 1 if phase == Phase.DECODE else seq_len
        S_local = S // cp
        M = B * S_local
        
        bytes_per_elem = self.dtype.bytes_per_element
        
        elems = 0
        
        # Input x (replicated across EP, not sharded by TP for MoE input)
        elems += M * d
        
        # Router logits (replicated)
        elems += M * E
        
        # Expert intermediate (peak, one expert batch at a time)
        # With EP, each chip processes (k * M / ep) token-expert pairs
        # With TP, d_ff is sharded
        # Formula: (k * M / ep) × (d_ff / tp)
        d_ff_local = d_ff // tp
        E_shared = self.num_shared_experts
        
        if ep > 1:
            # With EP, only a fraction of token-expert pairs are local
            # Routed expert intermediate = (k * M / ep) * d_ff_local
            token_expert_pairs_local = (self.top_k * M) // ep
            elems += token_expert_pairs_local * d_ff_local
            
            # Shared expert intermediate (if any)
            # Shared experts process ALL M tokens, sharded by TP
            if E_shared > 0:
                shared_tokens_local = M // tp
                elems += shared_tokens_local * d_ff_local
        else:
            # Without EP, peak is processing all M tokens through one expert at a time
            elems += M * d_ff_local
            
            # Shared expert intermediate (no EP sharding)
            if E_shared > 0:
                shared_tokens_local = M // tp if tp > 1 else M
                elems += shared_tokens_local * d_ff_local
        
        # Output y
        elems += M * d
        
        return int(elems * bytes_per_elem)
    
    def compute_kv_cache(self, batch_size: int, seq_len: int) -> int:
        """MoE is an FFN layer, no KV cache."""
        return 0
    
    def _compute_communication_bytes(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase,
        hardware: dict
    ) -> int:
        """
        Compute communication bytes per chip for MoE.

        EP communication: All-to-all dispatch (tokens to experts) and combine (outputs back)
        TP communication: All-reduce on expert outputs
        CP communication: May require gather/scatter for routing

        Simplified model:
        - EP all-to-all: 2 * M * d * bytes (dispatch + combine)
        - TP all-reduce (routed): (k * M / ep) * d * bytes (local token-expert outputs)
        - TP all-reduce (shared): (M / ep) * d * bytes (if shared experts exist)
        """
        ep = self.parallelism.get("expert_parallel", 1)
        tp = self.parallelism.get("tensor_parallel", 1)
        cp = self.parallelism.get("context_parallel", 1)
        
        d = self.hidden_size
        k = self.top_k
        
        B = batch_size
        S = 1 if phase == Phase.DECODE else seq_len
        S_local = S // cp
        M = B * S_local
        
        bytes_per_elem = self.dtype.bytes_per_element
        
        comm_bytes = 0
        
        # EP all-to-all (dispatch + combine)
        if ep > 1:
            # Dispatch: send tokens to correct expert chips
            # Combine: return expert outputs
            ep_comm = 2 * M * d * bytes_per_elem
            comm_bytes += int(ep_comm)
        
        # TP all-reduce on expert outputs
        if tp > 1:
            # Each token activates k experts, each expert output needs all-reduce
            # With EP, experts are distributed so each chip handles (k * M / ep) pairs
            # All-reduce payload = local token-expert pairs × d
            routed_pairs_local = (k * M) // ep
            tp_comm = routed_pairs_local * d * bytes_per_elem

            # Shared experts also need TP all-reduce (M/ep tokens per chip)
            if self.num_shared_experts > 0:
                shared_pairs_local = M // ep
                tp_comm += shared_pairs_local * d * bytes_per_elem

            comm_bytes += int(tp_comm)
        
        return int(comm_bytes) if comm_bytes > 0 else None
