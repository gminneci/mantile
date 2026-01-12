"""
Base classes for LLM layer modeling.

This module provides the foundational abstractions for modeling LLM layers:
- Phase: Inference phase enum (prefill/decode)
- DataType: Numerical precision formats
- LayerMetrics: Output dataclass for layer computations
- Layer: Abstract base class for all layer types
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Phase(Enum):
    """Inference phase: prefill (prompt processing) or decode (token generation)"""
    PREFILL = "prefill"
    DECODE = "decode"


class DataType(Enum):
    """Numerical precision format"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    
    @property
    def bytes_per_element(self) -> float:
        """Return bytes per element for this data type"""
        return {
            "fp32": 4.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "fp8": 1.0,
            "int8": 1.0,
        }[self.value]


@dataclass
class LayerMetrics:
    """
    Output metrics from a layer computation.
    Includes both per-chip metrics (what each chip does) and aggregate metrics (total across all chips).
    
    Per-chip metrics (what ONE chip computes/stores):
        - flops_per_chip: FLOPs executed on one chip
        - weight_memory_per_chip: Parameter memory on one chip
        - activation_memory_per_chip: Activation memory on one chip
        - kv_cache_per_chip: KV cache on one chip (attention only)
    
    Aggregate metrics (total across all chips):
        - flops_total: Total FLOPs across all chips (= flops_per_chip * num_chips)
        - weight_memory_total: Total parameters across all chips
        - activation_memory_total: Total activations across all chips
        - kv_cache_total: Total KV cache across all chips
    
    Hardware-dependent (None if hardware not provided):
        - compute_time_ms: Time to execute (accounts for parallelism)
        - weight_load_time_ms: Time to load weights from memory
        - communication_bytes: Inter-chip data transfer size
        - communication_time_ms: Time for inter-chip communication
    
    Derived/simulation metrics (computed from execution model):
        - memory_bandwidth_per_chip_GBps: Peak DRAM bandwidth requirement per chip
        - kv_cache_bandwidth_per_chip_GBps: Peak KV cache bandwidth requirement per chip
        - communication_bandwidth_GBps: Inter-chip bandwidth requirement
        - wall_clock_time_ms: Actual execution time with overlaps considered
        - compute_communication_overlap: Whether compute/communication can overlap
    
    Parallelism info:
        - num_chips: Number of chips this layer uses
    """
    # Per-chip metrics
    flops_per_chip: int
    weight_memory_per_chip: int
    activation_memory_per_chip: int
    kv_cache_per_chip: int = 0
    
    # Aggregate metrics
    flops_total: int = 0
    weight_memory_total: int = 0
    activation_memory_total: int = 0
    kv_cache_total: int = 0
    
    # Hardware-dependent (optional)
    compute_time_ms: Optional[float] = None
    weight_load_time_ms: Optional[float] = None
    communication_bytes: Optional[int] = None
    communication_time_ms: Optional[float] = None
    
    # Derived/simulation metrics (optional, require execution model)
    memory_bandwidth_per_chip_GBps: Optional[float] = None
    kv_cache_bandwidth_per_chip_GBps: Optional[float] = None
    communication_bandwidth_GBps: Optional[float] = None
    wall_clock_time_ms: Optional[float] = None
    compute_communication_overlap: Optional[bool] = None
    
    # Parallelism
    num_chips: int = 1


class Layer(ABC):
    """
    Abstract base class for all LLM layer types.
    
    Each layer is "compiled" to a specific parallelism configuration at instantiation.
    Changing the sharding strategy requires creating a new layer instance.
    
    Subclasses must implement:
        - compute_flops(): Calculate FLOPs for given batch/seq_len/phase
        - compute_weight_memory(): Calculate parameter memory footprint
        - compute_activation_memory(): Calculate activation memory footprint
        - compute_kv_cache(): Calculate KV cache size (if applicable)
        - _get_num_chips(): Return number of chips this layer uses
    
    Subclasses should define:
        - SUPPORTED_PARALLELISM: Set of supported parallelism types (e.g., {"tensor_parallel", "pipeline_parallel"})
        - Can override _validate_parallelism() for custom validation logic
    """
    
    # Subclasses override this to declare supported parallelism types
    SUPPORTED_PARALLELISM: set[str] = set()

    @classmethod
    def get_supported_parallelism(cls) -> list[str]:
        """
        Return the supported parallelism types for this layer class.
        Provided as a classmethod for easy introspection without instantiation.
        """
        return sorted(list(cls.SUPPORTED_PARALLELISM))
    
    def __init__(self, layer_idx: int, dtype: DataType | str, parallelism: Optional[dict] = None):
        """
        Args:
            layer_idx: Layer index in the model (0-indexed)
            dtype: Numerical precision (DataType enum or string like 'bf16')
            parallelism: Parallelism configuration dict, e.g.:
                {"tensor_parallel": 2} - 2-way TP
                {"pipeline_parallel": {"stage": 0, "num_stages": 4}} - PP stage 0/4
                {"expert_parallel": 4, "tensor_parallel": 2} - Mixed EP+TP
                None - Single chip, no sharding
        """
        self.layer_idx = layer_idx
        self.dtype = DataType(dtype) if isinstance(dtype, str) else dtype
        self.parallelism = parallelism or {}
        
        # Validate that this layer supports the requested parallelism
        self._validate_parallelism()
    
    def _validate_parallelism(self) -> None:
        """
        Validate that the parallelism config is supported for this layer type.
        Default implementation checks against SUPPORTED_PARALLELISM class attribute.
        Subclasses can override for custom validation logic.
        
        Raises:
            ValueError: If unsupported parallelism type is requested
        """
        unsupported = set(self.parallelism.keys()) - self.SUPPORTED_PARALLELISM
        if unsupported:
            raise ValueError(
                f"{self.__class__.__name__} does not support parallelism types: {unsupported}. "
                f"Supported types: {self.SUPPORTED_PARALLELISM or 'none'}"
            )
    
    @abstractmethod
    def _get_num_chips(self) -> int:
        """
        Calculate number of chips this layer uses based on parallelism config.
        
        Returns:
            Number of chips (1 if no parallelism)
        """
        pass
    
    @abstractmethod
    def compute_flops(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase
    ) -> int:
        """
        Compute total FLOPs for this layer.
        
        Args:
            batch_size: Number of sequences in batch
            seq_len: Sequence length (prompt length for prefill, 1 for decode)
            phase: PREFILL or DECODE
            
        Returns:
            Total floating point operations
        """
        pass
    
    @abstractmethod
    def compute_weight_memory(self) -> int:
        """
        Compute memory required to store layer parameters.
        Uses self.dtype for precision.
        
        Returns:
            Memory in bytes
        """
        pass
    
    @abstractmethod
    def compute_activation_memory(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase
    ) -> int:
        """
        Compute memory required for activations during forward pass.
        Uses self.dtype for precision.
        
        Args:
            batch_size: Number of sequences in batch
            seq_len: Sequence length
            phase: PREFILL or DECODE
            
        Returns:
            Memory in bytes
        """
        pass
    
    def compute_kv_cache(
        self,
        batch_size: int,
        seq_len: int
    ) -> int:
        """
        Compute KV cache size (relevant for attention layers only).
        Uses self.dtype for precision.
        
        Args:
            batch_size: Number of sequences in batch
            seq_len: Total sequence length to cache
            
        Returns:
            Memory in bytes (0 for non-attention layers)
        """
        return 0
    
    def compute_metrics(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase | str,
        hardware: Optional[dict] = None
    ) -> LayerMetrics:
        """
        Compute all metrics for this layer (both per-chip and aggregate).
        Uses self.dtype for numerical precision.
        
        Args:
            batch_size: Number of sequences in batch
            seq_len: Sequence length
            phase: PREFILL or DECODE (or string)
            hardware: Optional hardware config dict with:
                - compute_tflops: Peak compute throughput per chip (TFLOPS)
                - memory_bandwidth_gb_s: Memory bandwidth per chip (GB/s)
                - interconnect_bandwidth_gb_s: Inter-chip bandwidth (GB/s)
                - interconnect_latency_us: Inter-chip latency (microseconds)
                
        Returns:
            LayerMetrics object with per-chip and aggregate metrics
        """
        # Convert string inputs to enums
        if isinstance(phase, str):
            phase = Phase(phase)
        
        # Get number of chips for this layer
        num_chips = self._get_num_chips()
        
        # Compute per-chip metrics (what ONE chip does)
        flops_per_chip = self.compute_flops(batch_size, seq_len, phase)
        weight_mem_per_chip = self.compute_weight_memory()
        activation_mem_per_chip = self.compute_activation_memory(batch_size, seq_len, phase)
        kv_cache_per_chip = self.compute_kv_cache(batch_size, seq_len)
        
        # Communication bytes (if applicable based on parallelism strategy)
        # This is computed regardless of hardware config, as it's a property of the layer
        comm_bytes = self._compute_communication_bytes(batch_size, seq_len, phase, hardware or {})
        
        # Compute aggregate metrics (total across all chips)
        # Note: FLOPs might not scale linearly with num_chips (e.g., PP doesn't replicate work)
        flops_total = flops_per_chip * num_chips  # Override in subclasses if needed
        weight_mem_total = weight_mem_per_chip * num_chips
        activation_mem_total = activation_mem_per_chip * num_chips
        kv_cache_total = kv_cache_per_chip * num_chips
        
        # Compute hardware-dependent metrics if config provided
        compute_time = None
        weight_load_time = None
        comm_time = None
        
        # Derived metrics (initialized as None)
        memory_bw_per_chip = None
        kv_cache_bw_per_chip = None
        comm_bw = None
        wall_clock_time = None
        can_overlap = None
        
        if hardware:
            # Compute time: FLOPs / (peak_tflops * 10^12)
            # Per-chip FLOPs since work is parallelized
            peak_flops = hardware.get("compute_tflops", 0) * 1e12
            if peak_flops > 0:
                compute_time = (flops_per_chip / peak_flops) * 1000  # Convert to ms
            
            # Weight load time: bytes / bandwidth (per-chip)
            mem_bw = hardware.get("memory_bandwidth_gb_s", 0) * 1e9
            if mem_bw > 0:
                weight_load_time = (weight_mem_per_chip / mem_bw) * 1000  # Convert to ms
            
            # Communication time (if communication bytes were computed)
            if comm_bytes is not None:
                interconnect_bw = hardware.get("interconnect_bandwidth_gb_s", 0) * 1e9
                interconnect_latency_us = hardware.get("interconnect_latency_us", 0)
                
                if interconnect_bw > 0:
                    # Time = latency + (bytes / bandwidth)
                    transfer_time_ms = (comm_bytes / interconnect_bw) * 1000
                    latency_ms = interconnect_latency_us / 1000
                    comm_time = latency_ms + transfer_time_ms
            
            # Compute derived/simulation metrics
            if compute_time is not None and compute_time > 0:
                # Memory bandwidth per chip: (weights + activations) / time
                # Use the max of compute_time and weight_load_time for realistic estimate
                effective_time_ms = max(compute_time, weight_load_time) if weight_load_time else compute_time
                
                total_memory_bytes = weight_mem_per_chip + activation_mem_per_chip
                memory_bw_per_chip = (total_memory_bytes / (effective_time_ms / 1000)) / 1e9  # GB/s
                
                # KV cache bandwidth (if applicable)
                if kv_cache_per_chip > 0:
                    kv_cache_bw_per_chip = (kv_cache_per_chip / (effective_time_ms / 1000)) / 1e9  # GB/s
            
            # Communication bandwidth (if applicable)
            if comm_bytes is not None and comm_time is not None and comm_time > 0:
                comm_bw = (comm_bytes / (comm_time / 1000)) / 1e9  # GB/s
            
            # Wall clock time: consider compute/communication overlap
            # Determine if overlap is possible (e.g., for TP all-reduce in row-parallel layers)
            can_overlap = hardware.get("supports_overlap", False)
            
            if compute_time is not None and comm_time is not None:
                if can_overlap:
                    # Overlapped: wall clock is max of compute and communication
                    wall_clock_time = max(compute_time, comm_time)
                else:
                    # Sequential: wall clock is sum
                    wall_clock_time = compute_time + comm_time
            elif compute_time is not None:
                wall_clock_time = compute_time
            elif comm_time is not None:
                wall_clock_time = comm_time
        
        return LayerMetrics(
            flops_per_chip=flops_per_chip,
            weight_memory_per_chip=weight_mem_per_chip,
            activation_memory_per_chip=activation_mem_per_chip,
            kv_cache_per_chip=kv_cache_per_chip,
            flops_total=flops_total,
            weight_memory_total=weight_mem_total,
            activation_memory_total=activation_mem_total,
            kv_cache_total=kv_cache_total,
            compute_time_ms=compute_time,
            weight_load_time_ms=weight_load_time,
            communication_bytes=comm_bytes,
            communication_time_ms=comm_time,
            memory_bandwidth_per_chip_GBps=memory_bw_per_chip,
            kv_cache_bandwidth_per_chip_GBps=kv_cache_bw_per_chip,
            communication_bandwidth_GBps=comm_bw,
            wall_clock_time_ms=wall_clock_time,
            compute_communication_overlap=can_overlap,
            num_chips=num_chips
        )
    
    def _compute_communication_bytes(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase,
        hardware: dict
    ) -> Optional[int]:
        """
        Compute inter-chip communication requirements.
        Default implementation returns None. Subclasses override for specific patterns.
        Uses self.dtype for precision.
        
        Args:
            batch_size: Number of sequences
            seq_len: Sequence length
            phase: PREFILL or DECODE
            hardware: Hardware config (includes num_chips, parallelism strategy, etc.)
            
        Returns:
            Bytes to communicate, or None if not applicable
        """
        return None
