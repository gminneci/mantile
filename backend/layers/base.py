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
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    NVFP8 = "nvfp8"
    NVFP4 = "nvfp4"
    INT8 = "int8"
    
    @property
    def bytes_per_element(self) -> float:
        """Return bytes per element for this data type"""
        return {
            "fp32": 4.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "fp8": 1.0,
            "mxfp8": 8.25/8,
            "nvfp8": 8.25/8,
            "mxfp4": 4.25/8,
            "nvfp4": 4.5/8,
            "int8": 1.0,
        }[self.value]


@dataclass
class LayerMetrics:
    """
    Output metrics from a layer computation.
    Includes both per-package metrics (what each package does) and aggregate metrics (total across all packages).
    
    Per-package metrics (what ONE package computes/stores):
        - flops_per_package: FLOPs executed on one package
        - weight_memory_per_package: Parameter memory on one package
        - activation_memory_per_package: Activation memory on one package
        - kv_cache_per_package: KV cache on one package (attention only)
    
    Aggregate metrics (total across all packages):
        - flops_total: Total FLOPs across all packages (= flops_per_package * num_packages)
        - weight_memory_total: Total parameters across all packages
        - activation_memory_total: Total activations across all packages
        - kv_cache_total: Total KV cache across all packages
    
    Hardware-dependent (None if hardware not provided):
        - compute_time_ms: Time to execute (accounts for parallelism)
        - weight_load_time_ms: Time to load weights from memory
        - communication_bytes: Inter-package data transfer size
        - communication_time_ms: Time for inter-package communication
    
    Derived/simulation metrics (computed from execution model):
        - memory_bandwidth_per_package_GBps: Peak DRAM bandwidth requirement per package
        - kv_cache_bandwidth_per_package_GBps: Peak KV cache bandwidth requirement per package
        - communication_bandwidth_GBps: Inter-package bandwidth requirement
        - wall_clock_time_ms: Actual execution time with overlaps considered
        - compute_communication_overlap: Whether compute/communication can overlap
    
    Parallelism info:
        - num_packages: Number of packages this layer uses
    """
    # Per-package metrics
    flops_per_package: int
    weight_memory_per_package: int
    activation_memory_per_package: int
    kv_cache_per_package: int = 0
    
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
    memory_bandwidth_per_package_GBps: Optional[float] = None
    kv_cache_bandwidth_per_package_GBps: Optional[float] = None
    communication_bandwidth_GBps: Optional[float] = None
    wall_clock_time_ms: Optional[float] = None
    compute_communication_overlap: Optional[bool] = None
    
    # Parallelism
    num_packages: int = 1


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
        - _get_num_packages(): Return number of packages this layer uses
    
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
    def _get_num_packages(self) -> int:
        """
        Calculate number of packages this layer uses based on parallelism config.
        
        Returns:
            Number of packages (1 if no parallelism)
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
            LayerMetrics object with per-package and aggregate metrics
        """
        # Convert string inputs to enums
        if isinstance(phase, str):
            phase = Phase(phase)
        
        # Get number of packages for this layer
        num_packages = self._get_num_packages()
        
        # Compute per-package metrics (what ONE package does)
        flops_per_package = self.compute_flops(batch_size, seq_len, phase)
        weight_mem_per_package = self.compute_weight_memory()
        activation_mem_per_package = self.compute_activation_memory(batch_size, seq_len, phase)
        kv_cache_per_package = self.compute_kv_cache(batch_size, seq_len)
        
        # Communication bytes (if applicable based on parallelism strategy)
        # This is computed regardless of hardware config, as it's a property of the layer
        comm_bytes = self._compute_communication_bytes(batch_size, seq_len, phase, hardware or {})
        
        # Compute aggregate metrics (total across all packages)
        # Note: FLOPs might not scale linearly with num_packages (e.g., PP doesn't replicate work)
        flops_total = flops_per_package * num_packages  # Override in subclasses if needed
        weight_mem_total = weight_mem_per_package * num_packages
        activation_mem_total = activation_mem_per_package * num_packages
        kv_cache_total = kv_cache_per_package * num_packages
        
        # Compute hardware-dependent metrics if config provided
        compute_time = None
        weight_load_time = None
        comm_time = None
        
        # Derived metrics (initialized as None)
        memory_bw_per_package = None
        kv_cache_bw_per_package = None
        comm_bw = None
        wall_clock_time = None
        can_overlap = None
        
        if hardware:
            # Compute time: FLOPs / (peak_tflops * 10^12)
            # Per-package FLOPs since work is parallelized
            peak_flops = hardware.get("compute_tflops", 0) * 1e12
            if peak_flops > 0:
                compute_time = (flops_per_package / peak_flops) * 1000  # Convert to ms
            
            # Weight load time: bytes / bandwidth (per-package)
            mem_bw = hardware.get("memory_bandwidth_gb_s", 0) * 1e9
            if mem_bw > 0:
                weight_load_time = (weight_mem_per_package / mem_bw) * 1000  # Convert to ms
            
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
                # Memory bandwidth per package: (weights + activations) / time
                # Use the max of compute_time and weight_load_time for realistic estimate
                effective_time_ms = max(compute_time, weight_load_time) if weight_load_time else compute_time
                
                total_memory_bytes = weight_mem_per_package + activation_mem_per_package
                memory_bw_per_package = (total_memory_bytes / (effective_time_ms / 1000)) / 1e9  # GB/s
                
                # KV cache bandwidth (if applicable)
                if kv_cache_per_package > 0:
                    kv_cache_bw_per_package = (kv_cache_per_package / (effective_time_ms / 1000)) / 1e9  # GB/s
            
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
            flops_per_package=flops_per_package,
            weight_memory_per_package=weight_mem_per_package,
            activation_memory_per_package=activation_mem_per_package,
            kv_cache_per_package=kv_cache_per_package,
            flops_total=flops_total,
            weight_memory_total=weight_mem_total,
            activation_memory_total=activation_mem_total,
            kv_cache_total=kv_cache_total,
            compute_time_ms=compute_time,
            weight_load_time_ms=weight_load_time,
            communication_bytes=comm_bytes,
            communication_time_ms=comm_time,
            memory_bandwidth_per_package_GBps=memory_bw_per_package,
            kv_cache_bandwidth_per_package_GBps=kv_cache_bw_per_package,
            communication_bandwidth_GBps=comm_bw,
            wall_clock_time_ms=wall_clock_time,
            compute_communication_overlap=can_overlap,
            num_packages=num_packages
        )
    
    def _compute_communication_bytes(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase,
        hardware: dict
    ) -> Optional[int]:
        """
        Compute inter-package communication requirements.
        Default implementation returns None. Subclasses override for specific patterns.
        Uses self.dtype for precision.
        
        Args:
            batch_size: Number of sequences
            seq_len: Sequence length
            phase: PREFILL or DECODE
            hardware: Hardware config (includes num_packages, parallelism strategy, etc.)
            
        Returns:
            Bytes to communicate, or None if not applicable
        """
        return None
