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
from typing import Optional, Dict, Any
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
        """
        Return bytes per element for this data type.
        
        Block-scaled formats (MX/NV-FP) include overhead for scaling factors:
        - MXFP8/NVFP8: 8 bits/element + 1 FP8 scale per 32 elements = (8*32 + 8)/32 = 8.25 bits = 8.25/8 bytes
        - MXFP4: 4 bits/element + 1 FP8 scale per 32 elements = (4*32 + 8)/32 = 4.25 bits = 4.25/8 bytes
        - NVFP4: 4 bits/element + 1 FP16 scale per 32 elements = (4*32 + 16)/32 = 4.5 bits = 4.5/8 bytes
        """
        return {
            "fp32": 4.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "fp8": 1.0,
            "mxfp8": 8.25/8,   # 8-bit mantissa + shared FP8 scale per 32-element block
            "nvfp8": 8.25/8,   # 8-bit mantissa + shared FP8 scale per 32-element block
            "mxfp4": 4.25/8,   # 4-bit mantissa + shared FP8 scale per 32-element block
            "nvfp4": 4.5/8,    # 4-bit mantissa + shared FP16 scale per 32-element block
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
        - load_time_ms: Time to load from memory (weights + KV cache during decode)
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
    
    # Hardware-dependent (0.0 if hardware not provided)
    compute_time_ms: float = 0.0
    load_time_ms: float = 0.0
    communication_bytes: int = 0
    communication_time_ms: float = 0.0
    
    # Derived/simulation metrics (0.0 if not computed)
    memory_bandwidth_per_package_GBps: float = 0.0
    kv_cache_bandwidth_per_package_GBps: float = 0.0
    communication_bandwidth_GBps: float = 0.0
    wall_clock_time_ms: float = 0.0
    bottleneck: str = "unknown"  # "compute", "memory", "balanced", or "unknown"
    
    # Parallelism
    num_packages: int = 1
    
    # Debug details (populated when debug=True)
    debug_details: Optional[Dict[str, Any]] = None


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
    
    # TODO: Make kernel count more sophisticated - account for:
    # - Different fusion strategies (e.g., flash attention vs naive)
    # - Hardware-specific fusion (some GPUs fuse better than others)
    # - Batch size effects (some kernels only launch once regardless of batch)
    # - Dynamic fusion decisions based on input shapes
    # For now, use a conservative static estimate per layer type
    default_kernel_count: int = 0

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
    
    def compute_weight_memory_read(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase
    ) -> int:
        """
        Compute weight memory actually READ during a forward pass.
        
        For most layers, this equals compute_weight_memory() since all weights
        are read. For MoE layers, only active experts are read, which depends
        on batch size and the probabilistic activation pattern.
        
        Args:
            batch_size: Number of sequences in batch
            seq_len: Sequence length
            phase: PREFILL or DECODE
            
        Returns:
            Memory in bytes actually read from HBM
        """
        # Default: read all weights (dense layers)
        return self.compute_weight_memory()
    
    def compute_metrics(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase | str,
        hardware: Optional[dict] = None,
        context_len: Optional[int] = None,
        debug: bool = False
    ) -> LayerMetrics:
        """
        Compute all metrics for this layer (both per-chip and aggregate).
        Uses self.dtype for numerical precision.
        
        Args:
            batch_size: Number of sequences in batch
            seq_len: Sequence length being processed (1 for decode)
            phase: PREFILL or DECODE (or string)
            hardware: Optional full hardware config dict from JSON file with:
                - compute_per_package_PFlops: Dict of peak compute per dtype (PFLOPs)
                - memory_per_package: List of memory configs (extracts HBM)
                - interconnect_bandwidth_GBs: Inter-chip bandwidth (GB/s)
                - interconnect_latency_us: Inter-chip latency (microseconds)
                - decode_load_overlap: Whether decode can overlap compute and memory
                - decode_comms_overlap: Whether decode can overlap compute and comms
                - fixed_overhead_per_kernel_us: Fixed overhead per kernel (microseconds)
            context_len: Total context length in KV cache (for decode phase)
            debug: If True, populate debug_details with all intermediate calculations
                
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
        
        # KV cache: use context_len for decode (full context), seq_len for prefill
        kv_seq_len = context_len if (phase == Phase.DECODE and context_len is not None) else seq_len
        kv_cache_per_package = self.compute_kv_cache(batch_size, kv_seq_len)
        
        # Communication bytes (if applicable based on parallelism strategy)
        # This is computed regardless of hardware config, as it's a property of the layer
        comm_bytes = self._compute_communication_bytes(batch_size, seq_len, phase, hardware or {})
        if comm_bytes is None:
            comm_bytes = 0
        
        # Compute aggregate metrics (total across all packages)
        # Note: FLOPs might not scale linearly with num_packages (e.g., PP doesn't replicate work)
        flops_total = flops_per_package * num_packages  # Override in subclasses if needed
        weight_mem_total = weight_mem_per_package * num_packages
        activation_mem_total = activation_mem_per_package * num_packages
        kv_cache_total = kv_cache_per_package * num_packages
        
        # Compute hardware-dependent metrics (assume hardware config always provided)
        # All time values are floats >= 0
        compute_time = 0.0
        load_time = 0.0
        comm_time = 0.0
        memory_bw_per_package = 0.0
        kv_cache_bw_per_package = 0.0
        comm_bw = 0.0
        wall_clock_time = 0.0
        
        if hardware:
            # TODO: Future support for complex memory hierarchies (e.g., TPUs)
            # Will need to specify where weights live vs where KV cache lives
            # For now, default to HBM for all memory operations
            hbm_memory = next(
                (m for m in hardware['memory_per_package'] if 'HBM' in m['type']),
                hardware['memory_per_package'][0]
            )
            mem_bw_gbs = hbm_memory['bandwidth_GBs']
            
            # Extract overlap and communication settings for clarity
            decode_load_overlap = hardware.get("decode_load_overlap", False)
            decode_comms_overlap = hardware.get("decode_comms_overlap", False)
            interconnect_bw_gbs = hardware["interconnect_bandwidth_GBs"]
            interconnect_latency_us = hardware["interconnect_latency_us"]
            
            # Compute time: FLOPs / peak_flops
            peak_pflops = hardware["compute_per_package_PFlops"][self.dtype.value]
            peak_flops = peak_pflops * 1e15  # Convert PFLOPs to FLOPs
            compute_time = (flops_per_package / peak_flops) * 1000  # Convert to ms
            
            # Load time: memory bandwidth for reading weights + KV cache from HBM
            mem_bw = mem_bw_gbs * 1e9  # Convert to bytes/s
            weight_bytes_read = self.compute_weight_memory_read(batch_size, seq_len, phase)
            
            load_bytes = weight_bytes_read
            if phase == Phase.DECODE:
                # Decode: read weights + KV cache from HBM to compute units
                load_bytes += kv_cache_per_package
            else:
                # Prefill: read weights + write KV cache
                load_bytes += kv_cache_per_package
                
            load_time = (load_bytes / mem_bw) * 1000  # Convert to ms
            
            # Communication time (always computed, 0 if no communication)
            if comm_bytes > 0:
                interconnect_bw = interconnect_bw_gbs * 1e9
                # Time = latency + (bytes / bandwidth)
                transfer_time_ms = (comm_bytes / interconnect_bw) * 1000
                latency_ms = interconnect_latency_us / 1000
                comm_time = latency_ms + transfer_time_ms
                        
            if phase == Phase.PREFILL:
                # Prefill: full overlap assumed
                wall_clock_time = max(compute_time, load_time, comm_time)
                
            else:  # DECODE
                wall_clock_time = compute_time
                # Decode: memory-bound, overlap depends on hardware
                if hardware.get("decode_load_overlap", False):
                    # Some hardware can overlap (e.g., with aggressive prefetching)
                    wall_clock_time = max(wall_clock_time, load_time)
                else:
                    # Most hardware cannot overlap in decode (streaming KV cache reads dominate)
                    wall_clock_time = wall_clock_time + load_time
                
                # Communication overlap in decode
                if hardware.get("decode_comms_overlap", False):
                    # Advanced interconnects (e.g., TPU) can overlap comms
                    wall_clock_time = max(wall_clock_time, comm_time)
                else:
                    # Most systems: communication is sequential
                    wall_clock_time = wall_clock_time + comm_time
            
            # Add fixed overhead based on kernel count (kernel launch, scheduling, etc.)
            fixed_overhead_per_kernel_us = hardware.get("fixed_overhead_per_kernel_us", 0.0)
            layer_kernel_overhead_us = self.default_kernel_count * fixed_overhead_per_kernel_us / 1000  # Convert us to ms
            wall_clock_time += layer_kernel_overhead_us
            
            # Compute derived bandwidth metrics using wall_clock_time as the effective execution time
            if wall_clock_time > 0:
                # Memory bandwidth: (weights + activations) / time
                total_memory_bytes = weight_mem_per_package + activation_mem_per_package
                memory_bw_per_package = (total_memory_bytes / (wall_clock_time / 1000)) / 1e9  # GB/s
                
                # KV cache bandwidth
                if kv_cache_per_package > 0:
                    kv_cache_bw_per_package = (kv_cache_per_package / (wall_clock_time / 1000)) / 1e9  # GB/s
                
                # Communication bandwidth
                if comm_time > 0:
                    comm_bw = (comm_bytes / (comm_time / 1000)) / 1e9  # GB/s
        
        # Determine bottleneck (only if hardware provided)
        bottleneck = "unknown"
        if hardware:
            if compute_time > load_time * 1.2:
                bottleneck = "compute"
            elif load_time > compute_time * 1.2:
                bottleneck = "memory"
            else:
                bottleneck = "balanced"
        
        # Build debug info if requested
        debug_info = None
        prompt = """
        You are an expert ML systems engineer verifying a performance model for LLM inference.

        TASK
        Independently derive each value in 'computed_metrics' and 'timing_breakdown' 
        using standard formulas for transformer inference. Compare your calculations 
        against the reported values.

        AVAILABLE DATA
        - 'inputs': workload parameters (batch size, sequence length, phase, etc.)
        - 'hardware_params': hardware specifications (bandwidth, FLOPS, latencies)
        - 'intermediate_calculations': pre-computed values you should also verify

        METHODOLOGY
        1. State any assumptions you make (e.g., overlap policy, what bytes are loaded)
        2. Show step-by-step calculations with units
        3. Compare each derived value to the reported value
        4. Flag discrepancies with severity: [CRITICAL] >5%, [WARNING] 1-5%, [INFO] <1%

        IF INFORMATION IS MISSING
        Do not guess. State explicitly what is needed and why.

        OUTPUT FORMAT
        - Summary table: [metric | derived | reported | Δ% | status]
        - List of assumptions made
        - List of missing information (if any)
        - Discrepancies with explanations
        """
        if debug:
            debug_info = {
                "prompt": prompt,
                "inputs": {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "phase": phase.value,
                    "context_len": context_len,
                    "dtype": self.dtype.value,
                    "dtype_bytes": self.dtype.bytes_per_element,
                    "layer_type": self.__class__.__name__,
                    "parallelism": self.parallelism,
                    "num_packages": num_packages,
                    "default_kernel_count": self.default_kernel_count,
                },
                "intermediate_calculations": {
                    "flops_per_package": flops_per_package,
                    "weight_mem_per_package": weight_mem_per_package,
                    "activation_mem_per_package": activation_mem_per_package,
                    "kv_cache_per_package": kv_cache_per_package,
                    "flops_total": flops_total,
                    "weight_mem_total": weight_mem_total,
                    "activation_mem_total": activation_mem_total,
                    "kv_cache_total": kv_cache_total,
                    "comm_bytes": comm_bytes,
                    "weight_bytes_read": weight_bytes_read if hardware else None,
                    "load_bytes": load_bytes if hardware else None
                },
                "hardware_params": {
                    "mem_bw_gbs": mem_bw_gbs if hardware else None,
                    "mem_bw_bytes_per_sec": mem_bw if hardware else None,
                    "peak_pflops": peak_pflops if hardware else None,
                    "peak_flops": peak_flops if hardware else None,
                    "interconnect_bw_gbs": interconnect_bw_gbs if hardware else None,
                    "interconnect_latency_us": interconnect_latency_us if hardware else None,
                    "decode_load_overlap": decode_load_overlap if hardware else None,
                    "decode_comms_overlap": decode_comms_overlap if hardware else None,
                    "fixed_overhead_per_kernel_us": fixed_overhead_per_kernel_us if hardware else None
                },
                "timing_breakdown": {
                    "compute_time_ms": compute_time,
                    "load_time_ms": load_time,
                    "comm_time_ms": comm_time,
                    "transfer_time_ms": transfer_time_ms if hardware and comm_bytes > 0 else None,
                    "latency_ms": latency_ms if hardware and comm_bytes > 0 else None,
                    "wall_clock_time_ms": wall_clock_time,
                    "layer_kernel_overhead_ms": layer_kernel_overhead_us
                },
                "computed_metrics": {
                    "flops_per_package": flops_per_package,
                    "weight_memory_per_package": weight_mem_per_package,
                    "activation_memory_per_package": activation_mem_per_package,
                    "kv_cache_per_package": kv_cache_per_package,
                    "compute_time_ms": compute_time,
                    "load_time_ms": load_time,
                    "communication_time_ms": comm_time,
                    "wall_clock_time_ms": wall_clock_time,
                    "num_packages": num_packages
                },
                # "effective_bandwidth": {
                #     "note": "Effective bandwidth = bytes_moved / wall_clock_time (may be lower than peak due to compute overlap or inefficiency)",
                #     "total_memory_bandwidth_per_package_GBps": memory_bw_per_package,
                #     "kv_cache_bandwidth_per_package_GBps": kv_cache_bw_per_package,
                #     "communication_bandwidth_GBps": comm_bw
                # }
                # "formulae": {
                #     "compute_time": "(flops_per_package / peak_flops) * 1000",
                #     "load_time": "(load_bytes / mem_bw_bytes_per_sec) * 1000",
                #     "comm_time": "latency_ms + transfer_time_ms if comm_bytes > 0 else 0",
                #     "transfer_time": "(comm_bytes / interconnect_bw_bytes_per_sec) * 1000",
                #     "wall_clock_time_decode_overlap": "max(compute_time, load_time + comm_time) if decode_load_overlap and decode_comms_overlap",
                #     "wall_clock_time_no_overlap": "compute_time + load_time + comm_time",
                #     "memory_bw": "(total_memory_bytes / wall_clock_time_sec) / 1e9",
                #     "kv_cache_bw": "(kv_cache_bytes / wall_clock_time_sec) / 1e9",
                #     "comm_bw": "(comm_bytes / comm_time_sec) / 1e9"
                # }
            }
        
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
            load_time_ms=load_time,
            communication_bytes=comm_bytes,
            communication_time_ms=comm_time,
            memory_bandwidth_per_package_GBps=memory_bw_per_package,
            kv_cache_bandwidth_per_package_GBps=kv_cache_bw_per_package,
            communication_bandwidth_GBps=comm_bw,
            wall_clock_time_ms=wall_clock_time,
            bottleneck=bottleneck,
            num_packages=num_packages,
            debug_details=debug_info
        )
    
    def _compute_communication_bytes(
        self,
        batch_size: int,
        seq_len: int,
        phase: Phase,
        hardware: dict
    ) -> int:
        """
        Compute inter-package communication requirements.
        Default implementation returns 0. Subclasses override for specific patterns.
        Uses self.dtype for precision.
        
        Args:
            batch_size: Number of sequences
            seq_len: Sequence length
            phase: PREFILL or DECODE
            hardware: Hardware config (includes num_packages, parallelism strategy, etc.)
            
        Returns:
            Bytes to communicate (0 if no communication needed)
        """
        return 0
