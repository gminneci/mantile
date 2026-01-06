"""
Backend service layer for the interactive configuration flow.
This module is UI-agnostic and can be called from FastAPI, Streamlit, or CLI.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from backend.models import HardwareSpecs, ModelIR
from backend.ir_builder import build_model_ir
from backend.hardware_library import load_hardware_config, list_available_configs
from backend.layers import (
    AttentionLayer,
    GroupedQueryAttentionLayer,
    MLPLayer,
    GatedMLPLayer,
    NormLayer,
    EmbeddingLayer,
    Phase,
    DataType,
)


@dataclass
class LayerConfig:
    """Configuration for a single layer type."""
    layer_type: str  # "embedding", "attention", "mlp", "norm"
    layer_name: str
    parallelism: Dict[str, int]  # {"tensor_parallel": 4, "context_parallel": 2, etc}
    num_instances: int = 1  # How many layers of this type (e.g., 80 attention layers)
    dtype: str = "bf16"  # Numerical precision format


@dataclass
class ModelValidation:
    """Validation results for a model."""
    valid: bool
    num_layers: int
    total_params: int
    expected_params: Optional[int]
    hidden_size: int
    vocab_size: int
    attention_type: str  # "MHA" or "GQA"
    mlp_type: str  # "dense" or "gated"
    issues: List[str]


@dataclass
class LayerGroupMetrics:
    """Aggregated metrics for a group of layers (e.g., all attention layers)."""
    layer_type: str
    num_layers: int
    total_params: int
    weight_memory_per_chip_gb: float
    activation_memory_per_chip_gb: float
    kv_cache_per_chip_gb: float
    flops_per_chip_tflops: float
    num_chips: int


@dataclass
class SystemRequirements:
    """Minimum system requirements for the configuration."""
    min_chips: int
    total_weight_memory_gb: float
    total_activation_memory_gb: float
    total_kv_cache_gb: float
    memory_per_chip_gb: float
    fits_on_hardware: bool
    hw_capacity_gb: float


class ConfigurationService:
    """Service for managing the interactive configuration flow."""
    
    def __init__(self):
        self.model_ir: Optional[ModelIR] = None
        self.hardware: Optional[HardwareSpecs] = None
        self.layer_configs: Dict[str, LayerConfig] = {}
    
    # Step 1: Model + Hardware Selection
    
    def list_available_hardware(self) -> List[str]:
        """List all available hardware configurations."""
        return list_available_configs()
    
    def load_hardware(self, config_name: str) -> HardwareSpecs:
        """Load hardware configuration."""
        self.hardware = load_hardware_config(config_name)
        return self.hardware
    
    def load_model(self, model_id: str) -> ModelIR:
        """Load model IR from HuggingFace."""
        self.model_ir = build_model_ir(model_id)
        return self.model_ir
    
    # Step 2: Validation & Parameter Check
    
    def validate_model(self) -> ModelValidation:
        """
        Validate the loaded model and return detailed info.
        Checks parameter count, layer configs, etc.
        """
        if not self.model_ir:
            return ModelValidation(
                valid=False,
                num_layers=0,
                total_params=0,
                expected_params=None,
                hidden_size=0,
                vocab_size=0,
                attention_type="unknown",
                mlp_type="unknown",
                issues=["Model not loaded"]
            )
        
        issues = []
        
        # Count parameters
        total_params = sum(layer.parameter_count for layer in self.model_ir.layers)
        
        # Determine attention type
        attention_type = "MHA"
        for layer in self.model_ir.layers:
            if layer.module_type == "attention" and layer.kv_heads:
                if layer.kv_heads < layer.num_heads:
                    attention_type = "GQA"
                    break
        
        # Determine MLP type (check for intermediate_size or gated variants)
        mlp_type = "dense"
        for layer in self.model_ir.layers:
            if layer.module_type == "feedforward" and layer.hidden_dim:
                # If hidden_dim is significantly larger than input_dim, likely gated
                if layer.hidden_dim > layer.input_dim * 2:
                    mlp_type = "gated"
                    break
        
        return ModelValidation(
            valid=len(issues) == 0,
            num_layers=self.model_ir.num_layers,
            total_params=total_params,
            expected_params=None,  # Could fetch from HF model card
            hidden_size=self.model_ir.hidden_size,
            vocab_size=self.model_ir.vocab_size,
            attention_type=attention_type,
            mlp_type=mlp_type,
            issues=issues
        )
    
    # Step 3: Per-Layer Parallelism Configuration
    
    def get_layer_types(self) -> List[str]:
        """Get all unique layer types in the model."""
        if not self.model_ir:
            return []
        return list(set(layer.module_type for layer in self.model_ir.layers))
    
    def configure_layer_parallelism(
        self,
        layer_type: str,
        tensor_parallel: int = 1,
        context_parallel: int = 1,
        sequence_parallel: int = 1
    ):
        """
        Configure parallelism for a specific layer type.
        """
        if not self.model_ir:
            raise ValueError("Model not loaded")
        
        # Count instances of this layer type
        num_instances = sum(
            1 for layer in self.model_ir.layers 
            if layer.module_type == layer_type
        )
        
        self.layer_configs[layer_type] = LayerConfig(
            layer_type=layer_type,
            layer_name=layer_type,
            parallelism={
                "tensor_parallel": tensor_parallel,
                "context_parallel": context_parallel,
                "sequence_parallel": sequence_parallel,
            },
            num_instances=num_instances
        )
    
    def get_layer_config(self, layer_type: str) -> Optional[LayerConfig]:
        """Get configuration for a specific layer type."""
        return self.layer_configs.get(layer_type)
    
    # Step 4: Minimum System Calculation
    
    @staticmethod
    def _instantiate_layer_static(layer_spec, parallelism: Dict[str, int], model_ir: ModelIR, hardware: HardwareSpecs):
        """
        Static method to instantiate a layer without instance state.
        Used for stateless API calls.
        """
        # Filter parallelism based on layer type
        if layer_spec.module_type == "attention":
            # Attention supports: TP, CP
            filtered_parallelism = {
                k: v for k, v in parallelism.items()
                if k in ["tensor_parallel", "context_parallel", "pipeline_parallel"]
            }
            
            if layer_spec.kv_heads and layer_spec.kv_heads < layer_spec.num_heads:
                # GQA
                return GroupedQueryAttentionLayer(
                    name=layer_spec.name,
                    layer_idx=layer_spec.layer_idx,
                    hidden_size=layer_spec.input_dim,
                    num_heads=layer_spec.num_heads,
                    num_kv_heads=layer_spec.kv_heads,
                    head_dim=layer_spec.head_dim,
                    parallelism=filtered_parallelism
                )
            else:
                # MHA
                return AttentionLayer(
                    name=layer_spec.name,
                    layer_idx=layer_spec.layer_idx,
                    hidden_size=layer_spec.input_dim,
                    num_heads=layer_spec.num_heads,
                    head_dim=layer_spec.head_dim,
                    parallelism=filtered_parallelism
                )
        
        elif layer_spec.module_type == "feedforward":
            # MLP supports: TP, SP (NOT CP)
            filtered_parallelism = {
                k: v for k, v in parallelism.items()
                if k in ["tensor_parallel", "sequence_parallel", "pipeline_parallel"]
            }
            
            if layer_spec.hidden_dim and layer_spec.hidden_dim > layer_spec.input_dim * 2:
                # Gated MLP
                return GatedMLPLayer(
                    name=layer_spec.name,
                    layer_idx=layer_spec.layer_idx,
                    hidden_size=layer_spec.input_dim,
                    intermediate_size=layer_spec.hidden_dim,
                    parallelism=filtered_parallelism
                )
            else:
                # Regular MLP
                return MLPLayer(
                    name=layer_spec.name,
                    layer_idx=layer_spec.layer_idx,
                    hidden_size=layer_spec.input_dim,
                    intermediate_size=layer_spec.hidden_dim or layer_spec.input_dim * 4,
                    parallelism=filtered_parallelism
                )
        
        elif layer_spec.module_type == "norm":
            # Norm doesn't use parallelism (replicated)
            return NormLayer(
                name=layer_spec.name,
                layer_idx=layer_spec.layer_idx,
                hidden_size=layer_spec.input_dim,
                has_bias=False,
                parallelism={}
            )
        
        elif layer_spec.module_type == "embedding":
            # Embedding doesn't use parallelism (replicated)
            return EmbeddingLayer(
                name=layer_spec.name,
                vocab_size=model_ir.vocab_size,
                hidden_size=model_ir.hidden_size,
                parallelism={}
            )
        
        else:
            return None

    def _instantiate_layer(self, layer_spec, parallelism: Dict[str, int]):
        """
        Instantiate an actual layer object from spec.
        Only passes supported parallelism types for each layer.
        """
        return ConfigurationService._instantiate_layer_static(
            layer_spec, parallelism, self.model_ir, self.hardware
        )

    def _instantiate_layer_old(self, layer_spec, parallelism: Dict[str, int]):
        """Old implementation kept for reference during migration."""
        # Filter parallelism based on layer type
        if layer_spec.module_type == "attention":
            # Attention supports: TP, CP
            filtered_parallelism = {
                k: v for k, v in parallelism.items()
                if k in ["tensor_parallel", "context_parallel", "pipeline_parallel"]
            }
            
            if layer_spec.kv_heads and layer_spec.kv_heads < layer_spec.num_heads:
                # GQA
                return GroupedQueryAttentionLayer(
                    name=layer_spec.name,
                    layer_idx=layer_spec.layer_idx,
                    hidden_size=layer_spec.input_dim,
                    num_heads=layer_spec.num_heads,
                    num_kv_heads=layer_spec.kv_heads,
                    head_dim=layer_spec.head_dim,
                    parallelism=filtered_parallelism
                )
            else:
                # MHA
                return AttentionLayer(
                    name=layer_spec.name,
                    layer_idx=layer_spec.layer_idx,
                    hidden_size=layer_spec.input_dim,
                    num_heads=layer_spec.num_heads,
                    head_dim=layer_spec.head_dim,
                    parallelism=filtered_parallelism
                )
        
        elif layer_spec.module_type == "feedforward":
            # MLP supports: TP, SP (NOT CP)
            filtered_parallelism = {
                k: v for k, v in parallelism.items()
                if k in ["tensor_parallel", "sequence_parallel", "pipeline_parallel"]
            }
            
            if layer_spec.hidden_dim and layer_spec.hidden_dim > layer_spec.input_dim * 2:
                # Gated MLP
                return GatedMLPLayer(
                    name=layer_spec.name,
                    layer_idx=layer_spec.layer_idx,
                    hidden_size=layer_spec.input_dim,
                    intermediate_size=layer_spec.hidden_dim,
                    parallelism=filtered_parallelism
                )
            else:
                # Regular MLP
                return MLPLayer(
                    name=layer_spec.name,
                    layer_idx=layer_spec.layer_idx,
                    hidden_size=layer_spec.input_dim,
                    intermediate_size=layer_spec.hidden_dim or layer_spec.input_dim * 4,
                    parallelism=filtered_parallelism
                )
        
        elif layer_spec.module_type == "norm":
            # Norm doesn't use parallelism (replicated)
            return NormLayer(
                name=layer_spec.name,
                layer_idx=layer_spec.layer_idx,
                hidden_size=layer_spec.input_dim,
                has_bias=False,
                parallelism={}
            )
        
        elif layer_spec.module_type == "embedding":
            # Embedding doesn't use parallelism (replicated)
            return EmbeddingLayer(
                name=layer_spec.name,
                vocab_size=self.model_ir.vocab_size,
                hidden_size=self.model_ir.hidden_size,
                parallelism={}
            )
        
        else:
            return None

    def calculate_minimum_system(
        self,
        batch_size: int = 1,
        seq_length: int = 2048,
        phase: Phase = Phase.PREFILL,
        dtype: DataType = DataType.BF16
    ) -> SystemRequirements:
        """
        Calculate minimum system requirements given current layer configs.
        Assumes all layer weights must fit simultaneously on the same chips.
        """
        if not self.hardware or not self.model_ir:
            raise ValueError("Hardware and model must be loaded")
        
        # For each layer type, calculate memory requirements using actual layers
        total_weight_memory = 0.0
        total_activation_memory = 0.0
        total_kv_cache = 0.0
        max_chips = 1
        
        for layer_type, config in self.layer_configs.items():
            # Find a representative layer of this type
            sample_layer_spec = next(
                (l for l in self.model_ir.layers if l.module_type == layer_type),
                None
            )
            if not sample_layer_spec:
                continue
            
            # Instantiate the actual layer to compute metrics
            layer = self._instantiate_layer(sample_layer_spec, config.parallelism)
            if not layer:
                continue
            
            # Compute metrics for one instance
            metrics = layer.compute_metrics(
                batch_size=batch_size,
                seq_len=seq_length,
                phase=phase,
                dtype=dtype
            )
            
            # Get number of chips for this layer type
            max_chips = max(max_chips, metrics.num_chips)
            
            # Aggregate for all layers of this type
            total_weight_memory += metrics.weight_memory_per_chip * config.num_instances
            total_activation_memory += metrics.activation_memory_per_chip * config.num_instances
            total_kv_cache += metrics.kv_cache_per_chip * config.num_instances
        
        # Convert to GB
        total_weight_memory_gb = total_weight_memory / 1e9
        total_activation_memory_gb = total_activation_memory / 1e9
        total_kv_cache_gb = total_kv_cache / 1e9
        
        # Calculate memory per chip (assuming weights are distributed)
        memory_per_chip_gb = (
            total_weight_memory_gb + 
            total_activation_memory_gb + 
            total_kv_cache_gb
        )
        
        # Check if it fits
        hw_capacity_gb = self.hardware.hbm_capacity_gb
        fits = memory_per_chip_gb <= hw_capacity_gb
        
        return SystemRequirements(
            min_chips=max_chips,
            total_weight_memory_gb=total_weight_memory_gb,
            total_activation_memory_gb=total_activation_memory_gb,
            total_kv_cache_gb=total_kv_cache_gb,
            memory_per_chip_gb=memory_per_chip_gb,
            fits_on_hardware=fits,
            hw_capacity_gb=hw_capacity_gb
        )
    
    @staticmethod
    def compute_system_metrics_static(
        model_ir: ModelIR,
        hardware: HardwareSpecs,
        layer_configs: Dict[str, LayerConfig],
        batch_size: int = 1,
        input_seq: int = 2048,
        output_seq: int = 128,
        dtype: DataType = DataType.BF16
    ) -> Dict:
        """
        Static method to compute system metrics without instance state.
        Used for stateless API calls.
        """
        # Prefill phase metrics
        prefill_metrics = ConfigurationService._compute_phase_metrics_static(
            model_ir=model_ir,
            hardware=hardware,
            layer_configs=layer_configs,
            batch_size=batch_size,
            seq_length=input_seq,
            phase=Phase.PREFILL,
            dtype=dtype
        )
        
        # Decode phase metrics (per token)
        decode_metrics = ConfigurationService._compute_phase_metrics_static(
            model_ir=model_ir,
            hardware=hardware,
            layer_configs=layer_configs,
            batch_size=batch_size,
            seq_length=1,  # Decode is one token at a time
            phase=Phase.DECODE,
            dtype=dtype
        )
        
        # Calculate latencies
        # Use BF16 compute for now
        peak_tflops_per_chip = hardware.bf16_tflops
        hbm_bw_per_chip = hardware.hbm_bandwidth_gbps
        
        # Prefill latency (TTFT)
        prefill_compute_time_ms = (prefill_metrics["total_flops"] / 1e12) / peak_tflops_per_chip * 1000
        prefill_memory_time_ms = (prefill_metrics["total_weight_memory"] / 1e9) / hbm_bw_per_chip * 1000
        ttft_ms = max(prefill_compute_time_ms, prefill_memory_time_ms)
        
        # Decode latency (TPOT)
        decode_compute_time_ms = (decode_metrics["total_flops"] / 1e12) / peak_tflops_per_chip * 1000
        decode_memory_time_ms = (decode_metrics["total_weight_memory"] / 1e9) / hbm_bw_per_chip * 1000
        tpot_ms = max(decode_compute_time_ms, decode_memory_time_ms)
        
        # Throughput (tokens/sec)
        throughput_tokens_s = 1000.0 / tpot_ms if tpot_ms > 0 else 0
        
        # Total latency for full sequence
        total_latency_ms = ttft_ms + (tpot_ms * output_seq)
        
        # Bottleneck analysis
        if prefill_compute_time_ms > prefill_memory_time_ms * 1.2:
            bottleneck = "compute"
        elif prefill_memory_time_ms > prefill_compute_time_ms * 1.2:
            bottleneck = "memory"
        else:
            bottleneck = "balanced"
        
        return {
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "throughput_tokens_s": throughput_tokens_s,
            "total_latency_ms": total_latency_ms,
            "prefill": {
                "flops_total": prefill_metrics["total_flops"],
                "flops_per_chip": prefill_metrics["flops_per_chip"],
                "compute_time_ms": prefill_compute_time_ms,
                "memory_time_ms": prefill_memory_time_ms,
            },
            "decode": {
                "flops_total": decode_metrics["total_flops"],
                "flops_per_chip": decode_metrics["flops_per_chip"],
                "compute_time_ms": decode_compute_time_ms,
                "memory_time_ms": decode_memory_time_ms,
            },
            "memory": {
                "weight_memory_gb": prefill_metrics["total_weight_memory"] / 1e9,
                "activation_memory_gb": prefill_metrics["total_activation_memory"] / 1e9,
                "kv_cache_gb": prefill_metrics["total_kv_cache"] / 1e9,
                "total_memory_gb": (
                    prefill_metrics["total_weight_memory"] +
                    prefill_metrics["total_activation_memory"] +
                    prefill_metrics["total_kv_cache"]
                ) / 1e9,
                "memory_per_chip_gb": prefill_metrics["memory_per_chip"],
                "hw_capacity_gb": hardware.hbm_capacity_gb,
            },
            "system": {
                "num_chips": prefill_metrics["num_chips"],
                "bottleneck": bottleneck,
                "fits_on_hardware": prefill_metrics["memory_per_chip"] <= hardware.hbm_capacity_gb,
            }
        }
    
    @staticmethod
    def _compute_phase_metrics_static(
        model_ir: ModelIR,
        hardware: HardwareSpecs,
        layer_configs: Dict[str, LayerConfig],
        batch_size: int,
        seq_length: int,
        phase: Phase,
        dtype: DataType
    ) -> Dict:
        """Static method to compute metrics for a single phase."""
        total_flops = 0.0
        total_weight_memory = 0.0
        total_activation_memory = 0.0
        total_kv_cache = 0.0
        max_chips = 1
        
        for layer_type, config in layer_configs.items():
            # Find a representative layer
            sample_layer_spec = next(
                (l for l in model_ir.layers if l.module_type == layer_type),
                None
            )
            if not sample_layer_spec:
                continue
            
            # Instantiate layer using static method
            layer = ConfigurationService._instantiate_layer_static(
                sample_layer_spec, config.parallelism, model_ir, hardware
            )
            if not layer:
                continue
            
            # Compute metrics
            metrics = layer.compute_metrics(
                batch_size=batch_size,
                seq_len=seq_length,
                phase=phase,
                dtype=dtype
            )
            
            # Aggregate
            total_flops += metrics.flops_total * config.num_instances
            total_weight_memory += metrics.weight_memory_per_chip * config.num_instances
            total_activation_memory += metrics.activation_memory_per_chip * config.num_instances
            total_kv_cache += metrics.kv_cache_per_chip * config.num_instances
            max_chips = max(max_chips, metrics.num_chips)
        
        memory_per_chip = (
            total_weight_memory + 
            total_activation_memory + 
            total_kv_cache
        ) / 1e9 if max_chips > 0 else 0
        
        return {
            "total_flops": total_flops,
            "flops_per_chip": total_flops / max_chips if max_chips > 0 else 0,
            "total_weight_memory": total_weight_memory,
            "total_activation_memory": total_activation_memory,
            "total_kv_cache": total_kv_cache,
            "memory_per_chip": memory_per_chip,
            "num_chips": max_chips,
        }

    # Step 5: System-Level Metrics
    
    def compute_system_metrics(
        self,
        batch_size: int = 1,
        input_seq: int = 2048,
        output_seq: int = 128,
        dtype: DataType = DataType.BF16
    ) -> Dict:
        """
        Compute full system metrics (TTFT, TPOT, throughput, etc).
        This aggregates metrics from all configured layers.
        """
        if not self.hardware or not self.model_ir:
            raise ValueError("Hardware and model must be loaded")
        
        if not self.layer_configs:
            raise ValueError("No layer configurations set. Configure parallelism first.")
        
        # Prefill phase metrics
        prefill_metrics = self._compute_phase_metrics(
            batch_size=batch_size,
            seq_length=input_seq,
            phase=Phase.PREFILL,
            dtype=dtype
        )
        
        # Decode phase metrics (per token)
        decode_metrics = self._compute_phase_metrics(
            batch_size=batch_size,
            seq_length=1,  # Decode is one token at a time
            phase=Phase.DECODE,
            dtype=dtype
        )
        
        # Calculate latencies
        # Use BF16 compute for now
        peak_tflops_per_chip = self.hardware.bf16_tflops
        hbm_bw_per_chip = self.hardware.hbm_bandwidth_gbps
        
        # Prefill latency (TTFT)
        prefill_compute_time_ms = (prefill_metrics["total_flops"] / 1e12) / peak_tflops_per_chip * 1000
        prefill_memory_time_ms = (prefill_metrics["total_weight_memory"] / 1e9) / hbm_bw_per_chip * 1000
        ttft_ms = max(prefill_compute_time_ms, prefill_memory_time_ms)
        
        # Decode latency (TPOT)
        decode_compute_time_ms = (decode_metrics["total_flops"] / 1e12) / peak_tflops_per_chip * 1000
        decode_memory_time_ms = (decode_metrics["total_weight_memory"] / 1e9) / hbm_bw_per_chip * 1000
        tpot_ms = max(decode_compute_time_ms, decode_memory_time_ms)
        
        # Throughput (tokens/sec)
        throughput_tokens_s = 1000.0 / tpot_ms if tpot_ms > 0 else 0
        
        # Total latency for full sequence
        total_latency_ms = ttft_ms + (tpot_ms * output_seq)
        
        # Bottleneck analysis
        if prefill_compute_time_ms > prefill_memory_time_ms * 1.2:
            bottleneck = "compute"
        elif prefill_memory_time_ms > prefill_compute_time_ms * 1.2:
            bottleneck = "memory"
        else:
            bottleneck = "balanced"
        
        return {
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "throughput_tokens_s": throughput_tokens_s,
            "total_latency_ms": total_latency_ms,
            "prefill": {
                "flops_total": prefill_metrics["total_flops"],
                "flops_per_chip": prefill_metrics["flops_per_chip"],
                "compute_time_ms": prefill_compute_time_ms,
                "memory_time_ms": prefill_memory_time_ms,
            },
            "decode": {
                "flops_total": decode_metrics["total_flops"],
                "flops_per_chip": decode_metrics["flops_per_chip"],
                "compute_time_ms": decode_compute_time_ms,
                "memory_time_ms": decode_memory_time_ms,
            },
            "memory": {
                "weight_memory_gb": prefill_metrics["total_weight_memory"] / 1e9,
                "activation_memory_gb": prefill_metrics["total_activation_memory"] / 1e9,
                "kv_cache_gb": prefill_metrics["total_kv_cache"] / 1e9,
                "total_memory_gb": (
                    prefill_metrics["total_weight_memory"] +
                    prefill_metrics["total_activation_memory"] +
                    prefill_metrics["total_kv_cache"]
                ) / 1e9,
                "memory_per_chip_gb": prefill_metrics["memory_per_chip"],
                "hw_capacity_gb": self.hardware.hbm_capacity_gb,
            },
            "system": {
                "num_chips": prefill_metrics["num_chips"],
                "bottleneck": bottleneck,
                "fits_on_hardware": prefill_metrics["memory_per_chip"] <= self.hardware.hbm_capacity_gb,
            }
        }
    
    def _compute_phase_metrics(
        self,
        batch_size: int,
        seq_length: int,
        phase: Phase,
        dtype: DataType
    ) -> Dict:
        """Compute metrics for a single phase (prefill or decode)."""
        total_flops = 0.0
        total_weight_memory = 0.0
        total_activation_memory = 0.0
        total_kv_cache = 0.0
        max_chips = 1
        
        for layer_type, config in self.layer_configs.items():
            # Find a representative layer
            sample_layer_spec = next(
                (l for l in self.model_ir.layers if l.module_type == layer_type),
                None
            )
            if not sample_layer_spec:
                continue
            
            # Instantiate layer
            layer = self._instantiate_layer(sample_layer_spec, config.parallelism)
            if not layer:
                continue
            
            # Compute metrics
            metrics = layer.compute_metrics(
                batch_size=batch_size,
                seq_len=seq_length,
                phase=phase,
                dtype=dtype
            )
            
            # Aggregate
            total_flops += metrics.flops_total * config.num_instances
            total_weight_memory += metrics.weight_memory_per_chip * config.num_instances
            total_activation_memory += metrics.activation_memory_per_chip * config.num_instances
            total_kv_cache += metrics.kv_cache_per_chip * config.num_instances
            max_chips = max(max_chips, metrics.num_chips)
        
        memory_per_chip = (
            total_weight_memory + 
            total_activation_memory + 
            total_kv_cache
        ) / 1e9 if max_chips > 0 else 0
        
        return {
            "total_flops": total_flops,
            "flops_per_chip": total_flops / max_chips if max_chips > 0 else 0,
            "total_weight_memory": total_weight_memory,
            "total_activation_memory": total_activation_memory,
            "total_kv_cache": total_kv_cache,
            "memory_per_chip": memory_per_chip,
            "num_chips": max_chips,
        }
