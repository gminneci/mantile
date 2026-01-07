from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
import os, json

# Load environment variables from .env file
load_dotenv()

from pathlib import Path
from .models import HardwareSpecs, ModelIR
from .hardware_library import get_nvl72_specs, get_nvl72_rack_specs, list_available_configs, load_hardware_config
from .model_library import list_available_models, load_model_config
from . import config_service  # Pure functions module

MODELS_CFG_DIR = Path(__file__).parent / "data" / "model_configs"

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EstimateRequest(BaseModel):
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hardware_preset: str = "nvl72_rack"
    tp_size: int = 1
    batch_size: int = 1
    input_seq: int = 128
    output_seq: int = 128


from typing import Type
import importlib
import inspect

def _resolve_layer_class(name: str):
    """Resolve a JSON 'class' name to a Python Layer subclass in backend.layers.
    
    Assumes JSON class names are well-formed and directly importable.
    Returns class.
    """
    layers_pkg = importlib.import_module("backend.layers")
    return getattr(layers_pkg, name, None)


def load_model_config_json(model_id: str) -> dict:
    """Load raw model config JSON as dict."""
    config_path = MODELS_CFG_DIR / f"{model_id}.json"
    
    if not config_path.exists():
        available = list_available_models()
        raise FileNotFoundError(
            f"Model config '{model_id}' not found. Available models: {available}"
        )
    
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Validate unique layer names
    names = [lt.get("name") for lt in cfg["layer_types"]]
    if len(names) != len(set(names)):
        duplicates = [n for n in names if names.count(n) > 1]
        raise ValueError(f"Duplicate layer names found in model config: {set(duplicates)}")
    
    return cfg


@app.get("/hardware")
def list_hardware():
    """List all available hardware configurations."""
    configs = list_available_configs()
    return {"configs": configs}


@app.get("/hardware/{config_name}")
def get_hardware_details(config_name: str):
    """Get details for a specific hardware configuration."""
    try:
        hw = load_hardware_config(config_name)
        return hw.dict()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/models")
def list_models():
    """List all available pre-validated model configurations."""
    return {"models": sorted([cf.stem for cf in MODELS_CFG_DIR.glob("*.json")])}


@app.get("/api/layers")
def get_layers_info(model_id: str):
    """
    Get layer information for a specific model.
    """
    cfg = load_model_config_json(model_id)

    # Ensure expected format
    if "layer_types" not in cfg or not isinstance(cfg.get("layer_types"), list):
        raise HTTPException(status_code=400, detail="Model config missing 'layer_types' list")

    layers_out = []

    for lt in cfg["layer_types"]:
        cls = _resolve_layer_class(lt["class"])
        
        layers_out.append({
            **lt,  # Include all fields from JSON (name, class, count, specs)
            "available_parallelism": cls.get_supported_parallelism()
        })

    return {"layers": layers_out}


class LayerMetricsRequest(BaseModel):
    """Stateless request for layer metrics - includes all necessary context."""
    model_id: str
    hardware_config: str
    layer_type: str
    batch_size: int
    seq_length: int
    dtype: str
    phase: str = "prefill"
    tensor_parallel: int = 1
    context_parallel: int = 1
    sequence_parallel: int = 1


@app.post("/config/layer-metrics")
def compute_layer_metrics(req: LayerMetricsRequest):
    """
    Stateless endpoint: Compute metrics for a specific layer type.
    Uses layer name to look up specs from model config JSON.
    """
    from .layers import Phase, DataType
    
    # Load model config and hardware
    model_cfg = load_model_config_json(req.model_id)
    hardware = load_hardware_config(req.hardware_config)
    
    # Find layer type by name in config
    matches = [lt for lt in model_cfg["layer_types"] if lt.get("name") == req.layer_type]
    if len(matches) > 1:
        raise HTTPException(
            status_code=404, 
            detail=f"Duplicate layers '{req.layer_type}' found in model config"
        )
    elif len(matches) == 0:
        raise HTTPException(
            status_code=404, 
            detail=f"Layer '{req.layer_type}' missing from model config"
        )
    layer_type_entry = matches[0]
    
    # Extract layer info from config
    layer_class_name = layer_type_entry["class"]
    num_instances = layer_type_entry["count"]
    
    # Resolve layer class
    layer_class = _resolve_layer_class(layer_class_name)
    
    # Instantiate layer from specs
    parallelism = {p:getattr(req, p) for p in layer_class.get_supported_parallelism()}
    layer = layer_class(**layer_type_entry["specs"], parallelism=parallelism)

    # Compute metrics
    dtype_enum = DataType[req.dtype.upper()]
    
    # Map phase string to enum
    phase_map = {"prefill": Phase.PREFILL, "decode": Phase.DECODE}
    phase_enum = phase_map[req.phase.lower()]
    
    metrics = layer.compute_metrics(
        batch_size=req.batch_size,
        seq_len=req.seq_length,
        phase=phase_enum,
        dtype=dtype_enum
    )
    
    # Calculate aggregate for all instances of this layer
    total_weight_memory_gb = (metrics.weight_memory_per_chip * num_instances) / 1e9
    total_activation_memory_gb = (metrics.activation_memory_per_chip * num_instances) / 1e9
    total_kv_cache_gb = (metrics.kv_cache_per_chip * num_instances) / 1e9
    total_flops_tflops = (metrics.flops_per_chip * num_instances) / 1e12
    
    # Calculate bottleneck percentages
    compute_time = total_flops_tflops / hardware.bf16_tflops
    memory_time = total_weight_memory_gb / (hardware.hbm_bandwidth_gbps / 1000.0)
    comm_time = 0.001  # Simplified for now
    total_time = compute_time + memory_time + comm_time
    
    return {
        "layer_type": req.layer_type,
        "num_instances": num_instances,
        "num_chips": metrics.num_chips,
        "parallelism": parallelism,
        "memory": {
            "weights_per_chip_gb": metrics.weight_memory_per_chip / 1e9,
            "activation_per_chip_gb": metrics.activation_memory_per_chip / 1e9,
            "kv_cache_per_chip_gb": metrics.kv_cache_per_chip / 1e9,
            "total_weights_gb": total_weight_memory_gb,
            "total_activation_gb": total_activation_memory_gb,
            "total_kv_cache_gb": total_kv_cache_gb,
        },
        "compute": {
            "flops_per_chip_tflops": metrics.flops_per_chip / 1e12,
            "total_flops_tflops": total_flops_tflops,
        },
        "bottleneck": {
            "compute_percent": (compute_time / total_time * 100) if total_time > 0 else 0,
            "memory_percent": (memory_time / total_time * 100) if total_time > 0 else 0,
            "comm_percent": (comm_time / total_time * 100) if total_time > 0 else 0,
        }
    }


class SystemMetricsRequest(BaseModel):
    """Stateless request for system metrics - includes all necessary context."""
    model_id: str
    hardware_config: str
    batch_size: int = 1
    input_seq: int = 2048
    output_seq: int = 128
    # Layer configurations with parallelism and dtype
    layers: Dict[str, Dict[str, Any]] = {}
    # Example: {
    #   "attention": {"tensor_parallel": 4, "context_parallel": 2, "dtype": "bf16"},
    #   "feedforward": {"tensor_parallel": 8, "sequence_parallel": 1, "dtype": "bf16"}
    # }


@app.post("/config/system-metrics")
def compute_system_metrics(req: SystemMetricsRequest):
    """
    Stateless endpoint: Compute full system-level metrics.
    Builds ModelIR and hardware on-the-fly from request.
    """
    try:
        from .layers import DataType
        
        # Load pre-validated model config and hardware
        model_ir = load_model_config(req.model_id)
        hardware = load_hardware_config(req.hardware_config)
        
        # Build layer configs from request
        layer_configs = {}
        for layer_type, config in req.layers.items():
            # Count instances of this layer type
            num_instances = sum(
                1 for layer in model_ir.layers 
                if layer.module_type == layer_type
            )
            
            from .config_service import LayerConfig
            layer_configs[layer_type] = LayerConfig(
                layer_type=layer_type,
                layer_name=layer_type,
                parallelism={
                    "tensor_parallel": config.get("tensor_parallel", 1),
                    "context_parallel": config.get("context_parallel", 1),
                    "sequence_parallel": config.get("sequence_parallel", 1),
                },
                num_instances=num_instances,
                dtype=config.get("dtype", "bf16")
            )
        
        # Compute metrics using stateless function
        metrics = config_service.compute_system_metrics(
            model_ir=model_ir,
            hardware=hardware,
            layer_configs=layer_configs,
            batch_size=req.batch_size,
            input_seq=req.input_seq,
            output_seq=req.output_seq,
            dtype=DataType.BF16
        )
        
        return metrics
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# DEPLOYMENT API: Given a full deployment config, return metrics
# ============================================================

class DeploymentConfig(BaseModel):
    """Complete deployment configuration."""
    model_id: str
    hardware_config: str
    batch_size: int = 1
    input_seq: int = 2048
    output_seq: int = 128
    layer_parallelism: Dict[str, Dict[str, int]] = {}
    # Example: {
    #   "attention": {"tensor_parallel": 4, "context_parallel": 2},
    #   "feedforward": {"tensor_parallel": 8, "sequence_parallel": 1}
    # }


@app.post("/deployment/estimate")
def estimate_deployment(config: DeploymentConfig):
    """
    Comprehensive API: Given a complete deployment configuration,
    return all performance metrics.
    
    Stateless endpoint - all context provided in request.
    """
    try:
        from .layers import Phase, DataType
        
        # Load model and hardware
        model_ir = load_model_config(config.model_id)
        hardware = load_hardware_config(config.hardware_config)
        
        # Validate
        validation = config_service.validate_model(model_ir)
        if not validation.valid:
            raise HTTPException(
                status_code=400,
                detail=f"Model validation failed: {validation.issues}"
            )
        
        # Build layer configs from request
        layer_configs = {}
        layer_types = config_service.get_layer_types(model_ir)
        
        for layer_type in layer_types:
            # Count instances of this layer type
            num_instances = sum(
                1 for layer in model_ir.layers 
                if layer.module_type == layer_type
            )
            
            # Get parallelism from request or use defaults
            parallelism = config.layer_parallelism.get(layer_type, {})
            
            from .config_service import LayerConfig
            layer_configs[layer_type] = LayerConfig(
                layer_type=layer_type,
                layer_name=layer_type,
                parallelism={
                    "tensor_parallel": parallelism.get("tensor_parallel", 1),
                    "context_parallel": parallelism.get("context_parallel", 1),
                    "sequence_parallel": parallelism.get("sequence_parallel", 1),
                },
                num_instances=num_instances,
                dtype="bf16"
            )
        
        # Calculate system requirements
        requirements = config_service.calculate_minimum_system(
            model_ir=model_ir,
            hardware=hardware,
            layer_configs=layer_configs,
            batch_size=config.batch_size,
            seq_length=config.input_seq,
            phase=Phase.PREFILL,
            dtype=DataType.BF16
        )
        
        # Compute full metrics
        metrics = config_service.compute_system_metrics(
            model_ir=model_ir,
            hardware=hardware,
            layer_configs=layer_configs,
            batch_size=config.batch_size,
            input_seq=config.input_seq,
            output_seq=config.output_seq,
            dtype=DataType.BF16
        )
        
        # Return comprehensive result
        return {
            "deployment": {
                "model_id": config.model_id,
                "hardware_config": config.hardware_config,
                "batch_size": config.batch_size,
                "input_seq": config.input_seq,
                "output_seq": config.output_seq,
            },
            "validation": {
                "total_params": validation.total_params,
                "num_layers": validation.num_layers,
                "attention_type": validation.attention_type,
                "mlp_type": validation.mlp_type,
            },
            "requirements": {
                "min_chips": requirements.min_chips,
                "memory_per_chip_gb": requirements.memory_per_chip_gb,
                "fits_on_hardware": requirements.fits_on_hardware,
            },
            "performance": metrics,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Estimation failed: {str(e)}")


# Legacy "/estimate" endpoint removed in favor of stateless system metrics endpoints.
