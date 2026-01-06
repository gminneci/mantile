from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from .models import HardwareSpecs, ParallelismConfig, ModelIR
from .hardware_library import get_nvl72_specs, get_nvl72_rack_specs, list_available_configs, load_hardware_config
from .model_library import list_available_models, load_model_config, get_model_metadata
from .estimator import estimate_performance
from . import config_service  # Pure functions module

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

@app.get("/hardware")
def list_hardware():
    """List all available hardware configurations."""
    configs = list_available_configs()
    return {
        "configs": configs,
        # Backward compatibility
        "nvl72_single": get_nvl72_specs().dict(),
        "nvl72_rack": get_nvl72_rack_specs().dict()
    }


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
    models = list_available_models()
    # Get metadata for each model
    model_list = []
    for model_id in models:
        try:
            metadata = get_model_metadata(model_id)
            model_list.append(metadata)
        except Exception as e:
            # Skip models that fail to load
            continue
    return {"models": model_list}


@app.get("/models/{model_id}")
def get_model_details(model_id: str):
    """Get details for a specific model configuration."""
    try:
        metadata = get_model_metadata(model_id)
        return metadata
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================
# NEW ENDPOINTS: Interactive Configuration Flow (Step 5)
# ============================================================

class LoadModelRequest(BaseModel):
    model_id: str
    hardware_config: str


@app.post("/config/load")
def load_model_and_hardware(req: LoadModelRequest):
    """
    Stateless: Load model and hardware, return info and validation.
    No state stored on server - client receives all data.
    """
    try:
        # Load model and hardware
        model_ir = load_model_config(req.model_id)
        hw = load_hardware_config(req.hardware_config)
        
        # Validate
        validation = config_service.validate_model(model_ir)
        
        return {
            "model": {
                "id": req.model_id,
                "num_layers": model_ir.num_layers,
                "hidden_size": model_ir.hidden_size,
                "vocab_size": model_ir.vocab_size,
            },
            "hardware": {
                "name": hw.name,
                "description": hw.description,
                "bf16_tflops": hw.bf16_tflops,
                "hbm_capacity_gb": hw.hbm_capacity_gb,
                "chips_per_node": hw.chips_per_node,
            },
            "validation": {
                "valid": validation.valid,
                "total_params": validation.total_params,
                "attention_type": validation.attention_type,
                "mlp_type": validation.mlp_type,
                "issues": validation.issues,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/layers")
def get_layers_info():
    """
    DEPRECATED: Use /api/layers with model_id parameter instead.
    Get detailed information about all layers in the model.
    Returns layer types, counts, and available parallelism strategies.
    """
    try:
        if not config_service.model_ir:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        # Group layers by type
        layer_info = {}
        for layer in config_service.model_ir.layers:
            layer_type = layer.module_type
            if layer_type not in layer_info:
                layer_info[layer_type] = {
                    "type": layer_type,
                    "count": 0,
                    "sample_layer": {
                        "name": layer.name,
                        "input_dim": layer.input_dim,
                        "output_dim": layer.output_dim,
                    },
                    "available_parallelism": []
                }
            layer_info[layer_type]["count"] += 1
        
        # Add available parallelism strategies per layer type
        for layer_type in layer_info:
            if layer_type == "attention":
                layer_info[layer_type]["available_parallelism"] = ["tensor_parallel", "context_parallel"]
            elif layer_type == "feedforward":
                layer_info[layer_type]["available_parallelism"] = ["tensor_parallel", "sequence_parallel"]
            elif layer_type == "norm":
                layer_info[layer_type]["available_parallelism"] = []  # replicated
            elif layer_type == "embedding":
                layer_info[layer_type]["available_parallelism"] = []  # replicated
            
            # All layers support dtype selection
            layer_info[layer_type]["available_dtypes"] = ["fp32", "fp16", "bf16", "fp8", "int8"]
        
        return {"layers": list(layer_info.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/layers")
def get_layers_info_stateless(model_id: str):
    """
    Stateless endpoint: Get layer information for a specific model.
    No server-side state required.
    """
    try:
        # Load pre-validated model config
        model_ir = load_model_config(model_id)
        
        # Group layers by type
        layer_info = {}
        for layer in model_ir.layers:
            layer_type = layer.module_type
            if layer_type not in layer_info:
                layer_info[layer_type] = {
                    "type": layer_type,
                    "count": 0,
                    "sample_layer": {
                        "name": layer.name,
                        "input_dim": layer.input_dim,
                        "output_dim": layer.output_dim,
                    },
                    "available_parallelism": []
                }
            layer_info[layer_type]["count"] += 1
        
        # Add available parallelism strategies per layer type
        for layer_type in layer_info:
            if layer_type == "attention":
                layer_info[layer_type]["available_parallelism"] = ["tensor_parallel", "context_parallel"]
            elif layer_type == "feedforward":
                layer_info[layer_type]["available_parallelism"] = ["tensor_parallel", "sequence_parallel"]
            elif layer_type == "norm":
                layer_info[layer_type]["available_parallelism"] = []  # replicated
            elif layer_type == "embedding":
                layer_info[layer_type]["available_parallelism"] = []  # replicated
            
            # All layers support dtype selection
            layer_info[layer_type]["available_dtypes"] = ["fp32", "fp16", "bf16", "fp8", "int8"]
        
        return {"layers": list(layer_info.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LayerMetricsRequest(BaseModel):
    """Stateless request for layer metrics - includes all necessary context."""
    model_id: str
    hardware_config: str
    layer_type: str
    batch_size: int = 1
    seq_length: int = 2048
    dtype: str = "bf16"
    tensor_parallel: int = 1
    context_parallel: int = 1
    sequence_parallel: int = 1


@app.post("/config/layer-metrics")
def compute_layer_metrics(req: LayerMetricsRequest):
    """
    Stateless endpoint: Compute metrics for a specific layer type.
    Builds ModelIR and hardware on-the-fly from request.
    """
    try:
        from .layers import Phase, DataType
        
        # Load pre-validated model config
        model_ir = load_model_config(req.model_id)
        hardware = load_hardware_config(req.hardware_config)
        
        # Find a representative layer of this type
        sample_layer_spec = next(
            (l for l in model_ir.layers if l.module_type == req.layer_type),
            None
        )
        if not sample_layer_spec:
            raise HTTPException(status_code=404, detail=f"No layers of type {req.layer_type} found")
        
        # Build parallelism config from request
        parallelism = {
            "tensor_parallel": req.tensor_parallel,
            "context_parallel": req.context_parallel,
            "sequence_parallel": req.sequence_parallel,
        }
        
        # Instantiate the layer
        layer = config_service.instantiate_layer(
            sample_layer_spec, parallelism, model_ir, hardware
        )
        if not layer:
            raise HTTPException(status_code=500, detail="Failed to instantiate layer")
        
        # Compute metrics
        try:
            dtype_enum = DataType[req.dtype.upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid dtype: {req.dtype}")
        
        metrics = layer.compute_metrics(
            batch_size=req.batch_size,
            seq_len=req.seq_length,
            phase=Phase.PREFILL,
            dtype=dtype_enum
        )
        
        # Count instances of this layer type
        num_instances = sum(
            1 for l in model_ir.layers 
            if l.module_type == req.layer_type
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# ============================================================
# EXISTING ENDPOINT: Original Estimation Flow
# ============================================================

@app.post("/estimate")
def run_estimation(req: EstimateRequest):
    # 1. Load Hardware
    if req.hardware_preset == "nvl72_rack":
        hw = get_nvl72_rack_specs()
    elif req.hardware_preset == "nvl72_single":
        hw = get_nvl72_specs()
    else:
        raise HTTPException(status_code=404, detail="Hardware preset not found")

    # 2. Load Model Config
    try:
        # Load pre-validated model config
        ir = load_model_config(req.model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
    # 3. Parallel Config
    par = ParallelismConfig(
        tp_size=req.tp_size,
        batch_size=req.batch_size,
        input_seq_len=req.input_seq,
        output_seq_len=req.output_seq
    )
    
    # 4. Estimate
    result = estimate_performance(hw, ir, par)
    
    return result
