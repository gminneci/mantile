from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List

from .models import HardwareSpecs, ParallelismConfig, ModelIR
from .hardware_library import get_nvl72_specs, get_nvl72_rack_specs, list_available_configs, load_hardware_config
from .ir_builder import build_model_ir
from .estimator import estimate_performance
from .config_service import ConfigurationService

app = FastAPI()

# Global service instance (in production, use dependency injection)
config_service = ConfigurationService()

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


# ============================================================
# NEW ENDPOINTS: Interactive Configuration Flow (Step 5)
# ============================================================

class LoadModelRequest(BaseModel):
    model_id: str
    hardware_config: str


@app.post("/config/load")
def load_model_and_hardware(req: LoadModelRequest):
    """
    Step 1: Load model and hardware.
    Returns model info and hardware specs.
    """
    try:
        # Load hardware
        hw = config_service.load_hardware(req.hardware_config)
        
        # Load model
        model_ir = config_service.load_model(req.model_id)
        
        # Validate
        validation = config_service.validate_model()
        
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


@app.get("/config/layer-types")
def get_layer_types():
    """
    Step 2: Get available layer types for parallelism configuration.
    """
    try:
        layer_types = config_service.get_layer_types()
        return {"layer_types": layer_types}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LayerParallelismRequest(BaseModel):
    layer_type: str
    tensor_parallel: int = 1
    context_parallel: int = 1
    sequence_parallel: int = 1


@app.post("/config/layer-parallelism")
def configure_layer_parallelism(req: LayerParallelismRequest):
    """
    Step 3: Configure parallelism for a specific layer type.
    """
    try:
        config_service.configure_layer_parallelism(
            layer_type=req.layer_type,
            tensor_parallel=req.tensor_parallel,
            context_parallel=req.context_parallel,
            sequence_parallel=req.sequence_parallel
        )
        
        config = config_service.get_layer_config(req.layer_type)
        
        return {
            "layer_type": req.layer_type,
            "config": {
                "parallelism": config.parallelism,
                "num_instances": config.num_instances,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SystemRequirementsRequest(BaseModel):
    batch_size: int = 1
    seq_length: int = 2048


@app.post("/config/system-requirements")
def calculate_system_requirements(req: SystemRequirementsRequest):
    """
    Step 4: Calculate minimum system requirements.
    """
    try:
        from .layers import Phase, DataType
        
        requirements = config_service.calculate_minimum_system(
            batch_size=req.batch_size,
            seq_length=req.seq_length,
            phase=Phase.PREFILL,
            dtype=DataType.BF16
        )
        
        return {
            "min_chips": requirements.min_chips,
            "total_weight_memory_gb": requirements.total_weight_memory_gb,
            "total_activation_memory_gb": requirements.total_activation_memory_gb,
            "total_kv_cache_gb": requirements.total_kv_cache_gb,
            "memory_per_chip_gb": requirements.memory_per_chip_gb,
            "fits_on_hardware": requirements.fits_on_hardware,
            "hw_capacity_gb": requirements.hw_capacity_gb,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SystemMetricsRequest(BaseModel):
    batch_size: int = 1
    input_seq: int = 2048
    output_seq: int = 128


@app.post("/config/system-metrics")
def compute_system_metrics(req: SystemMetricsRequest):
    """
    Step 5: Compute full system-level metrics (TTFT, TPOT, throughput, etc).
    """
    try:
        from .layers import DataType
        
        metrics = config_service.compute_system_metrics(
            batch_size=req.batch_size,
            input_seq=req.input_seq,
            output_seq=req.output_seq,
            dtype=DataType.BF16
        )
        
        return metrics
    except Exception as e:
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
    
    This is the main API endpoint for querying deployment performance.
    It handles the full flow: load model + hardware → configure parallelism → compute metrics.
    """
    try:
        # Create a fresh service instance for this request
        service = ConfigurationService()
        
        # Step 1: Load model and hardware
        service.load_hardware(config.hardware_config)
        service.load_model(config.model_id)
        
        # Step 2: Validate
        validation = service.validate_model()
        if not validation.valid:
            raise HTTPException(
                status_code=400,
                detail=f"Model validation failed: {validation.issues}"
            )
        
        # Step 3: Configure parallelism for each layer type
        if config.layer_parallelism:
            for layer_type, parallelism in config.layer_parallelism.items():
                service.configure_layer_parallelism(
                    layer_type=layer_type,
                    tensor_parallel=parallelism.get("tensor_parallel", 1),
                    context_parallel=parallelism.get("context_parallel", 1),
                    sequence_parallel=parallelism.get("sequence_parallel", 1)
                )
        else:
            # Default: no parallelism
            for layer_type in service.get_layer_types():
                service.configure_layer_parallelism(layer_type=layer_type)
        
        # Step 4: Calculate system requirements
        from .layers import Phase, DataType
        requirements = service.calculate_minimum_system(
            batch_size=config.batch_size,
            seq_length=config.input_seq,
            phase=Phase.PREFILL,
            dtype=DataType.BF16
        )
        
        # Step 5: Compute full metrics
        metrics = service.compute_system_metrics(
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

    # 2. Build IR
    try:
        # Generic builder works for any transformer architecture
        ir = build_model_ir(req.model_id)
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
