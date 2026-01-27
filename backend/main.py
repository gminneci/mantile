from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import os, json
from typing import Type
import importlib

from pathlib import Path
from .layers import Phase, DataType

MODELS_CFG_DIR = Path(__file__).parent / "data" / "model_configs"
HARDWARE_CONFIGS_DIR = Path(__file__).parent / "data" / "hardware_configs"

# Path prefix for deployment (e.g., /estimator-api)
ROOT_PATH = os.getenv("ROOT_PATH", "")

# CORS origins (comma-separated)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

app = FastAPI(
    root_path=ROOT_PATH,
    title="Mantile API",
    description="LLM Performance Estimation API"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PhaseMetricsRequest(BaseModel):
    """Stateless request for a single phase (prefill or decode) - includes all necessary context."""
    model_id: str
    hardware_id: str
    batch_size: int
    seq_len: int  # Sequence length being processed (full prompt for prefill, 1 for decode)
    context_len: Optional[int] = None  # Total context in KV cache (for decode phase)
    debug: bool = False  # If True, include debug_details in layer metrics
    # Layer configurations with parallelism and dtype
    layers: Dict[str, Dict[str, Any]] = {}


class SystemMetricsRequest(BaseModel):
    """Wrapper for two-phase system metrics request."""
    prefill_req: PhaseMetricsRequest
    decode_req: PhaseMetricsRequest


def _construct_layer(layer_class: Type, specs: Dict[str, Any], dtype_enum: DataType, parallelism: Dict[str, Any]):
    """Safely construct a layer instance by filtering JSON specs to the class __init__ signature.

    Many JSON configs include fields like input_dim/output_dim/parameter_count that are not accepted by
    specific layer constructors. This helper filters the specs to only accepted parameters and ensures a
    sensible default for required fields like layer_idx.
    """
    import inspect
    sig = inspect.signature(layer_class.__init__)
    allowed = {k: v for k, v in specs.items() if k in sig.parameters}
    if 'layer_idx' in sig.parameters and 'layer_idx' not in allowed:
        # default to 0 if not present in specs
        allowed['layer_idx'] = specs.get('layer_idx', 0)
    return layer_class(**allowed, dtype=dtype_enum, parallelism=parallelism)


def _resolve_layer_class(name: str):
    """Resolve a JSON 'class' name to a Python Layer subclass in backend.layers.
    
    Assumes JSON class names are well-formed and directly importable.
    Returns class.
    """
    layers_pkg = importlib.import_module("backend.layers")
    return getattr(layers_pkg, name, None)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Mantile API is running"}


@app.get("/models")
def list_models() -> List[Dict[str, Any]]:
    """List all available model configurations."""
    models = []
    for config_file in MODELS_CFG_DIR.glob("*.json"):
        model_id = config_file.stem
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        models.append({
            "id": model_id,
            "name": cfg.get("name", model_id.replace("_", " ").replace("-", " ")),
            "total_params": cfg.get("total_params", 0)
        })
    return sorted(models, key=lambda x: x["total_params"])

@app.get("/models/{model_id}")
def load_model_config(model_id: str) -> dict:
    """Load raw model config JSON as dict."""
    config_path = MODELS_CFG_DIR / f"{model_id}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config '{model_id}' not found"
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
def list_hardware() -> List[Dict[str, str]]:
    """List all available hardware configurations."""
    hardware_list = []
    for config_file in HARDWARE_CONFIGS_DIR.glob("*.json"):
        config_id = config_file.stem
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        hardware_list.append({
            "id": config_id,
            "name": cfg.get("name", config_id.replace("_", " ").replace("-", " "))
        })
    return sorted(hardware_list, key=lambda x: x["name"])

@app.get("/hardware/{config_name}")
def load_hardware_config(config_name: str) -> Dict[str, Any]:
    """
    Load hardware configuration from JSON file
    """
    config_path = HARDWARE_CONFIGS_DIR / f"{config_name}.json"
    with open(config_path, 'r') as f:
        return json.load(f)

@app.get("/api/layers")
def get_layers_info(model_id: str):
    """
    Get layer information for a specific model.
    """
    cfg = load_model_config(model_id)

    # Ensure expected format
    if "layer_types" not in cfg or not isinstance(cfg.get("layer_types"), list):
        raise HTTPException(status_code=400, detail="Model config missing 'layer_types' list")

    layers_out = []

    for lt in cfg["layer_types"]:
        cls = _resolve_layer_class(lt["class"])
        layers_out.append({
            **lt,
            "available_parallelism": cls.get_supported_parallelism()
        })

    return {"layers": layers_out}


def compute_phase_metrics(phase_req: PhaseMetricsRequest, phase: Phase) -> dict:
    """
    Helper function: Compute metrics for a single phase (prefill or decode).
    
    Args:
        phase_req: Phase-specific request with layer configurations
        phase: Phase enum (PREFILL or DECODE)
        
    Returns:
        Dict with aggregated phase metrics
    """

    # Load model and hardware config
    model_cfg = load_model_config(phase_req.model_id)
    hardware_cfg = load_hardware_config(phase_req.hardware_id)

    total_weight_memory_gb = 0.0
    total_activation_memory_gb = 0.0
    total_kv_cache_gb = 0.0
    wall_clock_time_ms = 0.0
    total_compute_time_ms = 0.0
    total_load_time_ms = 0.0
    max_packages = 1
    layer_debug_details = {}  # Collect debug details from each layer if debug=True

    for layer_name, config in phase_req.layers.items():
        # Find layer definition in model config
        layer_type = next((lt for lt in model_cfg['layer_types'] if lt['name'] == layer_name), None)
        if not layer_type:
            raise HTTPException(
                status_code=400,
                detail=f"Layer '{layer_name}' not found in model config"
            )

        # Resolve and instantiate layer
        layer_class = _resolve_layer_class(layer_type['class'])
        parallelism = {p: config.get(p, 1) for p in layer_class.get_supported_parallelism()}
        dtype_enum = DataType(config.get("dtype", "bf16").lower())
        layer = _construct_layer(layer_class, layer_type["specs"], dtype_enum, parallelism)

        num_instances = layer_type.get("count", 1)

        # Compute metrics for this phase (base.py handles hardware-aware timing)
        # Pass full hardware config - compute_metrics will extract HBM memory
        # For decode, use context_len for KV cache size if provided
        m = layer.compute_metrics(
            batch_size=phase_req.batch_size,
            seq_len=phase_req.seq_len,
            phase=phase,
            hardware=hardware_cfg,
            context_len=phase_req.context_len,
            debug=phase_req.debug
        )
        
        max_packages = max(max_packages, m.num_packages)
        total_weight_memory_gb += (m.weight_memory_per_package * num_instances) / 1e9
        total_activation_memory_gb += (m.activation_memory_per_package * num_instances) / 1e9
        total_kv_cache_gb += (m.kv_cache_per_package * num_instances) / 1e9
        
        # Aggregate wall_clock_time which includes phase-aware overlap model
        wall_clock_time_ms += m.wall_clock_time_ms * num_instances
        
        # Track separate timing components for debugging
        total_compute_time_ms += m.compute_time_ms * num_instances
        total_load_time_ms += m.load_time_ms * num_instances
        
        # Collect debug details if requested
        if phase_req.debug and m.debug_details:
            m.debug_details['layer_params'] = layer_type["specs"]
            layer_debug_details[layer_name] = m.debug_details

    return {
        "total_weight_memory_gb": total_weight_memory_gb,
        "total_activation_memory_gb": total_activation_memory_gb,
        "total_kv_cache_gb": total_kv_cache_gb,
        "wall_clock_time_ms": wall_clock_time_ms,
        "max_packages": max_packages,
        "total_compute_time_ms": total_compute_time_ms,
        "total_load_time_ms": total_load_time_ms,
        "layer_debug_details": layer_debug_details if phase_req.debug else None
    }


class LayerMetricsRequest(BaseModel):
    """Request for computing metrics for a single layer."""
    model_id: str
    hardware_id: str
    batch_size: int
    seq_len: int
    layer_name: str
    phase: Phase
    layer_config: Dict[str, Any]  # Contains tensor_parallel, context_parallel, sequence_parallel, dtype
    context_len: Optional[int] = None  # Total context in KV cache (for decode phase)
    debug: bool = False  # If True, include debug_details in response


@app.post("/config/layer-metrics")
def compute_layer_metrics(request: LayerMetricsRequest):
    """
    Compute metrics for a single layer.
    
    Args:
        request: LayerMetricsRequest with layer-specific configuration
        
    Returns:
        Dict with layer-specific metrics (FLOPs, memory, time, bottleneck)
    """
    # Load model and hardware config
    model_cfg = load_model_config(request.model_id)
    hardware_cfg = load_hardware_config(request.hardware_id)
    
    # Find the layer type in the model config
    layer_type = next((lt for lt in model_cfg["layer_types"] if lt["name"] == request.layer_name), None)
    if not layer_type:
        raise HTTPException(status_code=404, detail=f"Layer '{request.layer_name}' not found in model config")
    
    # Parse dtype
    dtype_str = request.layer_config.get("dtype", "bf16")
    try:
        dtype_enum = DataType[dtype_str.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid dtype: {dtype_str}")
    
    # Resolve layer class
    layer_class = _resolve_layer_class(layer_type["class"])
    parallelism = {p: request.layer_config.get(p, 1) for p in layer_class.get_supported_parallelism()}
    
    # Construct layer instance
    layer = _construct_layer(layer_class, layer_type["specs"], dtype_enum, parallelism)
    
    # Compute layer metrics (returns LayerMetrics dataclass)
    # Pass full hardware config - compute_metrics will extract HBM memory
    metrics = layer.compute_metrics(
        hardware=hardware_cfg,
        batch_size=request.batch_size,
        seq_len=request.seq_len,
        phase=request.phase,
        context_len=request.context_len,
        debug=request.debug
    )
    
    # Get layer constructor parameters for debugging
    layer_config = {
        "layer_idx": getattr(layer, 'layer_idx', None),
        "dtype": str(layer.dtype),
        "parallelism": layer.parallelism,
    }
    # Add layer-specific parameters
    for attr in ['hidden_size', 'num_heads', 'num_kv_heads', 'head_dim', 'intermediate_size', 
                 'num_experts', 'top_k', 'vocab_size', 'window_size', 'sliding_window',
                 'num_projections', 'num_shared_experts']:
        if hasattr(layer, attr):
            layer_config[attr] = getattr(layer, attr)
    
    response = {
        "layer_name": request.layer_name,
        "layer_type": layer_type["class"],
        "layer_config": layer_config,
        "request_params": {
            "batch_size": request.batch_size,
            "seq_len": request.seq_len,
            "phase": str(request.phase),
        },
        "metrics": {
            "flops_per_package": metrics.flops_per_package,
            "weight_memory_per_package": metrics.weight_memory_per_package,
            "activation_memory_per_package": metrics.activation_memory_per_package,
            "kv_cache_per_package": metrics.kv_cache_per_package,
            "compute_time_ms": metrics.compute_time_ms,
            "load_time_ms": metrics.load_time_ms,
            "communication_time_ms": metrics.communication_time_ms,
            "wall_clock_time_ms": metrics.wall_clock_time_ms,
            "bottleneck": metrics.bottleneck,
            "num_packages": metrics.num_packages,
        }
    }
    
    # Include debug details if requested
    if request.debug and metrics.debug_details:
        metrics.debug_details['layer_params'] = layer_type["specs"]
        response["debug_details"] = metrics.debug_details
    
    return response


@app.post("/config/system-metrics")
def compute_system_metrics(request: SystemMetricsRequest):
    """
    Stateless endpoint: Compute full system-level metrics.
    Takes separate requests for prefill and decode phases.
    
    Args:
        request: SystemMetricsRequest containing prefill_req and decode_req
    """
    prefill_req = request.prefill_req
    decode_req = request.decode_req
    
    # Load pre-validated model config and hardware (use prefill_req as source)
    model_cfg = load_model_config(prefill_req.model_id)
    hardware_cfg = load_hardware_config(prefill_req.hardware_id)

    # Compute metrics for prefill phase
    prefill_metrics = compute_phase_metrics(prefill_req, Phase.PREFILL)
    
    # Compute metrics for decode phase
    decode_metrics = compute_phase_metrics(decode_req, Phase.DECODE)

    # Aggregate system-level metrics
    max_packages = max(prefill_metrics["max_packages"], decode_metrics["max_packages"])
    
    # Use prefill memory totals for weights/activations (same for both phases)
    # BUT use decode KV cache since that determines decode bandwidth requirements!
    total_weight_memory_gb = prefill_metrics["total_weight_memory_gb"]
    total_activation_memory_gb = prefill_metrics["total_activation_memory_gb"]
    total_kv_cache_gb = decode_metrics["total_kv_cache_gb"]  # Decode KV scales with context!

    # Derive latencies (overhead now applied per-layer in base.py)
    ttft_ms = prefill_metrics["wall_clock_time_ms"]
    tpot_ms = decode_metrics["wall_clock_time_ms"]
    
    # TPS/User is the per-user token generation rate (independent of batch size)
    tps_user = 1000.0 / tpot_ms if tpot_ms > 0 else 0.0
    
    # System throughput is the total tokens/sec across all users in the batch
    # Use decode batch size since throughput is determined by decode phase
    throughput_tokens_s = tps_user * decode_req.batch_size
    
    total_latency_ms = ttft_ms + (tpot_ms * decode_req.seq_len)

    # Bottleneck analysis removed - wall_clock_time includes phase-aware overlap model
    bottleneck = "hardware-modeled"

    # Memory per package and capacity check
    memory_per_package_gb = (
        total_weight_memory_gb + total_activation_memory_gb + total_kv_cache_gb
    ) / max_packages if max_packages > 0 else 0.0
    
    hbm_memory = next((m for m in hardware_cfg['memory_per_package'] if 'HBM' in m['type']), hardware_cfg['memory_per_package'][0])
    hw_capacity_gb = hbm_memory['capacity_GB']
    fits_on_hardware = memory_per_package_gb <= hw_capacity_gb
    
    # Power and TCO metrics (scaled by number of packages)
    power_kw = hardware_cfg.get('power_kw', 0.0) * max_packages
    tco_sec_usd = hardware_cfg.get('tco_sec_usd', 0.0) * max_packages
    
    # MFU placeholder (TODO: compute at layer level)
    mfu = 0.37

    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tps_user": tps_user,
        "throughput_tokens_s": throughput_tokens_s,
        "total_latency_ms": total_latency_ms,
        "memory": {
            "weight_memory_gb": total_weight_memory_gb,
            "activation_memory_gb": total_activation_memory_gb,
            "kv_cache_gb": total_kv_cache_gb,
            "memory_per_package_gb": memory_per_package_gb,
            "hw_capacity_gb": hw_capacity_gb,
        },
        "system": {
            "num_packages": max_packages,
            "power_kw": power_kw,
            "mfu": mfu,
            "tco_sec_usd": tco_sec_usd,
            "bottleneck": bottleneck,
            "fits_on_hardware": fits_on_hardware,
        },
        "debug": {
            "decode_compute_ms": decode_metrics["total_compute_time_ms"],
            "decode_load_ms": decode_metrics["total_load_time_ms"],
            "prefill_compute_ms": prefill_metrics["total_compute_time_ms"],
            "prefill_load_ms": prefill_metrics["total_load_time_ms"],
            "decode_kv_cache_gb": decode_metrics["total_kv_cache_gb"],
            "prefill_kv_cache_gb": prefill_metrics["total_kv_cache_gb"],
            "decode_context_len": decode_req.context_len,
            "decode_seq_len": decode_req.seq_len,
            "prefill_seq_len": prefill_req.seq_len,
            "prefill_layer_details": prefill_metrics.get("layer_debug_details"),
            "decode_layer_details": decode_metrics.get("layer_debug_details"),
        },
    }

