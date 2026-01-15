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

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PhaseMetricsRequest(BaseModel):
    """Stateless request for a single phase (prefill or decode) - includes all necessary context."""
    model_id: str
    hardware_id: str
    batch_size: int
    seq_len: int
    # Layer configurations with parallelism and dtype
    layers: Dict[str, Dict[str, Any]] = {}


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
    compute_time_ms = 0.0
    memory_time_ms = 0.0
    max_packages = 1

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

        # Hardware characteristics per package
        hbm_memory = next((m for m in hardware_cfg['memory'] if 'HBM' in m['type']), hardware_cfg['memory'][0])
        hbm_bw_per_package = hbm_memory['bandwidth_gbps']
        assert dtype_enum.value in hardware_cfg['compute_per_package_GFlops'], f"Hardware missing compute spec for dtype {dtype_enum.value}"
        peak_gflops_per_package = hardware_cfg['compute_per_package_GFlops'][dtype_enum.value]

        # Compute metrics for this phase
        m = layer.compute_metrics(
            batch_size=phase_req.batch_size,
            seq_len=phase_req.seq_len,
            phase=phase,
        )
        max_packages = max(max_packages, m.num_packages)
        total_weight_memory_gb += (m.weight_memory_per_package * num_instances) / 1e9
        total_activation_memory_gb += (m.activation_memory_per_package * num_instances) / 1e9
        total_kv_cache_gb += (m.kv_cache_per_package * num_instances) / 1e9

        # Total compute available = peak per package * number of packages in use
        total_peak_gflops = peak_gflops_per_package * m.num_packages
        compute_time_ms += ((m.flops_per_package * num_instances) / 1e9) / total_peak_gflops * 1000
        memory_time_ms += ((m.weight_memory_per_package * num_instances) / 1e9) / hbm_bw_per_package * 1000

    return {
        "total_weight_memory_gb": total_weight_memory_gb,
        "total_activation_memory_gb": total_activation_memory_gb,
        "total_kv_cache_gb": total_kv_cache_gb,
        "compute_time_ms": compute_time_ms,
        "memory_time_ms": memory_time_ms,
        "max_packages": max_packages,
    }


@app.post("/config/system-metrics")
def compute_system_metrics(prefill_req: PhaseMetricsRequest, decode_req: PhaseMetricsRequest):
    """
    Stateless endpoint: Compute full system-level metrics.
    Takes separate requests for prefill and decode phases.
    
    Args:
        prefill_req: Configuration for prefill phase (prompt processing, seq_len = input prompt length)
        decode_req: Configuration for decode phase (token generation, seq_len = number of output tokens)
    """
    # Load pre-validated model config and hardware (use prefill_req as source)
    model_cfg = load_model_config(prefill_req.model_id)
    hardware_cfg = load_hardware_config(prefill_req.hardware_id)

    # Compute metrics for prefill phase
    prefill_metrics = compute_phase_metrics(prefill_req, Phase.PREFILL)
    
    # Compute metrics for decode phase
    decode_metrics = compute_phase_metrics(decode_req, Phase.DECODE)

    # Aggregate system-level metrics
    max_packages = max(prefill_metrics["max_packages"], decode_metrics["max_packages"])
    
    # Use prefill memory totals (weights/activations/kv are same for both phases)
    total_weight_memory_gb = prefill_metrics["total_weight_memory_gb"]
    total_activation_memory_gb = prefill_metrics["total_activation_memory_gb"]
    total_kv_cache_gb = prefill_metrics["total_kv_cache_gb"]

    # Derive latencies
    ttft_ms = max(prefill_metrics["compute_time_ms"], prefill_metrics["memory_time_ms"])
    tpot_ms = max(decode_metrics["compute_time_ms"], decode_metrics["memory_time_ms"])
    
    # TPS/User is the per-user token generation rate (independent of batch size)
    tps_user = 1000.0 / tpot_ms if tpot_ms > 0 else 0.0
    
    # System throughput is the total tokens/sec across all users in the batch
    # Use decode batch size since throughput is determined by decode phase
    throughput_tokens_s = tps_user * decode_req.batch_size
    
    total_latency_ms = ttft_ms + (tpot_ms * decode_req.seq_len)

    # Bottleneck analysis
    if prefill_metrics["compute_time_ms"] > prefill_metrics["memory_time_ms"] * 1.2:
        bottleneck = "compute"
    elif prefill_metrics["memory_time_ms"] > prefill_metrics["compute_time_ms"] * 1.2:
        bottleneck = "memory"
    else:
        bottleneck = "balanced"

    # Memory per package and capacity check
    memory_per_package_gb = (
        total_weight_memory_gb + total_activation_memory_gb + total_kv_cache_gb
    ) / max_packages if max_packages > 0 else 0.0
    
    hbm_memory = next((m for m in hardware_cfg['memory'] if 'HBM' in m['type']), hardware_cfg['memory'][0])
    hw_capacity_gb = hbm_memory['capacity_gb']
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
    }

