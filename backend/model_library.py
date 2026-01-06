"""
Model Configuration Library

Provides access to pre-validated model configurations stored as JSON files.
Similar pattern to hardware_library.py but for model IRs.

All model configs are generated offline using scripts/build_model_config.py
and manually validated before being committed to the repository.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from .models import ModelIR, LayerSpecs


def get_model_configs_dir() -> Path:
    """Get the absolute path to the model_configs directory."""
    backend_dir = Path(__file__).parent
    return backend_dir / "data" / "model_configs"


def list_available_models() -> List[str]:
    """
    List all available model configurations.
    
    Returns:
        List of model IDs (e.g., ["tinyllama_1.1b", "llama_3.3_70b"])
    """
    configs_dir = get_model_configs_dir()
    if not configs_dir.exists():
        return []
    
    models = []
    for config_file in configs_dir.glob("*.json"):
        model_id = config_file.stem
        models.append(model_id)
    
    return sorted(models)


def load_model_config(model_id: str) -> ModelIR:
    """
    Load a pre-validated model configuration from JSON.
    
    Args:
        model_id: Model identifier (e.g., "tinyllama_1.1b", "llama_3.3_70b")
        
    Returns:
        ModelIR object reconstructed from the JSON config
        
    Raises:
        FileNotFoundError: If the model config doesn't exist
        ValueError: If the config is invalid
    """
    configs_dir = get_model_configs_dir()
    config_path = configs_dir / f"{model_id}.json"
    
    if not config_path.exists():
        available = list_available_models()
        raise FileNotFoundError(
            f"Model config '{model_id}' not found. Available models: {available}"
        )
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    # Check if this is new format (layer_types) or old format (layers)
    if "layer_types" in data:
        # New format: expand layer_types into full layers list
        layers = []
        layer_idx = 0
        
        for layer_type_data in data["layer_types"]:
            module_type = layer_type_data["name"]
            count = layer_type_data["count"]
            specs = layer_type_data["specs"]
            
            # Create count instances of this layer type
            for i in range(count):
                layer_spec = LayerSpecs(
                    name=f"layer_{layer_idx}_{module_type}",
                    layer_idx=layer_idx,
                    module_type=module_type,
                    input_dim=specs["input_dim"],
                    output_dim=specs["output_dim"],
                    parameter_count=specs["parameter_count"],
                    num_heads=specs.get("num_heads"),
                    head_dim=specs.get("head_dim"),
                    kv_heads=specs.get("num_kv_heads"),
                    hidden_dim=specs.get("intermediate_size")
                )
                layers.append(layer_spec)
                layer_idx += 1
    else:
        # Old format: direct layers array (backward compatibility)
        layers = []
        for layer_data in data.get("layers", []):
            layer_spec = LayerSpecs(
                name=layer_data["name"],
                layer_idx=layer_data["layer_idx"],
                module_type=layer_data["module_type"],
                input_dim=layer_data["input_dim"],
                output_dim=layer_data["output_dim"],
                parameter_count=layer_data["parameter_count"],
                num_heads=layer_data.get("num_heads"),
                head_dim=layer_data.get("head_dim"),
                kv_heads=layer_data.get("kv_heads"),
                hidden_dim=layer_data.get("hidden_dim")
            )
            layers.append(layer_spec)
    
    # Construct ModelIR
    model_ir = ModelIR(
        name=data["model_id"],
        hidden_size=data["hidden_size"],
        num_layers=data["num_layers"],
        vocab_size=data["vocab_size"],
        layers=layers
    )
    
    return model_ir


def get_model_metadata(model_id: str) -> Dict[str, Any]:
    """
    Get metadata about a model without loading the full IR.
    
    Returns:
        Dict with keys: name, architecture, total_params, validated, etc.
    """
    configs_dir = get_model_configs_dir()
    config_path = configs_dir / f"{model_id}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model config '{model_id}' not found")
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    return {
        "model_id": data["model_id"],
        "name": data.get("name", data["model_id"]),
        "architecture": data.get("architecture", "unknown"),
        "hidden_size": data["hidden_size"],
        "num_layers": data["num_layers"],
        "vocab_size": data["vocab_size"],
        "total_params": data.get("total_params", 0),
        "total_params_formatted": data.get("total_params_formatted", ""),
        "validated": data.get("validated", False),
        "validation_notes": data.get("validation_notes", ""),
        "hf_model_id": data.get("hf_model_id", "")
    }
