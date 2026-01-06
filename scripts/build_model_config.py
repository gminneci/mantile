#!/usr/bin/env python3
"""
Build Model Configuration Script

Generates a JSON configuration file for a transformer model by:
1. Fetching model config from HuggingFace
2. Building the ModelIR
3. Validating parameter counts
4. Saving to backend/data/model_configs/

Usage:
    python scripts/build_model_config.py <hf_model_id> [--model-id <custom_id>]
    
Examples:
    python scripts/build_model_config.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model-id tinyllama_1.1b
    python scripts/build_model_config.py meta-llama/Llama-3.3-70B-Instruct --model-id llama_3.3_70b
"""

import sys
import json
import argparse
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ir_builder import build_model_ir
from backend.models import ModelIR


def calculate_total_params(model_ir: ModelIR) -> int:
    """Calculate total parameter count from all layers."""
    total = 0
    for layer in model_ir.layers:
        total += layer.parameter_count
    
    # Add embedding parameters (vocab_size * hidden_size)
    # Note: Some models tie embeddings, but we count them here
    total += model_ir.vocab_size * model_ir.hidden_size
    
    return total


def format_params(params: int) -> str:
    """Format parameter count in human-readable form."""
    if params >= 1e9:
        return f"{params / 1e9:.1f}B"
    elif params >= 1e6:
        return f"{params / 1e6:.1f}M"
    else:
        return f"{params:,}"


def serialize_model_ir(model_ir: ModelIR, hf_model_id: str, model_id: str) -> dict:
    """Convert ModelIR to JSON-serializable dict."""
    
    # Serialize layers
    layers_data = []
    for layer in model_ir.layers:
        layer_dict = {
            "name": layer.name,
            "layer_idx": layer.layer_idx,
            "module_type": layer.module_type,
            "input_dim": layer.input_dim,
            "output_dim": layer.output_dim,
            "parameter_count": layer.parameter_count,
        }
        
        # Add optional fields if present
        if layer.num_heads is not None:
            layer_dict["num_heads"] = layer.num_heads
        if layer.head_dim is not None:
            layer_dict["head_dim"] = layer.head_dim
        if layer.kv_heads is not None:
            layer_dict["kv_heads"] = layer.kv_heads
        if layer.hidden_dim is not None:
            layer_dict["hidden_dim"] = layer.hidden_dim
        
        layers_data.append(layer_dict)
    
    # Calculate total params
    total_params = calculate_total_params(model_ir)
    
    # Build config dict
    config = {
        "model_id": model_id,
        "hf_model_id": hf_model_id,
        "name": hf_model_id.split("/")[-1],
        "hidden_size": model_ir.hidden_size,
        "num_layers": model_ir.num_layers,
        "vocab_size": model_ir.vocab_size,
        "total_params": total_params,
        "total_params_formatted": format_params(total_params),
        "layers": layers_data,
        "validated": False,
        "validation_notes": "Generated automatically - requires manual validation"
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Build a model configuration JSON from HuggingFace model"
    )
    parser.add_argument(
        "hf_model_id",
        help="HuggingFace model ID (e.g., meta-llama/Llama-3.3-70B-Instruct)"
    )
    parser.add_argument(
        "--model-id",
        help="Custom model ID for filename (e.g., llama_3.3_70b)",
        default=None
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for config file",
        default=None
    )
    
    args = parser.parse_args()
    
    # Determine model_id for filename
    if args.model_id:
        model_id = args.model_id
    else:
        # Auto-generate from HF model ID
        model_id = args.hf_model_id.split("/")[-1].lower().replace("-", "_")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "backend" / "data" / "model_configs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_id}.json"
    
    print(f"🔍 Building model IR from HuggingFace: {args.hf_model_id}")
    print(f"   Model ID: {model_id}")
    print(f"   Output: {output_path}")
    print()
    
    try:
        # Build ModelIR
        model_ir = build_model_ir(args.hf_model_id)
        
        # Calculate metrics
        total_params = calculate_total_params(model_ir)
        
        print(f"✅ Model IR built successfully!")
        print(f"   Hidden size: {model_ir.hidden_size}")
        print(f"   Layers: {model_ir.num_layers}")
        print(f"   Vocab size: {model_ir.vocab_size:,}")
        print(f"   Total params: {format_params(total_params)} ({total_params:,})")
        print()
        
        # Count layer types
        layer_types = {}
        for layer in model_ir.layers:
            layer_types[layer.module_type] = layer_types.get(layer.module_type, 0) + 1
        
        print("📊 Layer breakdown:")
        for layer_type, count in sorted(layer_types.items()):
            print(f"   {layer_type}: {count}")
        print()
        
        # Serialize to JSON
        config = serialize_model_ir(model_ir, args.hf_model_id, model_id)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Config saved to: {output_path}")
        print()
        print("⚠️  NEXT STEPS:")
        print("   1. Review the JSON file for correctness")
        print("   2. Verify parameter count matches expected value")
        print("   3. Update 'validated' field to true")
        print("   4. Add validation notes")
        print("   5. Commit the config file to the repository")
        print()
        
    except Exception as e:
        print(f"❌ Error building model config: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
