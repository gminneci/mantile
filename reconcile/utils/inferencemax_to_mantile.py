#!/usr/bin/env python3
"""
Configuration Converter: InferenceMAX to Mantile API

Converts InferenceMAX benchmark CSV rows into Mantile API request format.

Key Mappings:
- B200 → nvidia_nvl72_rack hardware config
- System-level TP/EP → layer-level parallelism configs
- Concurrency → batch_size
- FP4 → nvfp4 dtype
"""

import json
from pathlib import Path
from typing import Dict, Any


# InferenceMAX to Mantile precision mapping
PRECISION_MAP = {
    'fp4': 'nvfp4',
    'fp8': 'nvfp8',
    'fp16': 'bf16',
    'bf16': 'bf16'
}


def load_model_config(model_id: str) -> Dict[str, Any]:
    """Load model configuration to get layer types."""
    config_path = Path(__file__).parent.parent.parent / "backend" / "data" / "model_configs" / f"{model_id}.json"
    with open(config_path) as f:
        return json.load(f)


def inferencemax_row_to_mantile_request(row: Dict[str, Any], model_id: str = "openai_GPT-OSS-120B") -> Dict[str, Any]:
    """
    Convert InferenceMAX CSV row to Mantile API request.
    
    Args:
        row: Dictionary with InferenceMAX columns
        model_id: Model configuration name (e.g., 'openai_GPT-OSS-120B')
        
    Returns:
        Dictionary matching Mantile SystemMetricsRequest format
    """
    # Hardware mapping
    gpu_model = row['gpu_model']
    tp = int(row['tensor_parallel'])
    ep = int(row['expert_parallel'])
    
    if gpu_model.startswith('b200'):
        hardware_id = 'nvidia_nvl72_rack'
    else:
        raise ValueError(f"Unsupported GPU model for MVP: {gpu_model}")
    
    # Load model config to get layer types
    model_cfg = load_model_config(model_id)
    
    batch_size = int(row['concurrency'])
    
    # Precision mapping
    dtype = PRECISION_MAP.get(row['precision'], 'nvfp4')
    
    # Build layer configurations
    # Apply TP to all layers, EP only to MoE layers
    layers = {}
    for layer_type in model_cfg.get('layer_types', []):
        layer_name = layer_type['name']
        layer_class = layer_type['class']
        
        if 'MoE' in layer_class:
            # MoE layer: apply both TP and EP
            layers[layer_name] = {
                'tp': tp,
                'ep': ep,
                'dtype': dtype
            }
        else:
            # Regular layer (attention, etc): apply only TP
            layers[layer_name] = {
                'tp': tp,
                'dtype': dtype
            }
    
    # Prefill phase: full input sequence
    prefill_req = {
        'model_id': model_id,
        'hardware_id': hardware_id,
        'batch_size': batch_size,
        'seq_len': int(row['input_seq_len']),
        'layers': layers
    }
    
    # Decode phase: seq_len represents past context length (KV cache size)
    # Each new token must attend to all past tokens in the KV cache
    decode_req = {
        'model_id': model_id,
        'hardware_id': hardware_id,
        'batch_size': batch_size,
        'seq_len': int(row['input_seq_len']),  # Past context = prompt length
        'layers': layers
    }
    
    return {
        'prefill_req': prefill_req,
        'decode_req': decode_req
    }


def test_converter():
    """Test the converter with a sample row."""
    sample_row = {
        'gpu_model': 'b200',
        'framework': 'vllm',
        'precision': 'fp4',
        'input_seq_len': 1024,
        'output_seq_len': 1024,
        'tensor_parallel': 2,
        'expert_parallel': 1,
        'concurrency': 16
    }
    
    print("Sample InferenceMAX row:")
    print(json.dumps(sample_row, indent=2))
    
    request = inferencemax_row_to_mantile_request(sample_row)
    
    print("\nConverted Mantile API request:")
    print(json.dumps(request, indent=2))


if __name__ == '__main__':
    test_converter()
