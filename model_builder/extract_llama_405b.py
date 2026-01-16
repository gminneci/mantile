
import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model_builder.utils import (
    get_model_config,
    inspect_model_structure,
    save_tensor_inspection,
    analyze_layer_structure,
    count_parameters,
    estimate_memory,
)

MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct" # Using Instruct version as it's common
OUTPUT_DIR = Path("model_builder/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print(f"--- Extracting info for {MODEL_ID} (Manual Override) ---")
    
    # Manually gathered parameters from public sources
    config = {
        "model_type": "llama",
        "hidden_size": 16384,
        "num_hidden_layers": 126,
        "intermediate_size": 53248,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "max_position_embeddings": 131072,
        "tie_word_embeddings": False
    }
    
    print("Saving manual config...")
    with open(OUTPUT_DIR / "llama_3.1_405b_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 2. Compute tensor shapes (using the utility function)
    print("Computing tensor shapes from config...")
    from model_builder.utils import compute_tensor_shapes_from_config
    tensors = compute_tensor_shapes_from_config(config)
    save_tensor_inspection(tensors, OUTPUT_DIR / "llama_3.1_405b_tensors.csv")
    
    # 3. Analyze layer structure
    print("Analyzing layer structure...")
    structure = analyze_layer_structure(tensors)
    with open(OUTPUT_DIR / "llama_3.1_405b_structure.json", "w") as f:
        json.dump(structure, f, indent=2)
        
    # 4. Count parameters
    print("Counting parameters...")
    params = count_parameters(tensors)
    print(f"Total parameters: {params['total_formatted']} ({params['total']:,})")
    
    # 5. Estimate memory
    print("Estimating memory...")
    memory = estimate_memory(tensors, dtype="float16")
    print(f"Memory (FP16): {memory['gb']:.2f} GB")
    
    print("--- Done ---")

if __name__ == "__main__":
    main()
