#!/usr/bin/env python3
"""
Test Llama 3.3 70B model construction with actual layer implementations.
"""

from backend.layers import (
    EmbeddingLayer,
    GroupedQueryAttentionLayer,
    GatedMLPLayer,
    NormLayer,
    Phase,
    DataType
)

def build_llama_33_70b():
    """
    Build Llama 3.3 70B model with actual layer instances.
    
    Architecture:
        - 80 transformer layers
        - 8192 hidden size
        - 64 query heads, 8 KV heads (GQA 8:1)
        - 128 head dimension
        - 28672 intermediate size (SwiGLU)
        - 128256 vocab size
        - RMSNorm
    """
    # Model config
    config = {
        "hidden_size": 8192,
        "num_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 28672,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
    }
    
    layers = []
    
    # 1. Embedding layer
    embedding = EmbeddingLayer(
        name="embedding",
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        parallelism={}  # Will configure per run
    )
    layers.append(embedding)
    
    # 2. Transformer layers
    for i in range(config["num_layers"]):
        # Input norm (RMSNorm)
        input_norm = NormLayer(
            name=f"layer_{i}_input_norm",
            layer_idx=i,
            hidden_size=config["hidden_size"],
            has_bias=False,  # RMSNorm has no bias
            parallelism={}
        )
        layers.append(input_norm)
        
        # Attention (GQA)
        attention = GroupedQueryAttentionLayer(
            name=f"layer_{i}_attention",
            layer_idx=i,
            hidden_size=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            parallelism={}
        )
        layers.append(attention)
        
        # Post-attention norm (RMSNorm)
        post_attn_norm = NormLayer(
            name=f"layer_{i}_post_attn_norm",
            layer_idx=i,
            hidden_size=config["hidden_size"],
            has_bias=False,
            parallelism={}
        )
        layers.append(post_attn_norm)
        
        # MLP (SwiGLU - 3 projections)
        mlp = GatedMLPLayer(
            name=f"layer_{i}_mlp",
            layer_idx=i,
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            parallelism={}
        )
        layers.append(mlp)
    
    # 3. Final norm
    final_norm = NormLayer(
        name="final_norm",
        layer_idx=config["num_layers"],
        hidden_size=config["hidden_size"],
        has_bias=False,
        parallelism={}
    )
    layers.append(final_norm)
    
    return {
        "config": config,
        "layers": layers,
        "name": "Llama-3.3-70B"
    }

def test_single_chip_prefill():
    """Test single chip prefill metrics"""
    model = build_llama_33_70b()
    config = model["config"]
    layers = model["layers"]
    
    # Test config
    batch_size = 1
    seq_len = 2048
    phase = Phase.PREFILL
    dtype = DataType.BF16
    
    print(f"=== {model['name']} - Single Chip Prefill ===")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Phase: {phase.value}")
    print()
    
    total_flops = 0
    total_weight_memory = 0
    total_activation_memory = 0
    total_kv_cache = 0
    
    # Embedding
    embedding = layers[0]
    emb_weight = embedding.compute_weight_memory(dtype)
    emb_act = embedding.compute_activation_memory(batch_size, seq_len, phase, dtype)
    total_weight_memory += emb_weight
    total_activation_memory += emb_act
    
    print(f"Embedding:")
    print(f"  Weight: {emb_weight / 1e9:.2f} GB")
    print(f"  Activation: {emb_act / 1e6:.2f} MB")
    print()
    
    # Count layers by type
    attn_layers = [l for l in layers if "attention" in l.name]
    mlp_layers = [l for l in layers if "mlp" in l.name]
    norm_layers = [l for l in layers if "norm" in l.name]
    
    print(f"Layer counts:")
    print(f"  Attention: {len(attn_layers)}")
    print(f"  MLP: {len(mlp_layers)}")
    print(f"  Norm: {len(norm_layers)}")
    print()
    
    # Sample one transformer layer
    print("Sample Transformer Layer (layer_0):")
    for layer in layers[1:6]:  # First transformer layer components
        flops = layer.compute_flops(batch_size, seq_len, phase, dtype)
        weight = layer.compute_weight_memory(dtype)
        activation = layer.compute_activation_memory(batch_size, seq_len, phase, dtype)
        kv_cache = layer.compute_kv_cache(batch_size, seq_len, dtype) if hasattr(layer, 'compute_kv_cache') else 0
        
        print(f"  {layer.name}:")
        print(f"    FLOPs: {flops / 1e12:.3f} TFLOPs")
        print(f"    Weight: {weight / 1e6:.2f} MB")
        print(f"    Activation: {activation / 1e6:.2f} MB")
        if kv_cache > 0:
            print(f"    KV Cache: {kv_cache / 1e6:.2f} MB")
    
    # Compute totals
    for layer in layers:
        flops = layer.compute_flops(batch_size, seq_len, phase, dtype)
        weight = layer.compute_weight_memory(dtype)
        activation = layer.compute_activation_memory(batch_size, seq_len, phase, dtype)
        kv_cache = layer.compute_kv_cache(batch_size, seq_len, dtype) if hasattr(layer, 'compute_kv_cache') else 0
        
        total_flops += flops
        total_weight_memory += weight
        total_activation_memory += activation
        total_kv_cache += kv_cache
    
    print()
    print("=== Total Model Metrics ===")
    print(f"Total FLOPs: {total_flops / 1e12:.2f} TFLOPs")
    print(f"Total Weight Memory: {total_weight_memory / 1e9:.2f} GB")
    print(f"Total Activation Memory: {total_activation_memory / 1e9:.2f} GB")
    print(f"Total KV Cache: {total_kv_cache / 1e9:.2f} GB")
    print(f"Total Memory: {(total_weight_memory + total_activation_memory + total_kv_cache) / 1e9:.2f} GB")
    print()
    
    # Rough param count check
    total_params = total_weight_memory / dtype.bytes_per_element
    print(f"Total Parameters: {total_params / 1e9:.1f}B")
    print()

if __name__ == "__main__":
    test_single_chip_prefill()
