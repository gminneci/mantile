from transformers import AutoConfig
from .models import ModelIR, LayerSpecs
import os

def build_model_ir(model_id: str, token: str = None) -> ModelIR:
    """
    Generic IR builder that works for any transformer model.
    Derives all information from config with no defaults.
    """
    # Use provided token or check environment variable
    hf_token = token or os.getenv("HF_TOKEN")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    
    # Extract required fields - fail explicitly if missing
    try:
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        vocab_size = config.vocab_size
        num_heads = config.num_attention_heads
    except AttributeError as e:
        raise ValueError(f"Model config missing required field: {e}")
    
    # Optional fields with explicit fallback logic
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)  # MHA if not specified
    intermediate_size = getattr(config, "intermediate_size", None)
    
    # Infer intermediate size if not in config
    if intermediate_size is None:
        # Common pattern: 4x hidden for dense, but check for SwiGLU/Gated patterns
        # Most modern models use explicit intermediate_size, so this should rarely trigger
        raise ValueError(f"Model config missing 'intermediate_size'. Cannot infer MLP dimensions.")
    
    # Detect architecture patterns from config
    architecture = getattr(config, "architectures", [None])[0] if hasattr(config, "architectures") else "unknown"
    model_type = getattr(config, "model_type", "unknown")
    
    # Determine if gated MLP (e.g., LLaMA SwiGLU has 3 projections, standard has 2)
    # Check for activation function hints
    hidden_act = getattr(config, "hidden_act", None)
    is_gated_mlp = hidden_act in ["silu", "swish", "gelu_new"] or "llama" in model_type.lower()
    
    layers = []
    
    # Build layer specs for each transformer layer
    for i in range(num_layers):
        # 1. Attention Block
        # Standard: Q, K, V projections + output projection
        # Q: (hidden_size, num_heads * head_dim) ≈ (hidden, hidden)
        # K, V: (hidden_size, num_kv_heads * head_dim)
        # O: (hidden_size, hidden_size)
        head_dim = hidden_size // num_heads
        
        q_params = hidden_size * hidden_size
        k_params = hidden_size * (num_kv_heads * head_dim)
        v_params = hidden_size * (num_kv_heads * head_dim)
        o_params = hidden_size * hidden_size
        attn_params = q_params + k_params + v_params + o_params
        
        layers.append(LayerSpecs(
            name=f"layer_{i}_attn",
            layer_idx=i,
            module_type="attention",
            input_dim=hidden_size,
            output_dim=hidden_size,
            parameter_count=attn_params,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_heads=num_kv_heads
        ))
        
        # 2. Feedforward/MLP Block
        # Gated (LLaMA SwiGLU): gate_proj, up_proj, down_proj → 3 matrices
        # Standard: up_proj, down_proj → 2 matrices
        if is_gated_mlp:
            # gate: (hidden → intermediate), up: (hidden → intermediate), down: (intermediate → hidden)
            mlp_params = 2 * (hidden_size * intermediate_size) + (intermediate_size * hidden_size)
        else:
            # up: (hidden → intermediate), down: (intermediate → hidden)
            mlp_params = (hidden_size * intermediate_size) + (intermediate_size * hidden_size)
        
        layers.append(LayerSpecs(
            name=f"layer_{i}_mlp",
            layer_idx=i,
            module_type="feedforward",
            input_dim=hidden_size,
            output_dim=hidden_size,
            parameter_count=mlp_params,
            hidden_dim=intermediate_size
        ))
        
        # 3. Layer Normalization
        # RMSNorm/LayerNorm: single scale vector of size hidden_size
        layers.append(LayerSpecs(
            name=f"layer_{i}_norm",
            layer_idx=i,
            module_type="norm",
            input_dim=hidden_size,
            output_dim=hidden_size,
            parameter_count=hidden_size
        ))

    return ModelIR(
        name=model_id,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        layers=layers
    )


# Backward compatibility alias
def build_llama_ir(model_id: str) -> ModelIR:
    """Legacy function - use build_model_ir instead."""
    return build_model_ir(model_id)
