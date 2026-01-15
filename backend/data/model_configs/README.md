# Model Configuration Files

This directory contains JSON model specifications for various LLMs.

## Format

Each JSON file defines a model configuration with the following fields:

```json
{
  "model_id": "model_identifier",
  "hf_model_id": "huggingface/model-name",
  "name": "Human Readable Model Name",
  "hidden_size": 8192,
  "num_layers": 80,
  "vocab_size": 128256,
  "total_params": 69502369792,
  "total_params_formatted": "69.5B",
  "layer_types": [
    {
      "name": "attention",
      "class": "GroupedQueryAttentionLayer",  // or "AttentionLayer" for MHA
      "count": 80,
      "specs": {
        "layer_idx": 0,
        "hidden_size": 8192,
        "num_heads": 64,
        "num_kv_heads": 8,  // For GQA; omit for MHA
        "head_dim": 128,
        "parameter_count": 150994944  // Optional metadata
      }
    },
    {
      "name": "feedforward",
      "class": "GatedMLPLayer",  // or "MLPLayer" for standard 2-proj
      "count": 80,
      "specs": {
        "layer_idx": 0,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "parameter_count": 704643072  // Optional metadata
      }
    },
    {
      "name": "norm",
      "class": "NormLayer",
      "count": 80,
      "specs": {
        "layer_idx": 0,
        "hidden_size": 8192,
        "has_bias": false,
        "parameter_count": 8192  // Optional metadata
      }
    },
    {
      "name": "embedding",
      "class": "EmbeddingLayer",
      "count": 1,
      "specs": {
        "vocab_size": 128256,
        "hidden_size": 8192,
        "parameter_count": 1050673152  // Optional metadata
      }
    }
  ],
  "validated": true,
  "validation_notes": "Optional human-readable notes about validation"
}
```

## Layer Classes

Available layer implementations (from `backend/layers/`):

- **AttentionLayer**: Vanilla multi-head attention (MHA)
  - Required specs: `hidden_size`, `num_heads`, `head_dim`
  - Supported parallelism: `tensor_parallel`, `context_parallel`

- **GroupedQueryAttentionLayer**: Grouped query attention (GQA)
  - Required specs: `hidden_size`, `num_heads`, `num_kv_heads`, `head_dim`
  - Supported parallelism: `tensor_parallel`, `context_parallel`

- **MLPLayer**: Standard 2-projection MLP
  - Required specs: `hidden_size`, `intermediate_size`
  - Supported parallelism: `tensor_parallel`, `sequence_parallel`

- **GatedMLPLayer**: 3-projection gated MLP (SwiGLU)
  - Required specs: `hidden_size`, `intermediate_size`
  - Supported parallelism: `tensor_parallel`, `sequence_parallel`

- **NormLayer**: LayerNorm or RMSNorm
  - Required specs: `hidden_size`, `has_bias`
  - Typically replicated (no parallelism)

- **EmbeddingLayer**: Token embeddings
  - Required specs: `vocab_size`, `hidden_size`
  - Optional parallelism: `tensor_parallel` (for large vocabs)

## Available Configurations

- **llama_3.3_70b.json**: Meta Llama 3.3 70B Instruct
  - 80 layers, GQA (8 KV heads), SwiGLU MLP (28672 intermediate)
  - 69.5B parameters

- **tinyllama_1.1b.json**: TinyLlama 1.1B Chat
  - 22 layers, GQA (4 KV heads), SwiGLU MLP (5632 intermediate)
  - 1.1B parameters

## Usage

Model configs are loaded via the FastAPI endpoint:

```python
# GET /models/{model_id}
# Returns the JSON config as a dict

# Example: Load llama_3.3_70b
model_cfg = load_model_config("llama_3.3_70b")

# Access layer specifications:
for layer_type in model_cfg['layer_types']:
    name = layer_type['name']
    class_name = layer_type['class']
    count = layer_type['count']
    specs = layer_type['specs']
```

## Adding New Configurations

1. Create a new JSON file in this directory
2. Follow the format above (see existing configs for reference)
3. Use descriptive `model_id`: `{family}_{size}_{variant}` (e.g., `llama_3_8b`, `mistral_7b_instruct`)
4. Ensure `layer_types[].name` values are unique within the config
5. Match `class` names exactly to layer implementations in `backend/layers/`
6. Restart the backend to load the new config
7. Test via `GET /models/{model_id}` endpoint

## Validation

The backend validates:
- Unique layer names within a config
- Layer class names resolve to valid implementations
- Required specs are present for each layer type

Optional validation:
- Parameter count matches expected values
- Architecture matches published model cards
