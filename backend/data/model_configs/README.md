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

- **meta-llama_Llama-3.1-405B-Instruct.json**: Meta Llama 3.1 405B Instruct
  - 126 layers, GQA (8 KV heads), SwiGLU MLP (53248 intermediate)
  - 405.9B parameters

- **meta-llama_Llama-3.3-70B-Instruct.json**: Meta Llama 3.3 70B Instruct
  - 80 layers, GQA (8 KV heads), SwiGLU MLP (28672 intermediate)
  - 69.5B parameters

- **TinyLlama_TinyLlama-1.1B-Chat-v1.0.json**: TinyLlama 1.1B Chat
  - 22 layers, GQA (4 KV heads), SwiGLU MLP (5632 intermediate)
  - 1.1B parameters

- **mistralai_Mistral-7B-v0.1.json**: Mistral 7B v0.1
  - 32 layers, GQA (8 KV heads), SwiGLU MLP (14336 intermediate)
  - 7.2B parameters, sliding window attention (4096)

## Usage

Model configs are loaded via the FastAPI endpoint:

```python
# GET /models/{model_id}
# Returns the JSON config as a dict

# Example: Load Llama 3.3 70B
model_cfg = load_model_config("meta-llama_Llama-3.3-70B-Instruct")

# Access layer specifications:
for layer_type in model_cfg['layer_types']:
    name = layer_type['name']
    class_name = layer_type['class']
    count = layer_type['count']
    specs = layer_type['specs']
```

## Adding New Configurations

1. Create a new JSON file in this directory
2. **Naming convention**: Use the HuggingFace model ID with `/` replaced by `_`
   - Example: `mistralai/Mistral-7B-v0.1` → `mistralai_Mistral-7B-v0.1.json`
   - Example: `meta-llama/Llama-3.3-70B-Instruct` → `meta-llama_Llama-3.3-70B-Instruct.json`
3. Follow the format above (see existing configs for reference)
4. Set `model_id` to the filename without `.json` extension
5. Set `hf_model_id` to the original HuggingFace model ID (with `/`)
6. Ensure `layer_types[].name` values are unique within the config
7. Match `class` names exactly to layer implementations in `backend/layers/`
8. Restart the backend to load the new config
9. Test via `GET /models/{model_id}` endpoint

## Validation

The backend validates:
- Unique layer names within a config
- Layer class names resolve to valid implementations
- Required specs are present for each layer type

Optional validation:
- Parameter count matches expected values
- Architecture matches published model cards
