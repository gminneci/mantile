# Offline Model Configuration System

## Overview

The offline model config system allows you to pre-generate and validate model configurations instead of building them at runtime using the HuggingFace API. This provides:

- **Faster API responses**: No runtime HuggingFace API calls
- **Better reliability**: No dependency on external API availability
- **Manual validation**: Review and validate configs before production use
- **Transparent metrics**: Pre-computed parameter counts and architecture details

## Architecture

The system follows the same pattern as `hardware_library.py`:

```
backend/
  data/
    model_configs/          # JSON config files
      tinyllama_1.1b.json
      llama_3.3_70b.json
  model_library.py          # Load and list configs
scripts/
  build_model_config.py     # Offline IR generation tool
```

## Usage

### Listing Available Models

```python
from backend.model_library import list_available_models

models = list_available_models()
# Returns: ['llama_3.3_70b', 'tinyllama_1.1b']
```

### Loading a Model Config

```python
from backend.model_library import load_model_config

model_ir = load_model_config('tinyllama_1.1b')
# Returns: ModelIR object with pre-validated configuration
```

### Getting Model Metadata

```python
from backend.model_library import get_model_metadata

metadata = get_model_metadata('llama_3.3_70b')
# Returns: {
#   'model_id': 'llama_3.3_70b',
#   'name': 'Llama-3.3-70B-Instruct',
#   'architecture': 'LlamaForCausalLM',
#   'total_params': 69502369792,
#   'total_params_formatted': '69.5B',
#   'validated': True,
#   'validation_notes': 'Llama 3.3 70B confirmed...',
#   ...
# }
```

## Adding New Models

### Step 1: Generate Config

Use the `build_model_config.py` script to generate a new model config from HuggingFace:

```bash
python3 scripts/build_model_config.py meta-llama/Llama-3.3-70B-Instruct --model-id llama_3.3_70b
```

This will:
1. Download model architecture from HuggingFace
2. Build the complete IR with all layer specifications
3. Calculate total parameter count
4. Save to `backend/data/model_configs/llama_3.3_70b.json`
5. Set `validated: false` by default

### Step 2: Manual Validation

Review the generated JSON file:

```json
{
  "model_id": "llama_3.3_70b",
  "name": "Llama-3.3-70B-Instruct",
  "hidden_size": 8192,
  "num_layers": 80,
  "vocab_size": 128256,
  "total_params": 69502369792,
  "total_params_formatted": "69.5B",
  "architecture": "LlamaForCausalLM",
  "validated": false,
  "validation_notes": "Generated automatically - requires manual validation",
  "layers": [...]
}
```

Verify:
- Parameter counts match expected values
- Architecture is correctly detected (GQA, SwiGLU, etc.)
- Model version is correct (e.g., Llama 3.3 not 3.0)

### Step 3: Mark as Validated

Once verified, update the config:

```json
{
  "validated": true,
  "validation_notes": "Llama 3.3 70B confirmed (69.5B params), GQA with 8 KV heads, SwiGLU MLP (28672 intermediate)"
}
```

### Step 4: Commit

```bash
git add backend/data/model_configs/llama_3.3_70b.json
git commit -m "feat: add validated config for Llama 3.3 70B"
git push
```

## API Endpoints

### List Models

```bash
GET /models
```

Returns list of all available pre-validated models with metadata.

### Get Model Details

```bash
GET /models/{model_id}
```

Returns metadata for a specific model.

### Deployment Estimation

```bash
POST /deployment/estimate
{
  "model_id": "llama_3.3_70b",
  "hardware_config": "nvidia_nvl72_rack",
  "batch_size": 1,
  "input_seq": 2048,
  "output_seq": 512
}
```

Uses pre-validated model config for estimation.

## Available Models

### TinyLlama 1.1B
- **Model ID**: `tinyllama_1.1b`
- **HF Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Parameters**: 1.0B (1,034,465,280)
- **Architecture**: LlamaForCausalLM with GQA (4 KV heads), SwiGLU MLP
- **Validated**: ✅ Yes

### Llama 3.3 70B
- **Model ID**: `llama_3.3_70b`
- **HF Model**: `meta-llama/Llama-3.3-70B-Instruct`
- **Parameters**: 69.5B (69,502,369,792)
- **Architecture**: LlamaForCausalLM with GQA (8 KV heads), SwiGLU MLP (28672 intermediate)
- **Validated**: ✅ Yes

## Migration from ir_builder

All endpoints have been migrated from `build_model_ir` to `load_model_config`:

- ✅ `/api/layers` - Uses load_model_config
- ✅ `/config/layer-metrics` - Uses load_model_config  
- ✅ `/config/system-metrics` - Uses load_model_config
- ✅ `/deployment/estimate` - Uses load_model_config via ConfigurationService
- ✅ `/estimate` - Uses load_model_config

The `ir_builder.py` module is no longer used in the API but remains available for generating new configs.
