# Stateless API Migration Guide

## Overview

The backend has been refactored to be completely stateless. The server no longer maintains session state or configuration. Each request must include all necessary context (model ID, hardware config, layer configurations, etc.).

## Benefits

1. **No State Management**: Server doesn't need to track sessions
2. **Multiple Configurations**: Frontend can query metrics for multiple configs simultaneously
3. **Scalability**: Stateless endpoints are easier to load balance and scale
4. **Comparison Mode**: Can compare different model/hardware/parallelism configs without conflicts

## New Stateless Endpoints

### 1. GET /api/layers?model_id={model_id}

Get layer information for a specific model without loading it into server state.

**Request:**
```
GET /api/layers?model_id=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**Response:**
```json
{
  "layers": [
    {
      "type": "attention",
      "count": 22,
      "available_parallelism": ["tensor_parallel", "context_parallel"],
      "available_dtypes": ["fp32", "fp16", "bf16", "fp8", "int8"]
    },
    {
      "type": "feedforward",
      "count": 22,
      "available_parallelism": ["tensor_parallel", "sequence_parallel"],
      "available_dtypes": ["fp32", "fp16", "bf16", "fp8", "int8"]
    }
  ]
}
```

### 2. POST /config/layer-metrics

Compute metrics for a single layer type with full context.

**Request:**
```json
{
  "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "hardware_config": "nvidia_gb200_single",
  "layer_type": "attention",
  "batch_size": 1,
  "seq_length": 2048,
  "dtype": "bf16",
  "tensor_parallel": 4,
  "context_parallel": 2,
  "sequence_parallel": 1
}
```

**Response:**
```json
{
  "layer_type": "attention",
  "num_instances": 22,
  "num_chips": 4,
  "parallelism": {
    "tensor_parallel": 4,
    "context_parallel": 2,
    "sequence_parallel": 1
  },
  "memory": {
    "weights_per_chip_gb": 0.5,
    "activation_per_chip_gb": 0.2,
    "kv_cache_per_chip_gb": 1.5,
    "total_weights_gb": 11.0,
    "total_activation_gb": 4.4,
    "total_kv_cache_gb": 33.0
  },
  "compute": {
    "flops_per_chip_tflops": 25.5,
    "total_flops_tflops": 561.0
  },
  "bottleneck": {
    "compute_pct": 45.0,
    "memory_pct": 50.0,
    "communication_pct": 5.0
  }
}
```

### 3. POST /config/system-metrics

Compute full system metrics with complete configuration.

**Request:**
```json
{
  "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "hardware_config": "nvidia_gb200_single",
  "batch_size": 1,
  "input_seq": 2048,
  "output_seq": 128,
  "layers": {
    "attention": {
      "tensor_parallel": 4,
      "context_parallel": 2,
      "sequence_parallel": 1,
      "dtype": "bf16"
    },
    "feedforward": {
      "tensor_parallel": 8,
      "context_parallel": 1,
      "sequence_parallel": 1,
      "dtype": "bf16"
    },
    "norm": {
      "tensor_parallel": 1,
      "context_parallel": 1,
      "sequence_parallel": 1,
      "dtype": "bf16"
    },
    "embedding": {
      "tensor_parallel": 1,
      "context_parallel": 1,
      "sequence_parallel": 1,
      "dtype": "bf16"
    }
  }
}
```

**Response:**
```json
{
  "ttft_ms": 45.2,
  "tpot_ms": 1.8,
  "throughput_tokens_s": 555.5,
  "total_latency_ms": 275.6,
  "memory": {
    "weight_memory_gb": 2.2,
    "activation_memory_gb": 0.8,
    "kv_cache_gb": 2.1,
    "total_memory_gb": 5.1,
    "memory_per_chip_gb": 1.275,
    "hw_capacity_gb": 192.0
  },
  "system": {
    "num_chips": 4,
    "bottleneck": "memory",
    "fits_on_hardware": true
  }
}
```

## Deprecated Endpoints

These endpoints still exist for backward compatibility but should not be used:

- `POST /config/load` - Load model and hardware (stateful)
- `POST /config/layer-parallelism` - Configure layer parallelism (stateful)
- `GET /config/layers` - Get layers from loaded model (stateful)

## Frontend Migration

### Before (Stateful):
```javascript
// Step 1: Load model
await axios.post('/config/load', {
  model_id: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
  hardware_config: 'nvidia_gb200_single'
});

// Step 2: Get layers
const layers = await axios.get('/config/layers');

// Step 3: Compute metrics (uses server state)
const metrics = await axios.post('/config/layer-metrics', {
  layer_type: 'attention',
  batch_size: 1,
  seq_length: 2048
});
```

### After (Stateless):
```javascript
// Get layers (no server state needed)
const layers = await axios.get('/api/layers', {
  params: { model_id: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' }
});

// Compute metrics (include full context)
const metrics = await axios.post('/config/layer-metrics', {
  model_id: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
  hardware_config: 'nvidia_gb200_single',
  layer_type: 'attention',
  batch_size: 1,
  seq_length: 2048,
  dtype: 'bf16',
  tensor_parallel: 4,
  context_parallel: 2,
  sequence_parallel: 1
});
```

## Comparison Mode

With stateless API, comparison mode is straightforward:

```javascript
// Config 1
const metrics1 = await axios.post('/config/system-metrics', {
  model_id: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
  hardware_config: 'nvidia_gb200_single',
  layers: config1Layers,
  ...
});

// Config 2 (different model/hardware/parallelism)
const metrics2 = await axios.post('/config/system-metrics', {
  model_id: 'meta-llama/Llama-3.3-70B-Instruct',
  hardware_config: 'nvidia_nvl72_rack',
  layers: config2Layers,
  ...
});

// Compare metrics1 vs metrics2
```

No conflicts, no state management issues!
