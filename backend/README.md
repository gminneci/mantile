# Mantile Backend API

The Mantile backend is a stateless FastAPI service for computing LLM inference metrics. All endpoints are frontend-agnostic and require complete context in each request.

## Starting the Backend

```bash
./run_backend.sh
```

The backend runs on `http://localhost:8000`

## API Endpoints

### GET `/models/{model_id}`

Load a specific model configuration.

**Response:**
```json
{
  "layer_types": [
    {
      "name": "attention",
      "class": "Attention",
      "count": 80,
      "specs": { ... }
    },
    ...
  ]
}
```

---

### GET `/hardware/{config_name}`

Load a specific hardware configuration.

**Parameters:**
- `config_name`: Hardware ID (e.g., `nvidia_nvl72_rack`, `nvidia_h100_80gb`, `nvidia_gb200_single`)

**Response:**
```json
{
  "name": "NVIDIA NVL-72 Rack",
  "num_chips": 72,
  "memory": [
    {
      "type": "HBM3e",
      "capacity_gb": 141,
      "bandwidth_gbps": 4800
    }
  ],
  "compute": {
    "bf16": 2000.0,
    "fp16": 2000.0,
    "fp8": 4000.0,
    "int8": 4000.0
  },
  "interconnect": { ... }
}
```

---

### GET `/api/layers`

Get layer type information for a model, including supported parallelism strategies.

**Query Parameters:**
- `model_id`: Model identifier (e.g., `llama_3.3_70b`, `tinyllama_1.1b`)

**Example:**
```bash
curl "http://localhost:8000/api/layers?model_id=llama_3.3_70b"
```

**Response:**
```json
{
  "layers": [
    {
      "name": "attention",
      "class": "Attention",
      "count": 80,
      "specs": {
        "hidden_size": 8192,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128
      },
      "available_parallelism": ["tensor_parallel", "context_parallel"]
    },
    {
      "name": "feedforward",
      "class": "MLP",
      "count": 80,
      "specs": {
        "hidden_size": 8192,
        "intermediate_size": 28672
      },
      "available_parallelism": ["tensor_parallel", "sequence_parallel"]
    },
    ...
  ]
}
```

---

### POST `/config/system-metrics`

Compute complete system-level metrics for both prefill and decode phases.

**Request Body:**
```json
{
  "prefill_req": {
    "model_id": "llama_3.3_70b",
    "hardware_id": "nvidia_nvl72_rack",
    "batch_size": 128,
    "seq_len": 1024,
    "layers": {
      "attention": {
        "tensor_parallel": 8,
        "context_parallel": 1,
        "dtype": "fp8"
      },
      "feedforward": {
        "tensor_parallel": 8,
        "sequence_parallel": 1,
        "dtype": "fp8"
      },
      "norm": {
        "tensor_parallel": 1,
        "dtype": "fp8"
      },
      "embedding": {
        "tensor_parallel": 1,
        "dtype": "fp8"
      }
    }
  },
  "decode_req": {
    "model_id": "llama_3.3_70b",
    "hardware_id": "nvidia_nvl72_rack",
    "batch_size": 128,
    "seq_len": 1024,
    "layers": { ... }
  }
}
```

**Response:**
```json
{
  "system": {
    "total_time_ms": 1234.56,
    "throughput_tokens_s": 890.12,
    "total_memory_gb": 567.89,
    "utilization_percent": 78.5,
    "num_chips": 72,
    "bottleneck": "compute",
    "fits_on_hardware": true
  },
  "prefill": {
    "latency_ms": 45.67,
    "memory_gb": 123.45,
    "flops": 890.12,
    "communication_time_ms": 12.34
  },
  "decode": {
    "latency_ms": 2.34,
    "memory_gb": 123.45,
    "flops": 45.67,
    "communication_time_ms": 0.89
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/config/system-metrics \
  -H 'Content-Type: application/json' \
  -d '{
    "prefill_req": {
      "model_id": "llama_3.3_70b",
      "hardware_id": "nvidia_nvl72_rack",
      "batch_size": 128,
      "seq_len": 1024,
      "layers": {
        "attention": {"tensor_parallel": 8, "context_parallel": 1, "dtype": "fp8"},
        "feedforward": {"tensor_parallel": 8, "sequence_parallel": 1, "dtype": "fp8"},
        "norm": {"tensor_parallel": 1, "dtype": "fp8"},
        "embedding": {"tensor_parallel": 1, "dtype": "fp8"}
      }
    },
    "decode_req": {
      "model_id": "llama_3.3_70b",
      "hardware_id": "nvidia_nvl72_rack",
      "batch_size": 128,
      "seq_len": 1024,
      "layers": {
        "attention": {"tensor_parallel": 8, "context_parallel": 1, "dtype": "fp8"},
        "feedforward": {"tensor_parallel": 8, "sequence_parallel": 1, "dtype": "fp8"},
        "norm": {"tensor_parallel": 1, "dtype": "fp8"},
        "embedding": {"tensor_parallel": 1, "dtype": "fp8"}
      }
    }
  }'
```

---

## Architecture

The backend is organized into modular layers:

```
backend/
├── main.py              # FastAPI app and endpoints
├── ir_builder.py        # Model IR generation from HuggingFace
├── layers/              # Layer implementations
│   ├── base.py          # Base Layer class
│   ├── attention.py     # Attention layer
│   ├── mlp.py           # MLP/Feedforward layer
│   ├── norm.py          # Normalization layer
│   └── embedding.py     # Embedding layer
└── data/
    ├── hardware_configs/
    └── model_configs/
```

### Adding New Layers

1. Subclass `Layer` in `backend/layers/`
2. Implement `compute_metrics(batch_size, seq_len, phase)` 
3. Define `get_supported_parallelism()` classmethod
4. Update model configs to reference your new layer class

### Adding Hardware/Models

- Hardware configs: See [data/hardware_configs/README.md](data/hardware_configs/README.md)
- Model configs: See [data/model_configs/README.md](data/model_configs/README.md)

---

## Stateless Design

Every endpoint is **stateless**—no session state is maintained between requests. This ensures:

- **Horizontal scalability**: Multiple backend instances can serve requests
- **Frontend independence**: Any client can use the API
- **Reproducibility**: Same input always produces same output
- **Simplicity**: No state management complexity

Each request must include all necessary context (model ID, hardware ID, layer configs, etc.).
