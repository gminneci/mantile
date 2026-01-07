# Mantile – LLM Performance Estimator

An app to estimate LLM inference performance (latency, throughput, memory) on modern AI accelerators using a stateless backend API.

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### Run locally

Backend (FastAPI):
```bash
./run_backend.sh
```

Frontend (Vite):
```bash
./run_frontend.sh
```

Open http://localhost:5173 in your browser.

## Backend API (Stateless)

The backend provides stateless endpoints; each request includes all required context. Key routes:

- List hardware: `GET /hardware`
	- Returns `configs` with valid hardware IDs (e.g., `nvidia_gb200_single`, `nvidia_h100_80gb`, `nvidia_nvl72_rack`).

- List models: `GET /models`
	- Returns `models` (e.g., `llama_3.3_70b`, `tinyllama_1.1b`)

- Model metadata is available via `GET /models`. The per-model details endpoint has been removed to keep the API minimal and stateless.

- Layer types for a model: `GET /api/layers?model_id=...`
	- Returns layer categories (`attention`, `feedforward`, `norm`, `embedding`) with `count`, `specs`, and `available_parallelism`.

- Layer metrics: `POST /config/layer-metrics`
	- Body: `{ model_id, hardware_config, layer_type, batch_size, seq_length, dtype, tensor_parallel, context_parallel, sequence_parallel }`

- System metrics: `POST /config/system-metrics`
	- Body: `{ model_id, hardware_config, batch_size, input_seq, output_seq, layers: { attention, feedforward, norm, embedding } }`

### Example requests

Discover model and hardware IDs:
```bash
curl -s http://127.0.0.1:8000/models | jq '.models[0].model_id'
curl -s http://127.0.0.1:8000/hardware | jq '.configs'
```

List layer types for Llama 70B:
```bash
curl "http://127.0.0.1:8000/api/layers?model_id=llama_3.3_70b"
```

Compute attention layer metrics on NVL-72 rack:
```bash
curl -X POST http://127.0.0.1:8000/config/layer-metrics \
	-H 'Content-Type: application/json' \
	-d '{
		"model_id": "llama_3.3_70b",
		"hardware_config": "nvidia_nvl72_rack",
		"layer_type": "attention",
		"batch_size": 1,
		"seq_length": 2048,
		"dtype": "bf16",
		"tensor_parallel": 8,
		"context_parallel": 1,
		"sequence_parallel": 1
	}'
```

Compute system metrics:
```bash
curl -X POST http://127.0.0.1:8000/config/system-metrics \
	-H 'Content-Type: application/json' \
	-d '{
		"model_id": "llama_3.3_70b",
		"hardware_config": "nvidia_nvl72_rack",
		"batch_size": 1,
		"input_seq": 2048,
		"output_seq": 128,
		"layers": {
			"attention": { "tensor_parallel": 8, "context_parallel": 1, "dtype": "bf16" },
			"feedforward": { "tensor_parallel": 8, "sequence_parallel": 1, "dtype": "bf16" },
			"norm": { "tensor_parallel": 1, "dtype": "bf16" },
			"embedding": { "tensor_parallel": 1, "dtype": "bf16" }
		}
	}'
```

## Notes

- Hardware IDs must match values from `GET /hardware.configs` (e.g., `nvidia_nvl72_rack`).
- Model IDs come from `GET /models` via the `model_id` field.
- Legacy endpoints like `/config/load` and `/estimate` were removed; use the stateless endpoints above.

## Project Structure (simplified)

```
Mantile/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── models.py            # Data models
│   ├── config_service.py    # Stateless business logic
│   ├── hardware_library.py  # Hardware configs loader
│   ├── model_library.py     # Model IR configs loader
│   └── data/
│       ├── hardware_configs/
│       └── model_configs/
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── main.jsx
└── Instructions.md
```

## Gated Models

Some HuggingFace models require authentication. If you add gated configs:
1. Create a HuggingFace account and request model access
2. Generate a token at https://huggingface.co/settings/tokens
3. Login: `huggingface-cli login`

TinyLlama 1.1B is open-access and works without auth.
