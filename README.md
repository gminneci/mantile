# Mantile (μάντις)

**Mantile** references the Greek **μάντις** (_mantis_), meaning "prophet," "seer," or "one who divines." In ancient Greece, a _mantis_ was a religious specialist inspired by the gods, capable of foretelling the future. Like its namesake, Mantile predicts LLM inference performance—estimating latency, throughput, and memory usage on modern AI accelerators.

---

## Philosophy

Mantile is designed with three core principles:

1. **Test-Driven Development**: Contributors add tests for new layers, parallelism strategies, or model configurations. The codebase is designed for AI-assisted development, where agents can validate tests and help implement the necessary components.

2. **Stateless Backend**: The FastAPI backend is completely stateless and frontend-agnostic. Every request includes all required context, making the API reusable across different clients and tools.

3. **Extensible Architecture**: New hardware accelerators, model architectures, and optimization strategies can be added through simple JSON configurations and layer implementations.

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### Run Locally

**Backend (FastAPI):**
```bash
./run_backend.sh
```
Backend runs on http://localhost:8000

**Frontend (React + Vite):**
```bash
./run_frontend.sh
```
Frontend runs on http://localhost:5173

Open http://localhost:5173 in your browser to use the interactive configurator.

---

## Contributing

Mantile welcomes contributions! Here's how you can help:

- **Add Tests**: Define expected behavior for new layers or parallelism modes  
  → See [tests/README.md](/tests/README.md)

- **Add Hardware Configurations**: Define new AI accelerators  
  → See [backend/data/hardware_configs/README.md](/backend/data/hardware_configs/README.md)

- **Add Model Configurations**: Add support for new LLM architectures  
  → See [backend/data/model_configs/README.md](/backend/data/model_configs/README.md)

---

## Backend API

The backend provides a stateless REST API for computing LLM inference metrics. All endpoints are frontend-agnostic and fully documented.

**Start the backend:**
```bash
./run_backend.sh
```

**Full API reference:**  
→ See [backend/README.md](/backend/README.md)

**Quick example:**
```bash
# Get layer information for a model
curl "http://localhost:8000/api/layers?model_id=meta-llama_Llama-3.3-70B-Instruct" | jq

# Compute system metrics for prefill + decode
curl -X POST http://localhost:8000/config/system-metrics \
  -H 'Content-Type: application/json' \
  -d '{ "prefill_req": {...}, "decode_req": {...} }'
```

---

## Frontend

The frontend is a React application built with Vite, providing an interactive UI for configuring LLM inference scenarios and comparing system configurations side-by-side.

**Start the frontend:**
```bash
./run_frontend.sh
```

**Details:**  
→ See [frontend/README.md](/frontend/README.md)

---

## Model Reconciliation

Mantile includes tools for validating and calibrating performance estimates against real-world benchmark data from public sources like InferenceMAX.

**Workflow:**
1. Extract benchmark data from public sources
2. Generate Mantile predictions for same configurations
3. Compare and analyze prediction errors
4. Calibrate model parameters based on findings

**Details:**  
→ See [reconcile/README.md](/reconcile/README.md)

---

## Roadmap

### Layer-Level Metrics
- [ ] Add detailed per-layer metrics view showing compute, memory, and communication breakdown
- [ ] Visualize bottlenecks at the layer level (e.g., attention vs MLP compute/memory trade-offs)
- [ ] Support layer-by-layer profiling to identify optimization opportunities

### Validation & Calibration
- [x] Build reconciliation infrastructure for comparing predictions with benchmarks
- [x] Extract and standardize data from InferenceMAX
- [x] Automated batch prediction and comparison workflows
- [ ] Calibrate model FLOPs utilization (MFU) based on benchmark data
- [ ] Expand to additional benchmark sources and models

### Agentic IR Builder
- [ ] Make IR builder autonomous: automatically infer model architectures from HuggingFace configs
- [ ] Support automatic layer detection and parallelism strategy suggestions
- [ ] Enable AI-assisted model configuration generation from model cards
- [ ] Implement automated testing for newly added models

### Contribution Automation
- [ ] Create automated workflows for validating new hardware configurations
- [ ] Build CI/CD pipelines for testing model configurations against known benchmarks
- [ ] Enable community contributions through automated validation and testing
- [ ] Develop tooling for semi-automated addition of new hardware accelerators

---

## License

Copyright © 2026. All rights reserved.
