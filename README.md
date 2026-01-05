# Mantile - LLM Performance Estimator

An application to estimate LLM inference performance (latency, throughput, memory) on modern AI accelerators.

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+

### Running the Application

**Terminal 1 - Backend:**
```bash
./run_backend.sh
```

**Terminal 2 - Frontend:**
```bash
./run_frontend.sh
```

Then open your browser to: **http://localhost:5173**

## Important Notes

**Gated Models:** Some models like `meta-llama/Llama-3.1-8B` require HuggingFace authentication. To use gated models:
1. Create a HuggingFace account and request access to the model
2. Generate an access token at https://huggingface.co/settings/tokens
3. Login via CLI: `huggingface-cli login`

The default model (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) is open-access and works without authentication.

## Project Structure

```
Mantile/
├── backend/              # Python FastAPI backend
│   ├── models.py        # Data models
│   ├── hardware_library.py  # Hardware specs (NVL-72)
│   ├── ir_builder.py    # Model IR from HuggingFace
│   ├── estimator.py     # Performance calculation
│   └── main.py          # FastAPI server
├── frontend/            # React + Vite frontend
│   └── src/
│       ├── App.jsx      # Main dashboard
│       └── index.css    # Styling
└── Instructions.md      # Full design doc
```

## Current Features

- ✅ Hardware Profile: NVIDIA NVL-72 (single chip or full rack)
- ✅ Model Support: Llama-style dense models from HuggingFace
- ✅ Parallelism: Tensor Parallelism (TP)
- ✅ Metrics: TTFT, TPOT, Throughput, Memory Breakdown, Bottleneck Analysis
- ✅ Interactive UI: Real-time configuration and visualization

## Roadmap

- [ ] Pipeline Parallelism (PP)
- [ ] MoE support (DeepSeekV3)
- [ ] Multi-Layer Attention (MLA)
- [ ] Additional hardware profiles
- [ ] Efficiency calibration from benchmarks
