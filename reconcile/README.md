# Model Reconciliation

This directory contains tools and data for **reconciling Mantile's performance estimates** against real-world benchmark data from public sources.

## Purpose

Mantile provides theoretical performance estimates based on model architecture and hardware specifications. To validate and calibrate these estimates, we collect actual benchmark results from various public sources and compare them against Mantile's predictions.

## Structure

```
reconcile/
├── by_model/              # Organized by model (HuggingFace naming)
│   └── openai_GPT-OSS-120B/
│       ├── inferencemax.csv      # Benchmark data from InferenceMAX
│       └── README.md              # Model-specific reconciliation notes
├── sources/               # Data provider extraction tools
│   └── inferencemax/
│       ├── extract_inferencemax_data.py
│       ├── README.md
│       └── raw/           # Downloaded artifacts
└── utils/                 # Shared utilities (if needed)
```

## Workflow

1. **Extract** - Run data provider scripts to download benchmark results
2. **Standardize** - Convert to common CSV format in `by_model/<model>/`
3. **Compare** - Run Mantile estimates for same configurations
4. **Reconcile** - Analyze differences, calibrate model parameters

## Data Sources

- **InferenceMAX** - Comprehensive benchmark suite across NVIDIA/AMD GPUs
  - Website: https://inferencemax.semianalysis.com/
  - GitHub: https://github.com/InferenceMAX/InferenceMAX
  - Status: ✅ Active (GPT-OSS-120B)

## Standard CSV Format

Each `<provider>.csv` file in `by_model/` directories should contain:

| Column | Description |
|--------|-------------|
| `gpu_model` | Hardware identifier (h100, h200, b200, mi355x, etc.) |
| `framework` | Inference engine (vllm, sglang, trt, etc.) |
| `precision` | Quantization (fp4, fp8, fp16, etc.) |
| `input_seq_len` | Input sequence length |
| `output_seq_len` | Output sequence length |
| `tensor_parallel` | Tensor parallelism degree |
| `expert_parallel` | Expert parallelism degree (for MoE) |
| `concurrency` | Number of concurrent requests |
| `throughput_tokens_per_sec_per_gpu` | Total throughput per GPU |
| `output_throughput_per_gpu` | Output token throughput per GPU |
| `mean_time_to_first_token_sec` | Mean TTFT |
| `p99_time_to_first_token_sec` | P99 TTFT |
| `mean_time_per_output_token_sec` | Mean TPOT |
| `mean_end_to_end_latency_sec` | Mean E2E latency |
| `median_end_to_end_latency_sec` | Median E2E latency |
| `benchmark_date` | ISO 8601 timestamp |
| `run_id` | Provider-specific run identifier |

Additional provider-specific columns are allowed but should be documented.

## Adding New Data Sources

1. Create `reconcile/sources/<provider>/` directory
2. Add extraction script and README
3. Output standardized CSV to appropriate `by_model/<model>/` directory
4. Update this README with source status

## Adding New Models

1. Create `reconcile/by_model/<org>_<model>/` directory
2. Add provider CSVs as they become available
3. Create model-specific README documenting reconciliation approach
