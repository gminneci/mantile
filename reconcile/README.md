# Model Reconciliation

This directory contains tools and data for **reconciling Mantile's performance estimates** against real-world benchmark data from public sources.

## Purpose

Mantile provides theoretical performance estimates based on model architecture and hardware specifications. To validate and calibrate these estimates, we collect actual benchmark results from various public sources and compare them against Mantile's predictions.

## Structure

```
reconcile/
├── by_model/              # Organized by model (HuggingFace naming)
│   └── openai_GPT-OSS-120B/
│       ├── inferencemax.csv           # Source benchmark data
│       ├── inferencemax_b200_only.csv # Filtered subset (B200 only)
│       ├── README.md                  # Model-specific notes
│       └── RECONCILIATION_SUMMARY.md  # Analysis & findings
├── sources/               # Data provider extraction tools
│   └── inferencemax/
│       ├── extract_inferencemax_data.py  # Extract from GitHub artifacts
│       ├── README.md
│       └── raw/                          # Downloaded artifacts (cached)
└── utils/                 # Reconciliation workflow utilities
    ├── inferencemax_to_mantile.py    # Convert benchmark configs to Mantile API requests
    ├── run_mantile_estimates.py       # Batch estimate runner
    └── compare_predictions.py         # Error analysis and reporting
```

## Workflow

### Complete Reconciliation Process

1. **Extract** - Download benchmark data from provider
   ```bash
   cd reconcile/sources/inferencemax
   python extract_inferencemax_data.py --model openai_GPT-OSS-120B --collect-recent 5
   ```
   
2. **Filter** - Create model-specific filtered datasets (e.g., B200-only)
   ```bash
   # Manual filtering or custom scripts to create inferencemax_b200_only.csv
   ```

3. **Convert & Estimate** - Generate Mantile predictions for all benchmark configs
   ```bash
   cd reconcile/utils
   python run_mantile_estimates.py --model openai_GPT-OSS-120B \
     --input ../by_model/openai_GPT-OSS-120B/inferencemax_b200_only.csv \
     --output ../by_model/openai_GPT-OSS-120B/mantile_predictions.csv
   ```

4. **Compare** - Analyze prediction errors and generate reports
   ```bash
   python compare_predictions.py --model openai_GPT-OSS-120B \
     --actuals ../by_model/openai_GPT-OSS-120B/inferencemax_b200_only.csv \
     --predictions ../by_model/openai_GPT-OSS-120B/mantile_predictions.csv \
     --output ../by_model/openai_GPT-OSS-120B/comparison_report.csv
   ```

5. **Reconcile** - Document findings and calibrate model parameters
   - Update model configs based on systematic errors
   - Document in `by_model/<model>/RECONCILIATION_SUMMARY.md`

### Utilities

**`inferencemax_to_mantile.py`**
- Converts InferenceMAX CSV rows to Mantile API request format
- Handles hardware mapping (e.g., B200 → nvidia_nvl72_rack)
- Maps precision formats (fp4 → nvfp4, etc.)
- Configurable with `--model` parameter

**`run_mantile_estimates.py`**
- Batch processes benchmark configs through Mantile API
- Incremental saving (every 10 rows) for long runs
- Resume support to skip already-processed configs
- Outputs predictions with timing and memory metrics

**`compare_predictions.py`**
- Calculates error metrics (MAPE, absolute error, percentage error)
- Generates breakdown by framework, tensor parallelism, context length
- Identifies best/worst predictions for investigation
- Outputs detailed comparison CSV and summary statistics

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
