# InferenceMAX Data Extraction

## Overview

This directory contains tools to extract real-world benchmark data from [InferenceMAX](https://inferencemax.semianalysis.com/), a public performance benchmark database for LLM inference.

## Quick Start

### Prerequisites

1. Install GitHub CLI:
   ```bash
   brew install gh
   gh auth login
   ```

2. Install Python dependencies:
   ```bash
   pip install pandas
   ```

### Extract Data

```bash
cd reconcile/sources/inferencemax

# Auto-collect from recent runs (recommended)
python extract_inferencemax_data.py --collect-recent=30

# Or specify specific run IDs
python extract_inferencemax_data.py 21012127808 20764588467

# Force re-download (delete cache first)
rm -rf raw/ && python extract_inferencemax_data.py --collect-recent=30
```

### Caching Behavior

**Important**: The script downloads **ALL benchmark data** (all models, all hardware) to `./raw/` directory, not just GPT-OSS 120B. This enables:

- **Reuse**: Extract different models later without re-downloading artifacts
- **Efficiency**: Cached downloads are skipped automatically (look for 💾 icon)
- **Flexibility**: Change filtering logic without re-downloading raw data

The raw data includes models like:
- `openai/gpt-oss-120b` (currently extracted)
- `deepseek-ai/DeepSeek-R1-0528`
- `amd/DeepSeek-R1-0528-MXFP4-Preview`

Filtering for specific models happens during CSV creation, not during download.

### Output

The script creates `../../by_model/openai_GPT-OSS-120B/inferencemax.csv` with standardized columns filtered for GPT-OSS 120B only.

| Column | Description |
|--------|-------------|
| `gpu_model` | GPU hardware (h100, h200, b200, mi300x, mi325x, mi355x) |
| `framework` | Inference framework (vllm, sglang, trt, atom) |
| `precision` | Quantization (fp4, fp8) |
| `input_seq_len` | Input sequence length (ISL) |
| `output_seq_len` | Output sequence length (OSL) |
| `tensor_parallel` | Tensor parallelism degree |
| `expert_parallel` | Expert parallelism degree (for MoE) |
| `concurrency` | Number of concurrent requests |
| `throughput_tokens_per_sec_per_gpu` | Total throughput per GPU |
| `output_throughput_per_gpu` | Output token throughput per GPU |
| `mean_time_to_first_token_sec` | Mean TTFT in seconds |
| `p99_time_to_first_token_sec` | P99 TTFT in seconds |
| `mean_time_per_output_token_sec` | Mean TPOT in seconds |
| `mean_end_to_end_latency_sec` | Mean E2E latency in seconds |
| `median_end_to_end_latency_sec` | Median E2E latency in seconds |

## Data Source

- **Website**: https://inferencemax.semianalysis.com/
- **GitHub**: https://github.com/InferenceMAX/InferenceMAX
- **Method**: Downloads aggregated JSON results from GitHub Actions workflow artifacts

## Finding Specific Hardware

The script downloads from the latest successful benchmark run, which may only contain data for specific GPUs. To get data for different hardware:

1. Visit the workflow runs page:
   https://github.com/InferenceMAX/InferenceMAX/actions/workflows/run-sweep.yml

2. Find a completed run that benchmarked your target hardware (check the run details)

3. Get the run ID from the URL (e.g., `21012127808` from `.../runs/21012127808`)

4. Run the script with that ID:
   ```bash
   python utils/extract_inferencemax_data.py 21012127808
   ```

## Example Usage for Calibration

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../../by_model/openai_GPT-OSS-120B/inferencemax.csv')

# Filter for specific configuration
h100_fp8 = df[
    (df['gpu_model'] == 'h100') & 
    (df['precision'] == 'fp8') &
    (df['input_seq_len'] == 1024) &
    (df['output_seq_len'] == 1024)
]

# Plot throughput vs tensor parallelism
plt.plot(h100_fp8['tensor_parallel'], h100_fp8['throughput_tokens_per_sec_per_gpu'])
plt.xlabel('Tensor Parallel Size')
plt.ylabel('Throughput (tokens/sec/GPU)')
plt.title('GPT-OSS 120B on H100 (FP8, 1K/1K)')
plt.show()
```

## Current Data

As of the last extraction (Run ID: 21012127808):
- **GPUs**: B200, B200-TRT, GB200, H100, H200
- **Frameworks**: vLLM, TRT, Dynamo-TRT
- **Precision**: FP4
- **Context Configs**: 
  - 1024/1024 (input/output)
  - 1024/8192
  - 8192/1024
- **Total Data Points**: 235 unique configurations

To get comprehensive data across all GPUs, extract from multiple workflow runs using `--collect-recent`.

## Raw Data Location

Downloaded GitHub Actions artifacts are stored in `./raw/run_<RUN_ID>/` for provenance tracking.

## Notes

- InferenceMAX runs continuous benchmarks across different hardware
- Each workflow run typically focuses on specific GPU models
- Data includes various tensor/expert parallelism configurations
- All measurements are for GPT-OSS 120B model
- Throughput values are **per-GPU** metrics

## Alternative Data Access

If you need the full dataset across all hardware:

1. **Contact InferenceMAX team** - They may provide bulk data exports
2. **Clone their repo** - Run benchmarks yourself (requires GPU access)
3. **Multiple extractions** - Download artifacts from multiple workflow runs
