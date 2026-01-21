# MVP Plan: InferenceMAX vs Mantile Reconciliation

## Goal

Create a script that simulates InferenceMAX benchmark configurations using Mantile, then compare predicted vs actual performance metrics to validate and calibrate the model.

## Scope

**In Scope:**
- NVIDIA B200 GPUs in NVL-72 rack configuration only
- GPT-OSS 120B model
- Framework-agnostic comparison (ignore vLLM/TRT/ATOM differences)
- FP4 precision (mapped to nvfp4 in Mantile)
- Metrics: throughput, TTFT, TPOT, E2E latency
- Multi-GPU parallelism (TP: 1, 2, 4, 8)

**Out of Scope (for MVP):**
- H100, H200, AMD accelerators
- Framework-specific optimizations
- Multiple models
- Precision variations beyond FP4

## Hardware Mapping

- **B200** → `nvidia_nvl72_rack` (72 GB200 packages = 144 Blackwell GPUs)
- NVL72 rack supports up to 72-way parallelism
- InferenceMAX TP values represent actual GPU count being used in the rack

## Key Challenge: Parallelism Mapping

**InferenceMAX**: System-level parallelism
- `tensor_parallel`: 1, 2, 4, 8 (number of GPUs in rack used)
- `expert_parallel`: 1, 2, 4 (for MoE layers)
- Specified once per configuration

**Mantile**: Layer-level parallelism
- Each layer type can have different parallelism strategies
- Need to infer layer-level config from system-level spec

**Resolution Strategy:**
1. Assume uniform parallelism across all layers
2. Apply `tensor_parallel` value to both attention and MoE layers
3. Apply `expert_parallel` to MoE layers only
4. Both prefill and decode use same parallelism config
5. Document assumptions for future refinement

## Implementation Steps

### Phase 1: Data Preparation (15 min)

**1.1 Filter InferenceMAX Data**
- Create `reconcile/by_model/openai_GPT-OSS-120B/inferencemax_b200_only.csv`
- Keep only: B200 rows (gpu_model='b200' or 'b200-trt')
- Remove all other GPU types
- Document row count before/after

**1.2 Verify Data Coverage**
Check what configurations we have:
- TP values: 1, 2, 4, 8?
- EP values: 1, 2, 4?
- Context configs: 1K/1K, 1K/8K, 8K/1K?
- Concurrency levels: range?

### Phase 2: Configuration Converter (1.5 hours)

**Script**: `reconcile/utils/inferencemax_to_mantile.py`

**Input**: InferenceMAX CSV row
```csv
gpu_model,framework,precision,input_seq_len,output_seq_len,
tensor_parallel,expert_parallel,concurrency,...
```

**Output**: Mantile API request (JSON)
```json
{
  "prefill_req": {
    "model_id": "openai_GPT-OSS-120B",
    "hardware_id": "nvidia_nvl72_rack",
    "batch_size": 16,
    "seq_len": 1024,
    "layers": {
      "attention": {"tp": 2, "dtype": "nvfp4"},
      "feedforward": {"tp": 2, "ep": 1, "dtype": "nvfp4"}
    }
  },
  "decode_req": {
    "model_id": "openai_GPT-OSS-120B",
    "hardware_id": "nvidia_nvl72_rack",
    "batch_size": 16,
    "seq_len": 1,
    "layers": {
      "attention": {"tp": 2, "dtype": "nvfp4"},
      "feedforward": {"tp": 2, "ep": 1, "dtype": "nvfp4"}
    }
  }
}
```

**Mapping Logic:**
- `gpu_model="b200"` → `hardware_id="nvidia_nvl72_rack"`
- `concurrency` → `batch_size` (for both prefill and decode)
- `input_seq_len` → prefill `seq_len`
- `output_seq_len` → decode `seq_len` (actually tokens/step, so seq_len=1 for decode)
- `tensor_parallel` → `tp` in all layer configs
- `expert_parallel` → `ep` in MoE layer config only
- `precision="fp4"` → `dtype="nvfp4"`

**Layer Configuration Construction:**
1. Load model config to get layer types (attention, feedforward/MoE)
2. For each layer type:
   - Set `tp` = InferenceMAX `tensor_parallel`
   - Set `ep` = InferenceMAX `expert_parallel` (MoE only)
   - Set `dtype` = "nvfp4"

**Function Signature:**
```python
def inferencemax_row_to_mantile_request(row: dict) -> dict:
    """Convert InferenceMAX CSV row to Mantile API request."""
    pass
```

### Phase 3: Batch Estimator (1 hour)

**Script**: `reconcile/utils/run_mantile_estimates.py`

**Functionality:**
1. Read filtered B200-only CSV
2. For each row:
   - Convert to Mantile request using Phase 2 converter
   - Call Mantile backend API (`POST /api/system-metrics`)
   - Extract predictions from response
   - Handle errors gracefully (log and continue)
3. Save predictions to `reconcile/by_model/openai_GPT-OSS-120B/mantile_predictions.csv`

**Output Schema:**
```csv
config_id,gpu_model,framework,isl,osl,tp,ep,concurrency,
predicted_throughput_per_gpu,predicted_ttft_sec,predicted_tpot_sec,
predicted_e2e_latency_sec,api_response_time_ms,error_message
```

**Requirements:**
- Backend must be running locally (`cd backend && uvicorn main:app`)
- Default endpoint: `http://localhost:8000`
- Error handling for API failures (timeouts, 500s, etc.)
- Progress indicator (e.g., tqdm bar)
- Ability to resume from checkpoint:
  - Check if output CSV exists
  - Skip rows already processed
  - Append new predictions

**Usage:**
```bash
python reconcile/utils/run_mantile_estimates.py \
  --input reconcile/by_model/openai_GPT-OSS-120B/inferencemax_b200_only.csv \
  --output reconcile/by_model/openai_GPT-OSS-120B/mantile_predictions.csv \
  --api-url http://localhost:8000
```

### Phase 4: Comparison & Analysis (1 hour)

**Script**: `reconcile/utils/compare_predictions.py`

**Functionality:**
1. Load InferenceMAX actuals (B200 filtered)
2. Load Mantile predictions
3. Join on configuration keys: (isl, osl, tp, ep, concurrency)
4. Calculate error metrics for each config:
   - Absolute error
   - Percentage error: `(predicted - actual) / actual * 100`
5. Aggregate statistics:
   - MAPE (Mean Absolute Percentage Error) per metric
   - RMSE (Root Mean Squared Error) per metric
   - Min/max/median errors
6. Output comparison CSV and summary stats

**Output**: `reconcile/by_model/openai_GPT-OSS-120B/comparison_report.csv`
```csv
config_id,isl,osl,tp,ep,concurrency,framework,
actual_throughput,predicted_throughput,throughput_abs_error,throughput_pct_error,
actual_ttft,predicted_ttft,ttft_abs_error,ttft_pct_error,
actual_tpot,predicted_tpot,tpot_abs_error,tpot_pct_error,
actual_e2e_latency,predicted_e2e_latency,e2e_abs_error,e2e_pct_error
```

**Summary Stats Output**: `reconcile/by_model/openai_GPT-OSS-120B/comparison_summary.txt`
```
=== Mantile vs InferenceMAX Comparison Summary ===

Dataset: B200/NVL72, GPT-OSS 120B, FP4
Configurations: 150 (example)

Overall MAPE:
  Throughput:        25.3%
  TTFT:              18.7%
  TPOT:              22.1%
  E2E Latency:       20.5%

RMSE:
  Throughput:        5234 tokens/sec
  TTFT:              0.15 sec
  TPOT:              0.008 sec
  E2E Latency:       1.2 sec

By Framework:
  vLLM:    MAPE 23.1%
  TRT:     MAPE 28.5%
  ATOM:    MAPE 24.7%

By Parallelism:
  TP=1:    MAPE 19.2%
  TP=2:    MAPE 24.5%
  TP=4:    MAPE 28.1%
  TP=8:    MAPE 31.7%

Top 5 Best Predictions (lowest error):
  ...

Top 5 Worst Predictions (highest error):
  ...
```

**Usage:**
```bash
python reconcile/utils/compare_predictions.py \
  --actuals reconcile/by_model/openai_GPT-OSS-120B/inferencemax_b200_only.csv \
  --predictions reconcile/by_model/openai_GPT-OSS-120B/mantile_predictions.csv \
  --output reconcile/by_model/openai_GPT-OSS-120B/comparison_report.csv
```

### Phase 5: Visualization (Optional - 30 min)

**Notebook**: `reconcile/notebooks/mvp_analysis.ipynb`

Visualizations:
1. **Scatter plots**: Predicted vs Actual (per metric, with y=x reference line)
2. **Error distribution**: Histograms of percentage errors
3. **By parallelism**: Box plots of error by TP value
4. **By framework**: Compare Mantile predictions against vLLM/TRT/ATOM
5. **By context length**: Error patterns for 1K/1K vs 1K/8K vs 8K/1K

## Critical Assumptions & Limitations

### 1. Framework-Agnostic Modeling

**Assumption**: Mantile predictions should match average performance across frameworks.

**Reality**: vLLM, TRT, and ATOM have different kernel implementations and optimizations.

**Impact**: Expect ±20-30% variance due to framework differences.

**Mitigation**: 
- Report per-framework MAPE separately
- Focus on average trends rather than exact matches
- Future work: Add framework-specific correction factors

### 2. Decode Sequence Length

**InferenceMAX**: `output_seq_len` = total tokens to generate

**Mantile**: Decode operates per-token (seq_len=1 for decode phase)

**Mapping**: 
- Prefill: `seq_len` = `input_seq_len`
- Decode: `seq_len` = 1 (per-token processing)
- Use InferenceMAX metrics that account for full decode sequence

### 3. Throughput Metrics

**InferenceMAX**: Reports `throughput_tokens_per_sec_per_gpu`

**Mantile**: May report aggregate rack throughput

**Normalization**: Divide by number of GPUs used (TP value) to get per-GPU throughput.

### 4. Concurrency vs Batch Size

**InferenceMAX**: `concurrency` = number of concurrent requests

**Mantile**: `batch_size` = number of sequences processed together

**Assumption**: Map `concurrency` → `batch_size` directly, though these may not be exactly equivalent in practice.

## Success Criteria

**MVP Success:**
- ✅ Script runs end-to-end for all B200 configs
- ✅ Produces comparison report with error metrics
- ✅ MAPE < 50% for throughput (initial calibration target)
- ✅ Identifies systematic biases (e.g., always over/under predicting)
- ✅ Documents patterns by parallelism level and context length

**Stretch Goals:**
- MAPE < 30% for throughput
- MAPE < 40% for latency metrics
- Identify which configs Mantile predicts best/worst

**Future Iterations:**
- Add H100, H200 support
- Per-framework calibration factors
- Layer-level parallelism optimization
- Memory constraint validation

## Timeline Estimate

- Phase 1 (Data Prep): 15 min
- Phase 2 (Converter): 1.5 hours
- Phase 3 (Estimator): 1 hour
- Phase 4 (Comparison): 1 hour
- Phase 5 (Viz - optional): 30 min

**Total**: ~4 hours for MVP

## Execution Order

1. ✅ Document plan (this file)
2. Filter InferenceMAX data to B200 only
3. Implement configuration converter (`inferencemax_to_mantile.py`)
4. Test converter on 2-3 sample rows
5. Implement batch estimator (`run_mantile_estimates.py`)
6. Start backend API server
7. Run estimator (may take 30+ min for all configs)
8. Implement comparison script (`compare_predictions.py`)
9. Generate comparison report
10. Analyze results, document findings
11. Iterate: adjust Mantile model based on systematic biases

## Open Questions

1. **API stability**: Does backend handle all GPT-OSS layer types correctly?
2. **Runtime**: How long does one API call take? (estimate total runtime)
3. **Metrics alignment**: Do InferenceMAX and Mantile define TTFT/TPOT/E2E the same way?
4. **Failure modes**: What happens if backend returns errors for certain configs?

## Next Actions

After plan approval:
1. Filter CSV to B200 data
2. Start implementing converter script
3. Test against running backend
4. Proceed with batch estimation
