# InferenceMAX Data Extraction - TODO

## Task: Extract October/November 2025 Historical Data

### Background

InferenceMAX launched October 9, 2025, but the aggregated benchmark format (`results_bmk` artifact with `agg_bmk.json`) only started on **December 9, 2025** with the "Run Sweep" workflow.

Earlier benchmarks (October-November 2025) exist but in a **different, non-aggregated format** where each configuration is stored as a separate artifact.

### Current State

- ✅ Script extracts aggregated data from December 9, 2025 onwards
- ❌ October/November data requires different extraction approach
- Current earliest data: December 9, 2025 (run 20074006738)

### Older Workflows Available

#### 1. Full Sweep Scheduler - 1k1k
- **Workflow ID**: 200724779
- **Created**: October 24, 2025
- **Path**: `.github/workflows/1k1k-sweep.yml`
- **Runs**: https://github.com/InferenceMAX/InferenceMAX/actions/workflows/1k1k-sweep.yml
- **Date Range**: October 24 - October 31, 2025

Example runs:
```
18984029536  2025-10-31T20:11:47Z  completed
18978812542  2025-10-31T16:28:08Z  completed
18978773248  2025-10-31T16:26:34Z  completed
18950071657  2025-10-30T17:47:11Z  completed
```

#### 2. Test Sweep
- **Workflow ID**: 195698854
- **Created**: October 7, 2025
- **Path**: `.github/workflows/test.yml`
- **Runs**: https://github.com/InferenceMAX/InferenceMAX/actions/workflows/test.yml
- **Date Range**: October 7 - October 24, 2025

### Artifact Format in Older Runs

Instead of one `results_bmk` artifact with `agg_bmk.json`, each benchmark is a **separate artifact**:

Example artifact names from run 18984029536:
```
gptoss_1k1k_fp4_vllm_tp8_ep1_dpa_false_conc8_h100-cw_1
gptoss_1k1k_fp4_vllm_tp4_ep1_dpa_false_conc4_h200-nb_1
gptoss_1k1k_fp4_trt_tp2_ep1_dpa_false_conc16_b200-nv_0
gptoss_1k1k_fp4_vllm_tp2_ep1_dpa_false_conc16_mi325x-amd_0
gptoss_1k1k_fp4_trt_tp4_ep4_dpa_false_conc32_h200-nb_2
```

Naming pattern: `{model}_{isl}{osl}_{precision}_{framework}_tp{tp}_ep{ep}_dpa_{dpa}_conc{conc}_{gpu}_{index}`

### What Needs to be Implemented

#### 1. List Individual Artifacts
For each run, list all artifacts:
```bash
gh api /repos/InferenceMAX/InferenceMAX/actions/runs/<RUN_ID>/artifacts \
  --jq '.artifacts[] | select(.name | startswith("gptoss")) | {name, id, size_in_bytes}'
```

#### 2. Download Individual Artifacts
Download each artifact separately:
```bash
gh run download <RUN_ID> \
  --repo InferenceMAX/InferenceMAX \
  --name <ARTIFACT_NAME> \
  -D raw/run_<RUN_ID>/<ARTIFACT_NAME>
```

#### 3. Parse Individual Results
Each artifact likely contains:
- JSON or CSV with benchmark metrics
- Configuration parameters encoded in filename
- Need to reverse-engineer the exact format

**Investigation needed**:
- Download 1-2 sample artifacts
- Examine their internal structure
- Determine what metrics are available
- Compare to aggregated format

#### 4. Extract Configuration from Filename
Parse artifact names to extract:
- `model`: Model name (gptoss = GPT-OSS 120B)
- `isl`: Input sequence length (e.g., 1k = 1024)
- `osl`: Output sequence length
- `precision`: fp4, fp8, fp16, etc.
- `framework`: vllm, trt, etc.
- `tp`: Tensor parallel
- `ep`: Expert parallel
- `dpa`: Distributed parameter attention (true/false)
- `conc`: Concurrency
- `gpu`: GPU type (h100-cw, h200-nb, b200-nv, mi325x-amd, etc.)
- `index`: Run index/replica number

#### 5. Aggregate Results
- Merge all artifacts from a run into single dataset
- Apply same deduplication logic as current script
- Merge with December+ data

### Implementation Plan

1. **Phase 1: Investigation** (30 min)
   - Download 3-5 sample artifacts from different runs
   - Examine internal format
   - Document structure and available metrics
   - Verify it matches aggregated format

2. **Phase 2: Parser** (1-2 hours)
   - Create `parse_individual_artifact()` function
   - Extract config from filename
   - Parse benchmark results from file content
   - Convert to standard CSV format

3. **Phase 3: Workflow Integration** (1 hour)
   - Add `--workflow` parameter to script (default: run-sweep.yml)
   - Support both aggregated and individual artifact formats
   - Add `--date-range` to target specific months
   - Update deduplication to handle mixed sources

4. **Phase 4: Testing** (30 min)
   - Extract October data
   - Verify metrics match expected values
   - Compare with December data for consistency
   - Merge and validate final dataset

### Commands to Get Started

```bash
# List all runs from the 1k1k-sweep workflow
gh api /repos/InferenceMAX/InferenceMAX/actions/workflows/1k1k-sweep.yml/runs \
  --paginate \
  --jq '.workflow_runs[] | select(.status=="completed") | {id, created_at}'

# Check artifacts for a specific run
gh api /repos/InferenceMAX/InferenceMAX/actions/runs/18984029536/artifacts \
  --jq '.artifacts[] | select(.name | startswith("gptoss")) | .name'

# Download sample artifact
gh run download 18984029536 \
  --repo InferenceMAX/InferenceMAX \
  --name "gptoss_1k1k_fp4_vllm_tp8_ep1_dpa_false_conc8_h100-cw_1" \
  -D raw/sample_oct_artifact
```

### Expected Outcome

- Add ~50-100 additional configurations from October-November 2025
- Extend date range back to October 24, 2025 (or October 7 if Test Sweep has data)
- More comprehensive GPU coverage from early benchmarks

### Priority

**Low-Medium** - Current December+ data (203-235 configs) is sufficient for initial reconciliation. Historical data nice-to-have for trend analysis.

### Notes

- GitHub Actions artifacts are retained for 90 days by default, but InferenceMAX may have extended retention
- If artifacts are expired, may need to reconstruct from git history or contact maintainers
- Consider opening an issue on InferenceMAX repo asking if aggregated historical data is available
