# GPT-OSS-120B Reconciliation

## Model Information

- **Organization**: OpenAI
- **Model**: GPT-OSS-120B
- **Architecture**: Mixture of Experts (MoE) Transformer
- **Parameters**: 120B total
- **HuggingFace ID**: `openai/GPT-OSS-120B`

## Data Sources

### InferenceMAX
- **File**: `inferencemax.csv`
- **Records**: 235 unique configurations
- **Date Range**: 2025-12-17 to 2026-01-07
- **Hardware Coverage**:
  - B200 (56 configs)
  - B200-TRT (60 configs)
  - GB200 (25 configs)
  - H100 (43 configs)
  - H200 (51 configs)
- **Frameworks**: vLLM (150), TRT (60), Dynamo-TRT (25)
- **Precision**: FP4
- **Context Configs**: 1K/1K, 1K/8K, 8K/1K

## Reconciliation Status

- [ ] Mantile configuration validated against architecture
- [ ] Baseline estimates generated for benchmark configurations
- [ ] Comparison analysis completed
- [ ] Parameter calibration performed
- [ ] Results documented

## Key Configurations for Validation

Priority configurations to test:
1. **H100 @ FP8, 1K/1K, TP=4** - Most common training config
2. **B200 @ FP4, 8K/1K, TP=2** - Long context inference
3. **GB200 @ FP4, 1K/1K, disaggregated** - Multi-node setup

## Notes

- InferenceMAX data focuses on FP4 precision for this model
- All measurements are per-GPU metrics
- Deduplication keeps most recent results for each configuration
- Framework differences (vLLM vs TRT) may affect comparisons

## Next Steps

1. Extract hardware specs for each GPU type → match with Mantile configs
2. Run Mantile estimates for all 235 configurations
3. Calculate error metrics (MAPE, RMSE) for throughput/latency
4. Identify systematic biases by hardware/parallelism
5. Adjust Mantile layer implementations or calibration factors
