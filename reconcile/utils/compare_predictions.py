#!/usr/bin/env python3
"""
Comparison Script: Mantile Predictions vs InferenceMAX Actuals

Compares predicted vs actual performance metrics and generates error analysis.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def calculate_error_metrics(actual, predicted):
    """Calculate absolute and percentage errors."""
    abs_error = predicted - actual
    pct_error = (abs_error / actual) * 100 if actual != 0 else np.nan
    return abs_error, pct_error


def compare_predictions(actuals_path: Path, predictions_path: Path, output_path: Path):
    """
    Compare predictions with actuals and generate comparison report.
    
    Args:
        actuals_path: InferenceMAX filtered CSV
        predictions_path: Mantile predictions CSV
        output_path: Output comparison CSV
    """
    # Load data
    actuals = pd.read_csv(actuals_path)
    predictions = pd.read_csv(predictions_path)
    
    print(f"📊 Loaded {len(actuals)} actuals and {len(predictions)} predictions")
    
    # Verify same configs
    if len(actuals) != len(predictions):
        print(f"⚠️  Warning: Different number of rows!")
    
    # Since predictions were generated from actuals, they should have the same index
    # But let's merge on config keys to be safe
    config_cols = ['gpu_model', 'input_seq_len', 'output_seq_len', 
                   'tensor_parallel', 'expert_parallel', 'concurrency']
    
    # Prepare actuals
    actuals_clean = actuals[config_cols + [
        'framework',
        'throughput_tokens_per_sec_per_gpu',
        'mean_time_to_first_token_sec',
        'mean_time_per_output_token_sec',
        'mean_end_to_end_latency_sec'
    ]].copy()
    actuals_clean.columns = config_cols + [
        'framework',
        'actual_throughput',
        'actual_ttft',
        'actual_tpot',
        'actual_e2e_latency'
    ]
    
    # Prepare predictions
    predictions_clean = predictions[config_cols + [
        'predicted_throughput_per_gpu',
        'predicted_ttft_sec',
        'predicted_tpot_sec',
        'predicted_e2e_latency_sec',
        'fits_on_hardware',
        'bottleneck'
    ]].copy()
    
    # Merge
    comparison = actuals_clean.merge(
        predictions_clean,
        on=config_cols,
        how='inner'
    )
    
    print(f"✅ Merged {len(comparison)} matching configurations")
    
    # Calculate errors for each metric
    metrics = [
        ('throughput', 'actual_throughput', 'predicted_throughput_per_gpu'),
        ('ttft', 'actual_ttft', 'predicted_ttft_sec'),
        ('tpot', 'actual_tpot', 'predicted_tpot_sec'),
        ('e2e_latency', 'actual_e2e_latency', 'predicted_e2e_latency_sec')
    ]
    
    for metric_name, actual_col, pred_col in metrics:
        comparison[f'{metric_name}_abs_error'] = comparison[pred_col] - comparison[actual_col]
        comparison[f'{metric_name}_pct_error'] = (
            (comparison[pred_col] - comparison[actual_col]) / comparison[actual_col] * 100
        )
    
    # Save comparison
    comparison.to_csv(output_path, index=False)
    print(f"\n💾 Saved comparison to {output_path}")
    
    # Generate summary statistics
    generate_summary(comparison, output_path.parent / "comparison_summary.txt")
    
    return comparison


def generate_summary(comparison: pd.DataFrame, summary_path: Path):
    """Generate summary statistics and save to file."""
    
    lines = [
        "=" * 70,
        "Mantile vs InferenceMAX Comparison Summary",
        "=" * 70,
        "",
        f"Dataset: B200/NVL72, GPT-OSS 120B, FP4",
        f"Configurations: {len(comparison)}",
        "",
        "=" * 70,
        "Overall Error Metrics (MAPE = Mean Absolute Percentage Error)",
        "=" * 70,
        ""
    ]
    
    metrics = {
        'Throughput': 'throughput_pct_error',
        'TTFT': 'ttft_pct_error',
        'TPOT': 'tpot_pct_error',
        'E2E Latency': 'e2e_latency_pct_error'
    }
    
    for metric_name, col in metrics.items():
        if col in comparison.columns:
            mape = comparison[col].abs().mean()
            rmse = np.sqrt((comparison[col] ** 2).mean())
            median_error = comparison[col].median()
            lines.append(f"{metric_name:15} MAPE: {mape:6.1f}%  |  Median Error: {median_error:7.1f}%")
    
    lines.extend(["", "=" * 70, "By Framework", "=" * 70, ""])
    
    if 'framework' in comparison.columns:
        for framework in sorted(comparison['framework'].unique()):
            fw_data = comparison[comparison['framework'] == framework]
            mape = fw_data['throughput_pct_error'].abs().mean()
            lines.append(f"{framework:10} MAPE: {mape:6.1f}%  ({len(fw_data)} configs)")
    
    lines.extend(["", "=" * 70, "By Parallelism (TP)", "=" * 70, ""])
    
    for tp in sorted(comparison['tensor_parallel'].unique()):
        tp_data = comparison[comparison['tensor_parallel'] == tp]
        mape = tp_data['throughput_pct_error'].abs().mean()
        lines.append(f"TP={int(tp):2} MAPE: {mape:6.1f}%  ({len(tp_data)} configs)")
    
    lines.extend(["", "=" * 70, "By Context Length", "=" * 70, ""])
    
    for isl in sorted(comparison['input_seq_len'].unique()):
        for osl in sorted(comparison['output_seq_len'].unique()):
            ctx_data = comparison[
                (comparison['input_seq_len'] == isl) & 
                (comparison['output_seq_len'] == osl)
            ]
            if len(ctx_data) > 0:
                mape = ctx_data['throughput_pct_error'].abs().mean()
                lines.append(f"ISL={int(isl):4}/OSL={int(osl):4}  MAPE: {mape:6.1f}%  ({len(ctx_data)} configs)")
    
    lines.extend(["", "=" * 70, "Top 5 Best Predictions (lowest absolute error)", "=" * 70, ""])
    
    best_5 = comparison.nsmallest(5, 'throughput_pct_error', keep='all')[
        ['input_seq_len', 'output_seq_len', 'tensor_parallel', 'concurrency', 
         'actual_throughput', 'predicted_throughput_per_gpu', 'throughput_pct_error']
    ]
    for idx, row in best_5.iterrows():
        lines.append(
            f"ISL={int(row['input_seq_len']):4}/OSL={int(row['output_seq_len']):4} "
            f"TP={int(row['tensor_parallel'])} Conc={int(row['concurrency']):3}  "
            f"Actual={row['actual_throughput']:8.1f}  Pred={row['predicted_throughput_per_gpu']:8.1f}  "
            f"Error={row['throughput_pct_error']:6.1f}%"
        )
    
    lines.extend(["", "=" * 70, "Top 5 Worst Predictions (highest absolute error)", "=" * 70, ""])
    
    comparison['throughput_abs_pct_error'] = comparison['throughput_pct_error'].abs()
    worst_5 = comparison.nlargest(5, 'throughput_abs_pct_error')[
        ['input_seq_len', 'output_seq_len', 'tensor_parallel', 'concurrency',
         'actual_throughput', 'predicted_throughput_per_gpu', 'throughput_pct_error']
    ]
    for idx, row in worst_5.iterrows():
        lines.append(
            f"ISL={int(row['input_seq_len']):4}/OSL={int(row['output_seq_len']):4} "
            f"TP={int(row['tensor_parallel'])} Conc={int(row['concurrency']):3}  "
            f"Actual={row['actual_throughput']:8.1f}  Pred={row['predicted_throughput_per_gpu']:8.1f}  "
            f"Error={row['throughput_pct_error']:6.1f}%"
        )
    
    lines.append("")
    
    # Write to file and print
    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text)
    print(f"\n📊 Summary Statistics:\n{summary_text}")


def main():
    parser = argparse.ArgumentParser(description='Compare Mantile predictions with InferenceMAX actuals')
    parser.add_argument(
        '--actuals',
        type=Path,
        default=Path('reconcile/by_model/openai_GPT-OSS-120B/inferencemax_b200_only.csv'),
        help='InferenceMAX actuals CSV'
    )
    parser.add_argument(
        '--predictions',
        type=Path,
        default=Path('reconcile/by_model/openai_GPT-OSS-120B/mantile_predictions.csv'),
        help='Mantile predictions CSV'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('reconcile/by_model/openai_GPT-OSS-120B/comparison_report.csv'),
        help='Output comparison CSV'
    )
    
    args = parser.parse_args()
    
    compare_predictions(args.actuals, args.predictions, args.output)


if __name__ == '__main__':
    main()
