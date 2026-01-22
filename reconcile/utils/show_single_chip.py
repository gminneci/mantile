#!/usr/bin/env python3
"""Show single-chip (TP=1, EP=1) comparison results."""

import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)

df = pd.read_csv('reconcile/by_model/openai_GPT-OSS-120B/mantile_predictions_new.csv')

# Filter for single chip (TP=1, EP=1)
single = df[(df['tensor_parallel'] == 1) & (df['expert_parallel'] == 1)].copy()

# Select relevant columns
cols = ['input_seq_len', 'output_seq_len', 'concurrency', 
        'throughput_tokens_per_sec_per_gpu', 'predicted_throughput_per_gpu',
        'mean_time_to_first_token_sec', 'predicted_ttft_sec',
        'mean_time_per_output_token_sec', 'predicted_tpot_sec']

single = single[cols].sort_values(['input_seq_len', 'output_seq_len', 'concurrency'])

# Rename for clarity
single.columns = ['ISL', 'OSL', 'Batch', 
                  'IM_Tput', 'Mantile_Tput',
                  'IM_TTFT', 'Mantile_TTFT',
                  'IM_TPOT', 'Mantile_TPOT']

# Add error column
single['Tput_Err%'] = ((single['Mantile_Tput'] - single['IM_Tput']) / single['IM_Tput'] * 100).round(1)

print('=' * 130)
print('Single Chip Results (TP=1, EP=1) - GPT-OSS-120B on B200')
print('=' * 130)
print()
print(f"{'ISL':>5} {'OSL':>5} {'Batch':>6} | {'IM_Tput':>10} {'Mantile':>10} {'Err%':>8} | {'IM_TTFT':>8} {'M_TTFT':>8} | {'IM_TPOT':>8} {'M_TPOT':>8}")
print('-' * 130)

for _, row in single.iterrows():
    print(f"{int(row['ISL']):>5} {int(row['OSL']):>5} {int(row['Batch']):>6} | "
          f"{row['IM_Tput']:>10.1f} {row['Mantile_Tput']:>10.1f} {row['Tput_Err%']:>7.1f}% | "
          f"{row['IM_TTFT']:>8.4f} {row['Mantile_TTFT']:>8.4f} | "
          f"{row['IM_TPOT']:>8.5f} {row['Mantile_TPOT']:>8.5f}")

print()
print(f"Total configs: {len(single)}")
print(f"Mean Throughput Error: {single['Tput_Err%'].abs().mean():.1f}%")
