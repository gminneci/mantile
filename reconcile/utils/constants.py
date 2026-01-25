#!/usr/bin/env python3
"""
Shared constants for reconciliation utilities.
"""

# Configuration columns that uniquely identify a benchmark config
CONFIG_COLUMNS = [
    'gpu_model',
    'input_seq_len',
    'output_seq_len',
    'tensor_parallel',
    'expert_parallel',
    'concurrency'
]

# Metric definitions for comparison: (short_name, actual_col, predicted_col)
COMPARISON_METRICS = [
    ('throughput', 'actual_throughput', 'predicted_throughput_per_gpu'),
    ('ttft', 'actual_ttft', 'predicted_ttft_sec'),
    ('tpot', 'actual_tpot', 'predicted_tpot_sec'),
    ('e2e_latency', 'actual_e2e_latency', 'predicted_e2e_latency_sec')
]

# Metric display names for summary reports
METRIC_DISPLAY_NAMES = {
    'throughput_pct_error': 'Throughput',
    'ttft_pct_error': 'TTFT',
    'tpot_pct_error': 'TPOT',
    'e2e_latency_pct_error': 'E2E Latency'
}
