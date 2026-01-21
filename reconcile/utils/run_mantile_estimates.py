#!/usr/bin/env python3
"""
Batch Estimator: Run Mantile predictions for InferenceMAX configs

Processes filtered InferenceMAX CSV, calls Mantile API for each config,
and saves predictions for comparison.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import requests
from tqdm import tqdm

from inferencemax_to_mantile import inferencemax_row_to_mantile_request


def call_mantile_api(request: dict, api_url: str, timeout: int = 30) -> Optional[dict]:
    """
    Call Mantile API with request payload.
    
    Args:
        request: Mantile API request dict
        api_url: Base API URL (e.g., http://localhost:8000)
        timeout: Request timeout in seconds
        
    Returns:
        API response dict or None if error
    """
    try:
        response = requests.post(
            f"{api_url}/config/system-metrics",
            json=request,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def extract_metrics(api_response: dict) -> dict:
    """Extract relevant metrics from Mantile API response."""
    if 'error' in api_response:
        return {
            'predicted_throughput_per_gpu': None,
            'predicted_ttft_sec': None,
            'predicted_tpot_sec': None,
            'predicted_e2e_latency_sec': None,
            'predicted_tps_user': None,
            'memory_weight_gb': None,
            'memory_activation_gb': None,
            'memory_kv_cache_gb': None,
            'fits_on_hardware': None,
            'bottleneck': None,
            'error_message': api_response['error']
        }
    
    # Extract metrics from actual Mantile response format
    memory = api_response.get('memory', {})
    system = api_response.get('system', {})
    
    return {
        'predicted_throughput_per_gpu': api_response.get('throughput_tokens_s'),
        'predicted_ttft_sec': api_response.get('ttft_ms', 0) / 1000,
        'predicted_tpot_sec': api_response.get('tpot_ms', 0) / 1000,
        'predicted_e2e_latency_sec': api_response.get('total_latency_ms', 0) / 1000,
        'predicted_tps_user': api_response.get('tps_user'),
        'memory_weight_gb': memory.get('weight_memory_gb'),
        'memory_activation_gb': memory.get('activation_memory_gb'),
        'memory_kv_cache_gb': memory.get('kv_cache_gb'),
        'fits_on_hardware': system.get('fits_on_hardware'),
        'bottleneck': system.get('bottleneck'),
        'error_message': None
    }


def process_batch(
    input_csv: Path,
    output_csv: Path,
    api_url: str,
    resume: bool = True
):
    """
    Process all configs in input CSV and save predictions.
    
    Args:
        input_csv: Path to filtered InferenceMAX CSV
        output_csv: Path to save predictions
        api_url: Mantile API base URL
        resume: If True, skip already processed rows
    """
    # Load input data
    df = pd.read_csv(input_csv)
    print(f"📊 Loaded {len(df)} configurations from {input_csv.name}")
    
    # Check for existing output to resume
    if resume and output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        print(f"📋 Found {len(existing_df)} existing predictions, resuming...")
        # Create a set of already processed configs
        processed_configs = set(
            tuple(row[['gpu_model', 'input_seq_len', 'output_seq_len', 
                      'tensor_parallel', 'expert_parallel', 'concurrency']].values)
            for _, row in existing_df.iterrows()
        )
    else:
        existing_df = None
        processed_configs = set()
    
    # Prepare output file
    results = []
    
    # Process each configuration
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing configs"):
        # Check if already processed
        config_key = tuple(row[['gpu_model', 'input_seq_len', 'output_seq_len',
                                'tensor_parallel', 'expert_parallel', 'concurrency']].values)
        if config_key in processed_configs:
            continue
        
        # Convert to Mantile request
        try:
            request = inferencemax_row_to_mantile_request(row.to_dict())
        except Exception as e:
            results.append({
                **row.to_dict(),
                'predicted_throughput_per_gpu': None,
                'predicted_ttft_sec': None,
                'predicted_tpot_sec': None,
                'predicted_e2e_latency_sec': None,
                'error_message': f'Conversion error: {str(e)}'
            })
            continue
        
        # Call API
        start_time = time.time()
        api_response = call_mantile_api(request, api_url)
        api_time_ms = (time.time() - start_time) * 1000
        
        # Extract metrics
        metrics = extract_metrics(api_response)
        
        # Combine with input config
        result = {
            **row.to_dict(),
            **metrics,
            'api_response_time_ms': api_time_ms
        }
        results.append(result)
        
        # Save incrementally every 10 rows
        if len(results) % 10 == 0:
            save_results(results, existing_df, output_csv)
    
    # Final save
    save_results(results, existing_df, output_csv)
    
    print(f"\n✅ Processed {len(results)} new configurations")
    print(f"💾 Saved predictions to {output_csv}")


def save_results(new_results: list, existing_df: Optional[pd.DataFrame], output_path: Path):
    """Save results, combining with existing if resuming."""
    new_df = pd.DataFrame(new_results)
    
    if existing_df is not None:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    combined_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Run Mantile estimates for InferenceMAX configs')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('reconcile/by_model/openai_GPT-OSS-120B/inferencemax_b200_only.csv'),
        help='Input CSV with InferenceMAX configs'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('reconcile/by_model/openai_GPT-OSS-120B/mantile_predictions.csv'),
        help='Output CSV for predictions'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='Mantile API base URL'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignore existing predictions'
    )
    
    args = parser.parse_args()
    
    # Check if backend is reachable
    try:
        response = requests.get(f"{args.api_url}/docs", timeout=5)
        print(f"✅ Mantile backend reachable at {args.api_url}")
    except:
        print(f"⚠️  Warning: Could not reach Mantile backend at {args.api_url}")
        print(f"   Make sure backend is running: cd backend && uvicorn main:app")
        return
    
    process_batch(
        args.input,
        args.output,
        args.api_url,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()
