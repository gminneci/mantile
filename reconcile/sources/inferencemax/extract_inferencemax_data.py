#!/usr/bin/env python3
"""
Script to extract GPT-OSS 120B performance data from InferenceMAX.

Since the website data is JavaScript-rendered, this script provides options to:
1. Use GitHub CLI to download benchmark artifacts (recommended)
2. Scrape rendered page data using Selenium (if needed)
3. Extract data from embedded JavaScript variables

For now, implements the GitHub artifact approach.

Caching Strategy:
- Downloads ALL benchmark data (not just GPT-OSS) to ./raw/ directory
- Skips download if file already exists (won't re-download unless file is missing)
- Filters for specific model (GPT-OSS) only when creating the output CSV
- This allows future extraction of other models without re-downloading artifacts

Usage:
    # Single run
    python extract_inferencemax_data.py 21012127808
    
    # Multiple runs (merges, keeps most recent)
    python extract_inferencemax_data.py 21012127808 20764588467 20832485469
    
    # Auto-collect from multiple recent runs
    python extract_inferencemax_data.py --collect-recent 5
    
    # Force re-download (delete raw/ directory first)
    rm -rf raw/ && python extract_inferencemax_data.py --collect-recent 30
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd


def get_latest_benchmark_run() -> Optional[str]:
    """Find the latest successful benchmark run ID using gh CLI."""
    try:
        result = subprocess.run(
            [
                "gh", "run", "list",
                "--repo", "InferenceMAX/InferenceMAX",
                "--workflow", "run-sweep.yml",
                "--status", "success",
                "--limit", "1",
                "--json", "databaseId"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        if data:
            return str(data[0]["databaseId"])
        return None
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"Error finding latest run: {e}", file=sys.stderr)
        return None


def get_recent_benchmark_runs(limit: int = 10) -> List[Tuple[str, str]]:
    """Get recent successful benchmark run IDs and dates."""
    try:
        result = subprocess.run(
            [
                "gh", "run", "list",
                "--repo", "InferenceMAX/InferenceMAX",
                "--workflow", "run-sweep.yml",
                "--status", "success",
                "--limit", str(limit),
                "--json", "databaseId,createdAt"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        return [(str(run["databaseId"]), run["createdAt"]) for run in data]
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        print(f"Error finding recent runs: {e}", file=sys.stderr)
        return []


def download_benchmark_data(run_id: str, output_dir: Path) -> bool:
    """Download benchmark results artifact from GitHub Actions.
    
    Re-downloads if remote artifact is newer than cached file.
    Downloads ALL benchmark data, not just specific models.
    """
    # Check if we already have the data cached
    json_files = list(output_dir.rglob("agg_bmk.json"))
    
    if json_files:
        # Check if remote artifact is newer than cached file
        try:
            result = subprocess.run(
                ["gh", "api", f"/repos/InferenceMAX/InferenceMAX/actions/runs/{run_id}/artifacts"],
                capture_output=True,
                text=True,
                check=True
            )
            artifacts = json.loads(result.stdout).get("artifacts", [])
            results_bmk = next((a for a in artifacts if a["name"] == "results_bmk"), None)
            
            if results_bmk:
                # Compare timestamps
                remote_updated = datetime.fromisoformat(results_bmk["updated_at"].replace("Z", "+00:00"))
                local_mtime = datetime.fromtimestamp(json_files[0].stat().st_mtime).astimezone()
                
                if remote_updated > local_mtime:
                    # Remote is newer, re-download
                    pass  # Fall through to download
                else:
                    return True  # Cached version is up to date
            else:
                return True  # No artifact found, use cached
        except:
            # If checking fails, assume cached is fine
            return True
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "gh", "run", "download", run_id,
                "--repo", "InferenceMAX/InferenceMAX",
                "-n", "results_bmk",
                "-D", str(output_dir)
            ],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        # Silently fail - some runs don't have aggregated results
        return False


def filter_gptoss_data(data: List[Dict]) -> List[Dict]:
    """Filter for GPT-OSS 120B results."""
    return [
        entry for entry in data
        if entry.get("infmax_model_prefix") == "gptoss"
    ]


def convert_to_csv(json_path: Path, output_csv: Path):
    """Convert aggregated JSON results to CSV format."""
    with open(json_path) as f:
        data = json.load(f)
    
    # Filter for GPT-OSS
    gptoss_data = filter_gptoss_data(data)
    
    if not gptoss_data:
        print("No GPT-OSS data found in results", file=sys.stderr)
        return
    
    # Create DataFrame with key columns
    df = pd.DataFrame(gptoss_data)
    
    # Select and rename columns for clarity
    columns_of_interest = {
        'hw': 'gpu_model',
        'framework': 'framework',
        'precision': 'precision',
        'isl': 'input_seq_len',
        'osl': 'output_seq_len',
        'tp': 'tensor_parallel',
        'ep': 'expert_parallel',
        'conc': 'concurrency',
        'tput_per_gpu': 'throughput_tokens_per_sec_per_gpu',
        'output_tput_per_gpu': 'output_throughput_per_gpu',
        'mean_ttft': 'mean_time_to_first_token_sec',
        'p99_ttft': 'p99_time_to_first_token_sec',
        'mean_tpot': 'mean_time_per_output_token_sec',
        'mean_e2el': 'mean_end_to_end_latency_sec',
        'median_e2el': 'median_end_to_end_latency_sec',
    }
    
    # Filter columns that exist
    available_cols = {k: v for k, v in columns_of_interest.items() if k in df.columns}
    df_filtered = df[list(available_cols.keys())].rename(columns=available_cols)
    
    # Sort by GPU model and configuration
    df_filtered = df_filtered.sort_values(
        ['gpu_model', 'precision', 'input_seq_len', 'output_seq_len', 'tensor_parallel']
    )
    
    df_filtered.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df_filtered)} GPT-OSS 120B results to {output_csv}")
    print(f"\nGPU Models found: {sorted(df_filtered['gpu_model'].unique())}")
    print(f"Precisions: {sorted(df_filtered['precision'].unique())}")
    print(f"Context lengths (ISL/OSL): {sorted(df_filtered[['input_seq_len', 'output_seq_len']].drop_duplicates().values.tolist())}")


def extract_from_run(run_id: str, run_date: str) -> Optional[pd.DataFrame]:
    """Extract data from a single run, returning DataFrame with run metadata.
    
    Downloads ALL data if not cached, but filters for GPT-OSS when extracting.
    """
    output_dir = Path(f"./raw/run_{run_id}")
    
    # Download if not cached (will skip if already exists)
    if not download_benchmark_data(run_id, output_dir):
        return None
    
    # Find the aggregated JSON file
    json_files = list(output_dir.rglob("agg_bmk.json"))
    if not json_files:
        return None
    
    with open(json_files[0]) as f:
        data = json.load(f)
    
    # Filter for GPT-OSS (filtering happens here, not during download)
    gptoss_data = filter_gptoss_data(data)
    if not gptoss_data:
        return None
    
    df = pd.DataFrame(gptoss_data)
    
    # Add run metadata
    df['run_id'] = run_id
    df['run_date'] = run_date
    
    return df


def merge_and_deduplicate(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple dataframes and keep only most recent for each config."""
    if not dataframes:
        return pd.DataFrame()
    
    # Combine all data
    combined = pd.concat(dataframes, ignore_index=True)
    
    # Define configuration columns (everything except metrics and metadata)
    config_cols = [
        'hw', 'framework', 'precision', 'isl', 'osl', 'tp', 'ep', 
        'conc', 'spec_decoding', 'disagg', 'dp_attention'
    ]
    config_cols = [c for c in config_cols if c in combined.columns]
    
    # Sort by date (newest first) and drop duplicates keeping first (newest)
    combined = combined.sort_values('run_date', ascending=False)
    deduplicated = combined.drop_duplicates(subset=config_cols, keep='first')
    
    return deduplicated


def main():
    """Main extraction workflow."""
    print("🔍 Extracting GPT-OSS 120B data from InferenceMAX...")
    
    # Check for gh CLI
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ GitHub CLI (gh) not found. Install with: brew install gh", file=sys.stderr)
        print("   Then authenticate: gh auth login", file=sys.stderr)
        sys.exit(1)
    
    # Parse arguments
    run_ids = []
    collect_recent = None
    
    for arg in sys.argv[1:]:
        if arg.startswith("--collect-recent"):
            if "=" in arg:
                collect_recent = int(arg.split("=")[1])
            else:
                collect_recent = 10  # default
        elif arg.isdigit():
            run_ids.append(arg)
    
    # Determine which runs to process
    if collect_recent:
        print(f"🔎 Finding {collect_recent} most recent successful benchmark runs...")
        runs = get_recent_benchmark_runs(collect_recent)
        if not runs:
            print("❌ Could not find recent runs", file=sys.stderr)
            sys.exit(1)
        print(f"✅ Found {len(runs)} runs")
    elif run_ids:
        # Get dates for provided run IDs
        runs = []
        for run_id in run_ids:
            try:
                result = subprocess.run(
                    ["gh", "api", f"/repos/InferenceMAX/InferenceMAX/actions/runs/{run_id}"],
                    capture_output=True, text=True, check=True
                )
                run_data = json.loads(result.stdout)
                runs.append((run_id, run_data.get("created_at", "")))
            except:
                runs.append((run_id, ""))
    else:
        # Auto-detect latest
        print("🔎 Finding latest successful benchmark run...")
        run_id = get_latest_benchmark_run()
        if not run_id:
            print("❌ Could not find latest benchmark run", file=sys.stderr)
            print("   Try: python extract_inferencemax_data.py <RUN_ID>", file=sys.stderr)
            sys.exit(1)
        try:
            result = subprocess.run(
                ["gh", "api", f"/repos/InferenceMAX/InferenceMAX/actions/runs/{run_id}"],
                capture_output=True, text=True, check=True
            )
            run_data = json.loads(result.stdout)
            runs = [(run_id, run_data.get("created_at", ""))]
        except:
            runs = [(run_id, "")]
        print(f"✅ Found run: {run_id}")
    
    # Extract data from all runs
    print(f"\n📥 Processing {len(runs)} run(s)... (skipping already cached downloads)")
    dataframes = []
    successful_runs = 0
    
    for run_id, run_date in runs:
        output_dir = Path(f"./raw/run_{run_id}")
        cached = (output_dir / "agg_bmk.json").exists() or list(output_dir.rglob("agg_bmk.json"))
        cache_status = "💾" if cached else "⬇️"
        
        print(f"  {cache_status} Processing run {run_id} ({run_date[:10]})...", end=" ")
        df = extract_from_run(run_id, run_date)
        if df is not None and len(df) > 0:
            gptoss_count = len(df)
            print(f"✅ {gptoss_count} GPT-OSS results")
            dataframes.append(df)
            successful_runs += 1
        else:
            print("⏭️  No GPT-OSS data")
    
    if not dataframes:
        print("\n❌ No data extracted from any runs", file=sys.stderr)
        print("   Try different run IDs or check:", file=sys.stderr)
        print("   https://github.com/InferenceMAX/InferenceMAX/actions/workflows/run-sweep.yml", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n✅ Successfully extracted from {successful_runs}/{len(runs)} runs")
    
    # Merge and deduplicate
    print("🔄 Merging data and keeping most recent results...")
    merged = merge_and_deduplicate(dataframes)
    
    # Select and rename columns for final output
    columns_of_interest = {
        'hw': 'gpu_model',
        'framework': 'framework',
        'precision': 'precision',
        'isl': 'input_seq_len',
        'osl': 'output_seq_len',
        'tp': 'tensor_parallel',
        'ep': 'expert_parallel',
        'conc': 'concurrency',
        'tput_per_gpu': 'throughput_tokens_per_sec_per_gpu',
        'output_tput_per_gpu': 'output_throughput_per_gpu',
        'mean_ttft': 'mean_time_to_first_token_sec',
        'p99_ttft': 'p99_time_to_first_token_sec',
        'mean_tpot': 'mean_time_per_output_token_sec',
        'mean_e2el': 'mean_end_to_end_latency_sec',
        'median_e2el': 'median_end_to_end_latency_sec',
        'run_date': 'benchmark_date',
        'run_id': 'run_id',
    }
    
    available_cols = {k: v for k, v in columns_of_interest.items() if k in merged.columns}
    final_df = merged[list(available_cols.keys())].rename(columns=available_cols)
    
    # Sort by GPU model and configuration
    sort_cols = ['gpu_model', 'precision', 'input_seq_len', 'output_seq_len', 'tensor_parallel']
    sort_cols = [c for c in sort_cols if c in final_df.columns]
    final_df = final_df.sort_values(sort_cols)
    
    # Save to CSV in model-specific directory
    output_csv = Path("../../by_model/openai_GPT-OSS-120B/inferencemax.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Saved {len(final_df)} unique configurations to {output_csv}")
    print(f"\n📊 Data Summary:")
    if 'gpu_model' in final_df.columns:
        print(f"   GPUs: {sorted(final_df['gpu_model'].unique())}")
    if 'precision' in final_df.columns:
        print(f"   Precisions: {sorted(final_df['precision'].unique())}")
    if 'input_seq_len' in final_df.columns and 'output_seq_len' in final_df.columns:
        contexts = sorted(final_df[['input_seq_len', 'output_seq_len']].drop_duplicates().values.tolist())
        print(f"   Context configs (ISL/OSL): {contexts}")
    
    print(f"\n💡 Preview: head {output_csv}")
    print(f"   Total raw results before dedup: {sum(len(df) for df in dataframes)}")
    print(f"   Unique configs after dedup: {len(final_df)}")


if __name__ == "__main__":
    main()
