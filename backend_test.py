import sys
import os

# Add current directory to path so we can import backend
sys.path.append(os.getcwd())

try:
    from backend.hardware_library import get_nvl72_rack_specs
    from backend.ir_builder import build_llama_ir
    from backend.models import ParallelismConfig
    from backend.estimator import estimate_performance
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def run_test():
    print("--- Testing Mantile Backend Logic ---")
    
    # 1. Hardware
    hw = get_nvl72_rack_specs()
    print(f"Hardware: {hw.name}")
    print(f"  FP16 TFLOPS: {hw.fp16_tflops}")
    print(f"  HBM: {hw.hbm_capacity_gb} GB")
    
    # 2. Model (Test with a known small model to avoid huge download or just mock it)
    # We'll mock the IR if transformers is not set up, but let's try real first.
    # Using a small model ID that is Llama-like: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading Model IR for: {model_id}...")
    try:
        ir = build_llama_ir(model_id)
        print(f"  Hidden Size: {ir.hidden_size}")
        print(f"  Layers: {ir.num_layers}")
    except Exception as e:
        print(f"  Error loading model: {e}")
        return

    # 3. Estimates
    print("\n--- Running Estimation ---")
    par = ParallelismConfig(
        tp_size=1, # On a full rack? Maybe we want tp_size=72?
        # But parallel config implies how we slice it. If we treat the rack as a single resource usage,
        # tp_size=1 means "model is not split" (which is efficient communication-wise but maybe OOM).
        # But our estimator handles memory check.
        # Let's try tp_size=1 just to see math.
        batch_size=1,
        input_seq_len=128,
        output_seq_len=128
    )
    
    res = estimate_performance(hw, ir, par)
    
    print(f"Results:")
    print(f"  TTFT (ms): {res.time_to_first_token_ms:.4f}")
    print(f"  TPOT (ms): {res.time_per_output_token_ms:.4f}")
    print(f"  Tokens/s: {res.total_throughput_tokens_s:.2f}")
    print(f"  Total Mem (GB): {res.total_mem_gb:.2f} / {res.max_mem_capacity_gb}")
    print(f"  Bottleneck: {'Mem' if res.memory_bound_percent > 50 else 'Compute'}")

if __name__ == "__main__":
    run_test()
