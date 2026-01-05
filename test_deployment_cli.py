#!/usr/bin/env python3
"""
CLI tool to test the interactive configuration flow and deployment API.
This demonstrates the modular backend without needing the UI.
"""

from backend.config_service import ConfigurationService
from backend.layers import Phase, DataType


def test_interactive_flow():
    """Test the 5-step interactive configuration flow."""
    print("=" * 70)
    print("INTERACTIVE CONFIGURATION FLOW TEST (CLI)")
    print("=" * 70)
    
    # Initialize service
    service = ConfigurationService()
    
    # Step 1: Load Model + Hardware
    print("\n📦 Step 1: Load Model + Hardware")
    print("-" * 70)
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hardware_config = "nvidia_gb200_single"
    
    print(f"Loading model: {model_id}")
    model_ir = service.load_model(model_id)
    print(f"✅ Model loaded: {model_ir.num_layers} layers, {model_ir.hidden_size} hidden")
    
    print(f"\nLoading hardware: {hardware_config}")
    hw = service.load_hardware(hardware_config)
    print(f"✅ Hardware: {hw.name}")
    print(f"   BF16: {hw.bf16_tflops:,.0f} TFLOPs")
    print(f"   HBM: {hw.hbm_capacity_gb:.0f} GB @ {hw.hbm_bandwidth_gbps:,.0f} GB/s")
    
    # Step 2: Validate Model
    print("\n🔍 Step 2: Validate Model")
    print("-" * 70)
    
    validation = service.validate_model()
    print(f"✅ Valid: {validation.valid}")
    print(f"   Total params: {validation.total_params:,}")
    print(f"   Layers: {validation.num_layers}")
    print(f"   Attention: {validation.attention_type}")
    print(f"   MLP: {validation.mlp_type}")
    if validation.issues:
        print(f"   Issues: {validation.issues}")
    
    # Step 3: Configure Per-Layer Parallelism
    print("\n⚙️  Step 3: Configure Per-Layer Parallelism")
    print("-" * 70)
    
    layer_types = service.get_layer_types()
    print(f"Available layer types: {layer_types}")
    
    # Configure parallelism for each layer type
    configs = {
        "attention": {"tensor_parallel": 2, "context_parallel": 1},
        "feedforward": {"tensor_parallel": 4, "sequence_parallel": 1},
        "norm": {"tensor_parallel": 1},
        "embedding": {"tensor_parallel": 1},
    }
    
    for layer_type in layer_types:
        if layer_type in configs:
            cfg = configs[layer_type]
            service.configure_layer_parallelism(
                layer_type=layer_type,
                tensor_parallel=cfg.get("tensor_parallel", 1),
                context_parallel=cfg.get("context_parallel", 1),
                sequence_parallel=cfg.get("sequence_parallel", 1)
            )
            layer_config = service.get_layer_config(layer_type)
            print(f"✅ {layer_type}: {layer_config.parallelism}, "
                  f"{layer_config.num_instances} instances")
        else:
            # Default config
            service.configure_layer_parallelism(layer_type=layer_type)
            print(f"✅ {layer_type}: default (no parallelism)")
    
    # Step 4: Calculate Minimum System
    print("\n💾 Step 4: Calculate Minimum System Requirements")
    print("-" * 70)
    
    batch_size = 1
    seq_length = 2048
    
    requirements = service.calculate_minimum_system(
        batch_size=batch_size,
        seq_length=seq_length,
        phase=Phase.PREFILL,
        dtype=DataType.BF16
    )
    
    print(f"Configuration: batch={batch_size}, seq={seq_length}")
    print(f"✅ Minimum chips required: {requirements.min_chips}")
    print(f"   Weight memory: {requirements.total_weight_memory_gb:.2f} GB")
    print(f"   Activation memory: {requirements.total_activation_memory_gb:.2f} GB")
    print(f"   KV cache: {requirements.total_kv_cache_gb:.2f} GB")
    print(f"   Memory per chip: {requirements.memory_per_chip_gb:.2f} GB")
    print(f"   HW capacity: {requirements.hw_capacity_gb:.0f} GB")
    print(f"   Fits: {'✅ Yes' if requirements.fits_on_hardware else '❌ No (OOM!)'}")
    
    # Step 5: Compute System Metrics
    print("\n📊 Step 5: Compute System-Level Metrics")
    print("-" * 70)
    
    output_seq = 128
    
    metrics = service.compute_system_metrics(
        batch_size=batch_size,
        input_seq=seq_length,
        output_seq=output_seq,
        dtype=DataType.BF16
    )
    
    print(f"Configuration: batch={batch_size}, input={seq_length}, output={output_seq}")
    print(f"\n⏱️  Latency:")
    print(f"   TTFT (prefill): {metrics['ttft_ms']:.2f} ms")
    print(f"   TPOT (decode): {metrics['tpot_ms']:.4f} ms")
    print(f"   Total latency: {metrics['total_latency_ms']:.2f} ms")
    
    print(f"\n🚀 Throughput:")
    print(f"   {metrics['throughput_tokens_s']:.0f} tokens/s")
    
    print(f"\n💾 Memory:")
    print(f"   Weights: {metrics['memory']['weight_memory_gb']:.2f} GB")
    print(f"   Activations: {metrics['memory']['activation_memory_gb']:.2f} GB")
    print(f"   KV cache: {metrics['memory']['kv_cache_gb']:.2f} GB")
    print(f"   Total: {metrics['memory']['total_memory_gb']:.2f} GB")
    print(f"   Per chip: {metrics['memory']['memory_per_chip_gb']:.2f} GB / {metrics['memory']['hw_capacity_gb']:.0f} GB")
    
    print(f"\n🎯 System:")
    print(f"   Chips: {metrics['system']['num_chips']}")
    print(f"   Bottleneck: {metrics['system']['bottleneck']}")
    print(f"   Fits: {'✅ Yes' if metrics['system']['fits_on_hardware'] else '❌ No'}")
    
    print(f"\n🔬 Detailed:")
    print(f"   Prefill compute: {metrics['prefill']['compute_time_ms']:.2f} ms")
    print(f"   Prefill memory: {metrics['prefill']['memory_time_ms']:.2f} ms")
    print(f"   Decode compute: {metrics['decode']['compute_time_ms']:.4f} ms")
    print(f"   Decode memory: {metrics['decode']['memory_time_ms']:.4f} ms")
    
    return True


def test_deployment_api_style():
    """
    Test the deployment-style API where you provide a complete config
    and get back all metrics in one shot.
    """
    print("\n\n" + "=" * 70)
    print("DEPLOYMENT API STYLE TEST (Single Call)")
    print("=" * 70)
    
    # This simulates what the /deployment/estimate endpoint does
    service = ConfigurationService()
    
    # Complete deployment config
    config = {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "hardware_config": "nvidia_gb200_single",
        "batch_size": 1,
        "input_seq": 2048,
        "output_seq": 128,
        "layer_parallelism": {
            "attention": {"tensor_parallel": 2},
            "feedforward": {"tensor_parallel": 4},
        }
    }
    
    print(f"\nDeployment Config:")
    print(f"  Model: {config['model_id']}")
    print(f"  Hardware: {config['hardware_config']}")
    print(f"  Batch: {config['batch_size']}, Input: {config['input_seq']}, Output: {config['output_seq']}")
    print(f"  Parallelism: {config['layer_parallelism']}")
    
    # Execute the full flow
    print(f"\n🔄 Executing full estimation flow...")
    
    # Load
    service.load_hardware(config['hardware_config'])
    service.load_model(config['model_id'])
    validation = service.validate_model()
    
    # Configure parallelism
    for layer_type in service.get_layer_types():
        if layer_type in config['layer_parallelism']:
            p = config['layer_parallelism'][layer_type]
            service.configure_layer_parallelism(
                layer_type=layer_type,
                tensor_parallel=p.get('tensor_parallel', 1),
                context_parallel=p.get('context_parallel', 1),
                sequence_parallel=p.get('sequence_parallel', 1)
            )
        else:
            service.configure_layer_parallelism(layer_type=layer_type)
    
    # Calculate
    requirements = service.calculate_minimum_system(
        batch_size=config['batch_size'],
        seq_length=config['input_seq']
    )
    
    metrics = service.compute_system_metrics(
        batch_size=config['batch_size'],
        input_seq=config['input_seq'],
        output_seq=config['output_seq']
    )
    
    # Display result (like API would return)
    print(f"\n✅ Deployment Estimate Complete!")
    print(f"\n📊 Key Metrics:")
    print(f"   TTFT: {metrics['ttft_ms']:.2f} ms")
    print(f"   TPOT: {metrics['tpot_ms']:.4f} ms")
    print(f"   Throughput: {metrics['throughput_tokens_s']:.0f} tok/s")
    print(f"   Min chips: {requirements.min_chips}")
    print(f"   Memory: {metrics['memory']['total_memory_gb']:.2f} GB")
    print(f"   Bottleneck: {metrics['system']['bottleneck']}")
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MANTILE CLI TEST SUITE")
    print("=" * 70)
    
    try:
        # Test 1: Interactive flow
        success1 = test_interactive_flow()
        
        # Test 2: Deployment API style
        success2 = test_deployment_api_style()
        
        # Summary
        print("\n\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'✅' if success1 else '❌'} Interactive Flow Test")
        print(f"{'✅' if success2 else '❌'} Deployment API Test")
        
        if success1 and success2:
            print("\n🎉 ALL TESTS PASSED!")
            print("\nThe backend is ready for:")
            print("  • Interactive UI integration (React)")
            print("  • API queries via /deployment/estimate endpoint")
            print("  • CLI-based usage (this script)")
        else:
            print("\n❌ Some tests failed")
        
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
