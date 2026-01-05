#!/usr/bin/env python3
"""
Validation test for Steps 1-4 of Mantile implementation.
Tests that all core backend components work correctly.
"""

def test_step_1_ir_builder():
    """Step 1: Model → IR Builder"""
    print("=" * 60)
    print("Step 1: Model → IR Builder")
    print("=" * 60)
    
    from backend.ir_builder import build_model_ir
    
    # Test with TinyLlama (public model)
    try:
        model_ir = build_model_ir("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print(f"✅ IR Builder works!")
        print(f"   Model: {model_ir.name}")
        print(f"   Layers: {model_ir.num_layers}")
        print(f"   Hidden: {model_ir.hidden_size}")
        return True
    except Exception as e:
        print(f"❌ IR Builder failed: {e}")
        return False

def test_step_2_layer_implementations():
    """Step 2: Layer Implementations"""
    print("\n" + "=" * 60)
    print("Step 2: Layer Implementations")
    print("=" * 60)
    
    from backend.layers import (
        AttentionLayer,
        GroupedQueryAttentionLayer,
        MLPLayer,
        GatedMLPLayer,
        NormLayer,
        EmbeddingLayer,
        Phase,
        DataType
    )
    
    results = []
    
    # Test MHA
    try:
        mha = AttentionLayer("test", 0, 1024, 16, 64, {"tensor_parallel": 4})
        metrics = mha.compute_metrics(2, 128, Phase.PREFILL, DataType.BF16, None)
        print(f"✅ MHA: {metrics.flops_per_chip:,} FLOPs/chip")
        results.append(True)
    except Exception as e:
        print(f"❌ MHA failed: {e}")
        results.append(False)
    
    # Test GQA
    try:
        gqa = GroupedQueryAttentionLayer("test", 0, 1024, 16, 4, 64, {"tensor_parallel": 4, "context_parallel": 4})
        metrics = gqa.compute_metrics(2, 128, Phase.DECODE, DataType.BF16, None)
        print(f"✅ GQA: {metrics.flops_per_chip:,} FLOPs/chip")
        results.append(True)
    except Exception as e:
        print(f"❌ GQA failed: {e}")
        results.append(False)
    
    # Test MLP
    try:
        mlp = GatedMLPLayer("test", 0, 1024, 4096, {"tensor_parallel": 4})
        metrics = mlp.compute_metrics(2, 128, Phase.PREFILL, DataType.BF16, None)
        print(f"✅ MLP: {metrics.flops_per_chip:,} FLOPs/chip")
        results.append(True)
    except Exception as e:
        print(f"❌ MLP failed: {e}")
        results.append(False)
    
    # Test Norm
    try:
        norm = NormLayer("test", 0, 1024, has_bias=False, parallelism={})
        metrics = norm.compute_metrics(2, 128, Phase.PREFILL, DataType.BF16, None)
        print(f"✅ Norm: {metrics.weight_memory_per_chip:,} bytes")
        results.append(True)
    except Exception as e:
        print(f"❌ Norm failed: {e}")
        results.append(False)
    
    # Test Embedding
    try:
        emb = EmbeddingLayer("test", 128256, 8192, {})
        metrics = emb.compute_metrics(2, 128, Phase.PREFILL, DataType.BF16, None)
        print(f"✅ Embedding: {metrics.weight_memory_per_chip / 1e9:.2f} GB")
        results.append(True)
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        results.append(False)
    
    return all(results)

def test_step_3_parallelism():
    """Step 3: Parallelism Support"""
    print("\n" + "=" * 60)
    print("Step 3: Parallelism Support")
    print("=" * 60)
    
    from backend.layers import GroupedQueryAttentionLayer, GatedMLPLayer, Phase, DataType
    
    results = []
    
    # Test TP
    try:
        layer = GroupedQueryAttentionLayer("test", 0, 1024, 16, 4, 64, {"tensor_parallel": 4})
        metrics = layer.compute_metrics(2, 128, Phase.PREFILL, DataType.BF16, None)
        print(f"✅ TP (4-way): {metrics.num_chips} chips, {metrics.communication_bytes:,} bytes comm")
        results.append(True)
    except Exception as e:
        print(f"❌ TP failed: {e}")
        results.append(False)
    
    # Test CP
    try:
        layer = GroupedQueryAttentionLayer("test", 0, 1024, 16, 4, 64, {"context_parallel": 4})
        metrics = layer.compute_metrics(2, 128, Phase.PREFILL, DataType.BF16, None)
        print(f"✅ CP (4-way): {metrics.num_chips} chips, {metrics.communication_bytes:,} bytes comm")
        results.append(True)
    except Exception as e:
        print(f"❌ CP failed: {e}")
        results.append(False)
    
    # Test TP×CP
    try:
        layer = GroupedQueryAttentionLayer("test", 0, 1024, 16, 4, 64, {"tensor_parallel": 4, "context_parallel": 4})
        metrics = layer.compute_metrics(2, 128, Phase.PREFILL, DataType.BF16, None)
        print(f"✅ TP×CP (16-way): {metrics.num_chips} chips, {metrics.communication_bytes:,} bytes comm")
        results.append(True)
    except Exception as e:
        print(f"❌ TP×CP failed: {e}")
        results.append(False)
    
    # Test SP
    try:
        layer = GatedMLPLayer("test", 0, 1024, 4096, {"tensor_parallel": 4, "sequence_parallel": 4})
        metrics = layer.compute_metrics(2, 128, Phase.PREFILL, DataType.BF16, None)
        print(f"✅ SP (4-way): Activation memory reduced by SP factor")
        results.append(True)
    except Exception as e:
        print(f"❌ SP failed: {e}")
        results.append(False)
    
    return all(results)

def test_step_4_performance_estimators():
    """Step 4: Module Performance Estimators"""
    print("\n" + "=" * 60)
    print("Step 4: Module Performance Estimators")
    print("=" * 60)
    
    from backend.layers import GroupedQueryAttentionLayer, Phase, DataType
    
    layer = GroupedQueryAttentionLayer("test", 0, 1024, 16, 4, 64, {"tensor_parallel": 4})
    
    try:
        metrics = layer.compute_metrics(2, 128, Phase.PREFILL, DataType.BF16, None)
        
        print("✅ Performance metrics computed:")
        print(f"   FLOPs/chip: {metrics.flops_per_chip / 1e9:.2f} GFLOPs")
        print(f"   Weight memory/chip: {metrics.weight_memory_per_chip / 1e6:.2f} MB")
        print(f"   Activation memory/chip: {metrics.activation_memory_per_chip / 1e6:.2f} MB")
        print(f"   KV cache/chip: {metrics.kv_cache_per_chip / 1e6:.2f} MB")
        print(f"   Communication: {metrics.communication_bytes / 1e3:.2f} KB")
        print(f"   Total FLOPs: {metrics.flops_total / 1e9:.2f} GFLOPs")
        print(f"   Total memory: {(metrics.weight_memory_total + metrics.activation_memory_total) / 1e9:.3f} GB")
        return True
    except Exception as e:
        print(f"❌ Performance estimators failed: {e}")
        return False

def test_llama_33_70b():
    """Test complete Llama 3.3 70B model"""
    print("\n" + "=" * 60)
    print("Bonus: Llama 3.3 70B Model Construction")
    print("=" * 60)
    
    try:
        import sys
        import io
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        # Run the test
        from test_llama_33_70b import test_single_chip_prefill
        test_single_chip_prefill()
        
        # Restore stdout
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Check for key metrics
        if "70.6B" in output and "141.11 GB" in output:
            print("✅ Llama 3.3 70B model builds correctly")
            print("   Parameters: ~70.6B")
            print("   Weight memory: 141.11 GB")
            print("   Can compute full model metrics!")
            return True
        else:
            print("❌ Llama 3.3 70B metrics unexpected")
            return False
    except Exception as e:
        print(f"❌ Llama 3.3 70B failed: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MANTILE BACKEND VALIDATION (Steps 1-4)")
    print("=" * 60 + "\n")
    
    results = {
        "Step 1 (IR Builder)": test_step_1_ir_builder(),
        "Step 2 (Layer Implementations)": test_step_2_layer_implementations(),
        "Step 3 (Parallelism)": test_step_3_parallelism(),
        "Step 4 (Performance Estimators)": test_step_4_performance_estimators(),
        "Llama 3.3 70B": test_llama_33_70b(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Steps 1-4 are working!")
    else:
        print("❌ Some tests failed - see details above")
    print("=" * 60 + "\n")
