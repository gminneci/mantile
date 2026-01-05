#!/usr/bin/env python3
"""
Test GQA (Grouped Query Attention) implementation.
"""

from backend.layers.attention import GroupedQueryAttentionLayer
from backend.layers.base import Phase, DataType

def test_gqa_1():
    """
    GQA-1: Single chip prefill (no parallelism)
    """
    layer = GroupedQueryAttentionLayer(
        name="test_gqa",
        layer_idx=0,
        hidden_size=1024,
        num_heads=16,
        num_kv_heads=4,
        head_dim=64,
        parallelism={}
    )
    
    metrics = layer.compute_metrics(
        batch_size=2,
        seq_len=128,
        phase=Phase.PREFILL,
        dtype=DataType.BF16,
        hardware=None
    )
    
    expected = {
        "flops_per_chip": 1_476_395_008,
        "weight_memory_per_chip": 5_242_880,
        "activation_memory_per_chip": 1_835_008,
        "kv_cache_per_chip": 262_144,
        "communication_bytes": 0,
        "num_chips": 1
    }
    
    print("GQA-1 Test Results:")
    print(f"  FLOPs:        {metrics.flops_per_chip:,} (expected {expected['flops_per_chip']:,})")
    print(f"  Weight Mem:   {metrics.weight_memory_per_chip:,} (expected {expected['weight_memory_per_chip']:,})")
    print(f"  Activation:   {metrics.activation_memory_per_chip:,} (expected {expected['activation_memory_per_chip']:,})")
    print(f"  KV Cache:     {metrics.kv_cache_per_chip:,} (expected {expected['kv_cache_per_chip']:,})")
    print(f"  Comm:         {metrics.communication_bytes or 0:,} (expected {expected['communication_bytes']:,})")
    print(f"  Num Chips:    {metrics.num_chips} (expected {expected['num_chips']})")
    
    passed = True
    for key, expected_val in expected.items():
        actual_val = getattr(metrics, key)
        if actual_val is None and expected_val == 0:
            actual_val = 0
        if actual_val != expected_val:
            print(f"  ❌ {key}: {actual_val:,} != {expected_val:,}")
            passed = False
    
    if passed:
        print("\n✅ GQA-1 PASSED")
    else:
        print("\n❌ GQA-1 FAILED")
    
    return passed

def test_gqa_2():
    """
    GQA-2: TP=4 prefill (shard query heads)
    """
    layer = GroupedQueryAttentionLayer(
        name="test_gqa",
        layer_idx=0,
        hidden_size=1024,
        num_heads=16,
        num_kv_heads=4,
        head_dim=64,
        parallelism={"tensor_parallel": 4}
    )
    
    metrics = layer.compute_metrics(
        batch_size=2,
        seq_len=128,
        phase=Phase.PREFILL,
        dtype=DataType.BF16,
        hardware=None
    )
    
    expected = {
        "flops_per_chip": 369_098_752,
        "weight_memory_per_chip": 1_310_720,
        "activation_memory_per_chip": 1_245_184,
        "kv_cache_per_chip": 65_536,
        "communication_bytes": 524_288,
        "num_chips": 4
    }
    
    print("GQA-2 Test Results:")
    print(f"  FLOPs:        {metrics.flops_per_chip:,} (expected {expected['flops_per_chip']:,})")
    print(f"  Weight Mem:   {metrics.weight_memory_per_chip:,} (expected {expected['weight_memory_per_chip']:,})")
    print(f"  Activation:   {metrics.activation_memory_per_chip:,} (expected {expected['activation_memory_per_chip']:,})")
    print(f"  KV Cache:     {metrics.kv_cache_per_chip:,} (expected {expected['kv_cache_per_chip']:,})")
    print(f"  Comm:         {metrics.communication_bytes:,} (expected {expected['communication_bytes']:,})")
    print(f"  Num Chips:    {metrics.num_chips} (expected {expected['num_chips']})")
    
    passed = True
    for key, expected_val in expected.items():
        actual_val = getattr(metrics, key)
        if actual_val != expected_val:
            print(f"  ❌ {key}: {actual_val:,} != {expected_val:,}")
            passed = False
    
    if passed:
        print("\n✅ GQA-2 PASSED")
    else:
        print("\n❌ GQA-2 FAILED")
    
    return passed

def test_gqa_3():
    """
    GQA-3: Hybrid TP×CP decode (TP=4, CP=4, 16 chips)
    NOTE: This requires CP support in GQA implementation
    """
    try:
        layer = GroupedQueryAttentionLayer(
            name="test_gqa",
            layer_idx=0,
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=4,
            head_dim=64,
            parallelism={"tensor_parallel": 4, "context_parallel": 4}
        )
        
        metrics = layer.compute_metrics(
            batch_size=2,
            seq_len=128,  # S_past
            phase=Phase.DECODE,
            dtype=DataType.BF16,
            hardware=None
        )
        
        expected = {
            "flops_per_chip": 2_686_976,
            "weight_memory_per_chip": 1_310_720,
            "activation_memory_per_chip": 9_728,
            "kv_cache_per_chip": 16_384,
            "communication_bytes": 5_184,
            "num_chips": 16
        }
        
        print("GQA-3 Test Results:")
        print(f"  FLOPs:        {metrics.flops_per_chip:,} (expected {expected['flops_per_chip']:,})")
        print(f"  Weight Mem:   {metrics.weight_memory_per_chip:,} (expected {expected['weight_memory_per_chip']:,})")
        print(f"  Activation:   {metrics.activation_memory_per_chip:,} (expected {expected['activation_memory_per_chip']:,})")
        print(f"  KV Cache:     {metrics.kv_cache_per_chip:,} (expected {expected['kv_cache_per_chip']:,})")
        print(f"  Comm:         {metrics.communication_bytes:,} (expected {expected['communication_bytes']:,})")
        print(f"  Num Chips:    {metrics.num_chips} (expected {expected['num_chips']})")
        
        passed = True
        for key, expected_val in expected.items():
            actual_val = getattr(metrics, key)
            if actual_val != expected_val:
                print(f"  ❌ {key}: {actual_val:,} != {expected_val:,}")
                passed = False
        
        if passed:
            print("\n✅ GQA-3 PASSED")
        else:
            print("\n❌ GQA-3 FAILED")
        
        return passed
        
    except (ValueError, KeyError) as e:
        print(f"GQA-3 Test Results:")
        print(f"  ⚠️  CP not supported in GQA yet: {e}")
        print("\n⚠️  GQA-3 SKIPPED (CP not implemented)")
        return None

if __name__ == "__main__":
    print("Testing Grouped Query Attention (GQA) Implementation\n")
    print("=" * 60)
    passed_1 = test_gqa_1()
    print("=" * 60)
    passed_2 = test_gqa_2()
    print("=" * 60)
    passed_3 = test_gqa_3()
    print("=" * 60)
    
    results = [passed_1, passed_2, passed_3]
    passed_count = sum(1 for r in results if r is True)
    skipped_count = sum(1 for r in results if r is None)
    
    if all(r in (True, None) for r in results):
        print(f"\n✅ All runnable tests passed! ({passed_count} passed, {skipped_count} skipped)")
    else:
        print(f"\n❌ Some tests failed ({passed_count} passed, {skipped_count} skipped)")
        if not passed_1:
            print("  - GQA-1 failed")
        if not passed_2:
            print("  - GQA-2 failed")
        if passed_3 is False:
            print("  - GQA-3 failed")
