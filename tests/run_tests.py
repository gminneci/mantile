#!/usr/bin/env python3
"""
Test Runner for Mantile Layer Tests

Runs all test cases documented in tests/*.md files to verify layer implementations.
"""

import sys
from pathlib import Path

# Add project root to path (parent of tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.layers import MLPLayer, AttentionLayer, GroupedQueryAttentionLayer, MoELayer, SlidingWindowAttentionLayer
from backend.layers.base import DataType


def test_mlp_layers():
    """Run MLP layer tests from tests/mlp_tests.md"""
    print("=" * 80)
    print("MLP LAYER TESTS")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 0: Vanilla FFN, single chip
    print("\n--- Test 0: Vanilla FFN, single chip ---")
    try:
        layer = MLPLayer(
            layer_idx=0,
            hidden_size=16,
            intermediate_size=64,
            dtype='bf16',
            parallelism={'tensor_parallel': 1}
        )
        metrics = layer.compute_metrics(batch_size=4, seq_len=8, phase='prefill')
        
        expected = {
            'flops_per_chip': 131_072,
            'weight_memory_per_chip': 4_096,
            'activation_memory_per_chip': 9_216
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test 1: TP=4
    print("\n--- Test 1: TP=4 ---")
    try:
        layer = MLPLayer(
            layer_idx=0,
            hidden_size=1024,
            intermediate_size=4096,
            dtype='bf16',
            parallelism={'tensor_parallel': 4}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 1_073_741_824,
            'weight_memory_per_chip': 16_777_216,
            'activation_memory_per_chip': 1_572_864
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test 2: SP=4
    print("\n--- Test 2: SP=4 ---")
    try:
        layer = MLPLayer(
            layer_idx=0,
            hidden_size=1024,
            intermediate_size=4096,
            dtype='bf16',
            parallelism={'sequence_parallel': 4}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 1_073_741_824,
            'weight_memory_per_chip': 16_777_216,
            'activation_memory_per_chip': 1_179_648
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test 3: TP=4, SP=2
    print("\n--- Test 3: TP=4, SP=2 ---")
    try:
        layer = MLPLayer(
            layer_idx=0,
            hidden_size=1024,
            intermediate_size=4096,
            dtype='bf16',
            parallelism={'tensor_parallel': 4, 'sequence_parallel': 2}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 536_870_912,
            'weight_memory_per_chip': 16_777_216,
            'activation_memory_per_chip': 786_432
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    print(f"\n--- MLP Tests Summary: {tests_passed} passed, {tests_failed} failed ---")
    return tests_passed, tests_failed


def test_attention_layers():
    """Run Attention layer tests from tests/attention_tests.md"""
    print("\n" + "=" * 80)
    print("ATTENTION LAYER TESTS")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test MHA-1: Standard Attention Prefill, single chip
    print("\n--- Test MHA-1: Standard Attention Prefill, single chip ---")
    try:
        layer = AttentionLayer(
            layer_idx=0,
            hidden_size=1024,
            num_heads=16,
            head_dim=64,
            dtype='bf16',
            parallelism={'tensor_parallel': 1}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 2_281_701_376,
            'weight_memory_per_chip': 8_388_608,
            'activation_memory_per_chip': 2_621_440,
            'kv_cache_per_chip': 1_048_576
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test MHA-2: Standard Attention Prefill, TP=4
    print("\n--- Test MHA-2: Standard Attention Prefill, TP=4 ---")
    try:
        layer = AttentionLayer(
            layer_idx=0,
            hidden_size=1024,
            num_heads=16,
            head_dim=64,
            dtype='bf16',
            parallelism={'tensor_parallel': 4}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 570_425_344,
            'weight_memory_per_chip': 2_097_152,
            'activation_memory_per_chip': 1_441_792,
            'kv_cache_per_chip': 262_144
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test MHA-3: Standard Attention Decode, TP=4
    print("\n--- Test MHA-3: Standard Attention Decode (1 token), TP=4 ---")
    try:
        layer = AttentionLayer(
            layer_idx=0,
            hidden_size=1024,
            num_heads=16,
            head_dim=64,
            dtype='bf16',
            parallelism={'tensor_parallel': 4}
        )
        # For decode: seq_len represents past context length (128 cached tokens)
        # batch_size=2, new tokens=1 (decode step)
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='decode')
        
        expected = {
            'flops_per_chip': 4_458_496,
            'weight_memory_per_chip': 2_097_152,
            'activation_memory_per_chip': 11_264,
            'kv_cache_per_chip': 264_192
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test GQA-1: GQA Prefill, single chip
    print("\n--- Test GQA-1: GQA Prefill, single chip ---")
    try:
        layer = GroupedQueryAttentionLayer(
            layer_idx=0,
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=4,
            head_dim=64,
            dtype='bf16',
            parallelism={'tensor_parallel': 1}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 1_476_395_008,
            'weight_memory_per_chip': 5_242_880,
            'activation_memory_per_chip': 1_835_008,
            'kv_cache_per_chip': 262_144
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test GQA-2: GQA Prefill, TP=4
    print("\n--- Test GQA-2: GQA Prefill, TP=4 ---")
    try:
        layer = GroupedQueryAttentionLayer(
            layer_idx=0,
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=4,
            head_dim=64,
            dtype='bf16',
            parallelism={'tensor_parallel': 4}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 369_098_752,
            'weight_memory_per_chip': 1_310_720,
            'activation_memory_per_chip': 1_245_184,
            'kv_cache_per_chip': 65_536
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    print(f"\n--- Attention Tests Summary: {tests_passed} passed, {tests_failed} failed ---")
    return tests_passed, tests_failed


def test_moe_layers():
    """Run MoE layer tests from tests/moe_tests.md"""
    print("\n" + "=" * 80)
    print("MOE LAYER TESTS")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test MOE-1: Single chip baseline
    print("\n--- Test MOE-1: MoE Single Chip (baseline) ---")
    try:
        layer = MoELayer(
            layer_idx=0,
            hidden_size=1024,
            intermediate_size=4096,
            num_experts=8,
            top_k=2,
            num_projections=2,
            dtype='bf16',
            parallelism={'expert_parallel': 1, 'tensor_parallel': 1}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 8_594_128_896,
            'weight_memory_per_chip': 134_234_112,
            'activation_memory_per_chip': 3_149_824,
            'kv_cache_per_chip': 0,
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test MOE-2: Expert Parallel (EP=4)
    print("\n--- Test MOE-2: Expert Parallel (EP=4) ---")
    try:
        layer = MoELayer(
            layer_idx=0,
            hidden_size=1024,
            intermediate_size=4096,
            num_experts=8,
            top_k=2,
            num_projections=2,
            dtype='bf16',
            parallelism={'expert_parallel': 4, 'tensor_parallel': 1}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 2_151_677_952,
            'weight_memory_per_chip': 33_570_816,
            'activation_memory_per_chip': 2_101_248,
            'kv_cache_per_chip': 0,
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test MOE-3: Expert Parallel (EP=8, one expert per chip)
    print("\n--- Test MOE-3: Expert Parallel (EP=8, one expert per chip) ---")
    try:
        layer = MoELayer(
            layer_idx=0,
            hidden_size=1024,
            intermediate_size=4096,
            num_experts=8,
            top_k=2,
            num_projections=2,
            dtype='bf16',
            parallelism={'expert_parallel': 8, 'tensor_parallel': 1}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 1_077_936_128,
            'weight_memory_per_chip': 16_793_600,
            'activation_memory_per_chip': 1_576_960,
            'kv_cache_per_chip': 0,
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test MOE-4: Tensor Parallel within Experts (TP=4)
    print("\n--- Test MOE-4: Tensor Parallel within Experts (TP=4) ---")
    try:
        layer = MoELayer(
            layer_idx=0,
            hidden_size=1024,
            intermediate_size=4096,
            num_experts=8,
            top_k=2,
            num_projections=2,
            dtype='bf16',
            parallelism={'expert_parallel': 1, 'tensor_parallel': 4}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 2_151_677_952,
            'weight_memory_per_chip': 33_570_816,
            'activation_memory_per_chip': 1_576_960,
            'kv_cache_per_chip': 0,
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test MOE-5: Hybrid Expert Parallel + Tensor Parallel (EP=4, TP=2)
    print("\n--- Test MOE-5: Hybrid EP=4, TP=2 ---")
    try:
        layer = MoELayer(
            layer_idx=0,
            hidden_size=1024,
            intermediate_size=4096,
            num_experts=8,
            top_k=2,
            num_projections=2,
            dtype='bf16',
            parallelism={'expert_parallel': 4, 'tensor_parallel': 2}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 1_077_936_128,
            'weight_memory_per_chip': 16_793_600,
            'activation_memory_per_chip': 1_576_960,
            'kv_cache_per_chip': 0,
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test MOE-6: Large-Scale MoE with Shared Experts (EP=8, TP=4, shared_experts=2)
    print("\n--- Test MOE-6: Shared Experts (EP=8, TP=4, shared=2) ---")
    try:
        layer = MoELayer(
            layer_idx=0,
            hidden_size=1024,
            intermediate_size=4096,
            num_experts=8,
            top_k=2,
            num_shared_experts=2,
            num_projections=2,
            dtype='bf16',
            parallelism={'expert_parallel': 8, 'tensor_parallel': 4}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 541_065_216,
            'weight_memory_per_chip': 12_599_296,
            'activation_memory_per_chip': 1_314_816,  # (k*M/ep)*d_ff_local + (M/tp)*d_ff_local for shared
            'kv_cache_per_chip': 0,
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test MOE-7: MoE with Context Parallel (EP=4, CP=2)
    print("\n--- Test MOE-7: Context Parallel (EP=4, CP=2) ---")
    try:
        layer = MoELayer(
            layer_idx=0,
            hidden_size=1024,
            intermediate_size=4096,
            num_experts=8,
            top_k=2,
            num_projections=2,
            dtype='bf16',
            parallelism={'expert_parallel': 4, 'context_parallel': 2}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 1_075_838_976,
            'weight_memory_per_chip': 33_570_816,
            'activation_memory_per_chip': 1_050_624,
            'kv_cache_per_chip': 0,
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test MOE-8: Maximum Parallelism (EP=8, TP=4, CP=2)
    print("\n--- Test MOE-8: Maximum Parallelism (EP=8, TP=4, CP=2) ---")
    try:
        layer = MoELayer(
            layer_idx=0,
            hidden_size=1024,
            intermediate_size=4096,
            num_experts=8,
            top_k=2,
            num_projections=2,
            dtype='bf16',
            parallelism={'expert_parallel': 8, 'tensor_parallel': 4, 'context_parallel': 2}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=128, phase='prefill')
        
        expected = {
            'flops_per_chip': 136_314_880,
            'weight_memory_per_chip': 4_210_688,
            'activation_memory_per_chip': 591_872,
            'kv_cache_per_chip': 0,
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    print(f"\n--- MoE Tests Summary: {tests_passed} passed, {tests_failed} failed ---")
    return tests_passed, tests_failed


def test_swa_layers():
    """Run Sliding Window Attention tests from gaps/attention_gaps_gpt_oss.md"""
    print("\n" + "=" * 80)
    print("SLIDING WINDOW ATTENTION LAYER TESTS")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test SWA-1: Prefill, seq_len < window, single chip
    print("\n--- Test SWA-1: Prefill, seq_len < window, single chip ---")
    try:
        layer = SlidingWindowAttentionLayer(
            layer_idx=0,
            hidden_size=2880,
            num_query_heads=32,
            num_kv_heads=4,
            head_dim=128,
            sliding_window=128,
            num_sinks=0,
            has_bias=True,
            dtype='bf16',
            parallelism={'tensor_parallel': 1}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=64, phase='prefill')
        
        expected = {
            'flops_per_chip': 6_930_014_208,
            'weight_memory_per_chip': 53_100_160,
            'activation_memory_per_chip': 2_785_280,
            'kv_cache_per_chip': 262_144,
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test SWA-2: Prefill, seq_len > window, single chip
    print("\n--- Test SWA-2: Prefill, seq_len > window, single chip ---")
    try:
        layer = SlidingWindowAttentionLayer(
            layer_idx=0,
            hidden_size=2880,
            num_query_heads=32,
            num_kv_heads=4,
            head_dim=128,
            sliding_window=128,
            num_sinks=0,
            has_bias=True,
            dtype='bf16',
            parallelism={'tensor_parallel': 1}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=512, phase='prefill')
        
        expected = {
            'flops_per_chip': 56_513_855_488,
            'weight_memory_per_chip': 53_100_160,
            'activation_memory_per_chip': 22_282_240,
            'kv_cache_per_chip': 524_288,  # Limited to window
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test SWA-3: Decode, single chip
    print("\n--- Test SWA-3: Decode, single chip ---")
    try:
        layer = SlidingWindowAttentionLayer(
            layer_idx=0,
            hidden_size=2880,
            num_query_heads=32,
            num_kv_heads=4,
            head_dim=128,
            sliding_window=128,
            num_sinks=0,
            has_bias=True,
            dtype='bf16',
            parallelism={'tensor_parallel': 1}
        )
        # seq_len=256 represents past_seq_len for decode
        metrics = layer.compute_metrics(batch_size=2, seq_len=256, phase='decode')
        
        expected = {
            'flops_per_chip': 110_378_624,
            'weight_memory_per_chip': 53_100_160,
            'activation_memory_per_chip': 43_520,
            'kv_cache_per_chip': 524_288,  # Limited to window
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test SWA-4: Prefill with TP=4
    print("\n--- Test SWA-4: Prefill with TP=4 ---")
    try:
        layer = SlidingWindowAttentionLayer(
            layer_idx=0,
            hidden_size=2880,
            num_query_heads=32,
            num_kv_heads=4,
            head_dim=128,
            sliding_window=128,
            num_sinks=0,
            has_bias=True,
            dtype='bf16',
            parallelism={'tensor_parallel': 4}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=256, phase='prefill')
        
        expected = {
            'flops_per_chip': 7_064_231_936,
            'weight_memory_per_chip': 13_279_360,
            'activation_memory_per_chip': 7_208_960,
            'kv_cache_per_chip': 131_072,
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # Test SWA-5: Sliding Window with Attention Sinks
    print("\n--- Test SWA-5: Sliding Window with Attention Sinks ---")
    try:
        layer = SlidingWindowAttentionLayer(
            layer_idx=0,
            hidden_size=2880,
            num_query_heads=32,
            num_kv_heads=4,
            head_dim=128,
            sliding_window=128,
            num_sinks=64,
            has_bias=True,
            dtype='bf16',
            parallelism={'tensor_parallel': 1}
        )
        metrics = layer.compute_metrics(batch_size=2, seq_len=512, phase='prefill')
        
        expected = {
            'flops_per_chip': 57_587_597_312,
            'weight_memory_per_chip': 53_100_160,
            'activation_memory_per_chip': 22_282_240,
            'kv_cache_per_chip': 786_432,  # sinks (64) + window (128) = 192 positions
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        print(f"  KV cache/chip: {metrics.kv_cache_per_package:,} (expected: {expected['kv_cache_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip'] and
            metrics.kv_cache_per_package == expected['kv_cache_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    print(f"\n--- SWA Tests Summary: {tests_passed} passed, {tests_failed} failed ---")
    return tests_passed, tests_failed


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("MANTILE LAYER TESTS")
    print("=" * 80)
    
    total_passed = 0
    total_failed = 0
    
    # Run MLP tests
    mlp_passed, mlp_failed = test_mlp_layers()
    total_passed += mlp_passed
    total_failed += mlp_failed
    
    # Run Attention tests
    attn_passed, attn_failed = test_attention_layers()
    total_passed += attn_passed
    total_failed += attn_failed
    
    # Run MoE tests
    moe_passed, moe_failed = test_moe_layers()
    total_passed += moe_passed
    total_failed += moe_failed
    
    # Run SWA tests
    swa_passed, swa_failed = test_swa_layers()
    total_passed += swa_passed
    total_failed += swa_failed
    
    # Final summary
    print("\n" + "=" * 80)
    print(f"OVERALL SUMMARY: {total_passed} passed, {total_failed} failed")
    print("=" * 80)
    
    # Exit with appropriate code
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
