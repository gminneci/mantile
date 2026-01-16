#!/usr/bin/env python3
"""
Test Runner for Mantile Layer Tests

Runs all test cases documented in tests/*.md files to verify layer implementations.
"""

import sys
from pathlib import Path

# Add project root to path (parent of tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.layers import MLPLayer, AttentionLayer, GroupedQueryAttentionLayer
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
    
    # Final summary
    print("\n" + "=" * 80)
    print(f"OVERALL SUMMARY: {total_passed} passed, {total_failed} failed")
    print("=" * 80)
    
    # Exit with appropriate code
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
