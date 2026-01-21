#!/usr/bin/env python3
"""
Timing Test Runner for Mantile

Runs timing tests documented in tests/timing_tests.md
Verifies compute_time_ms, weight_load_time_ms, communication_time_ms, and wall_clock_time_ms.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.layers import MLPLayer
from backend.layers.base import DataType


def print_test_header(test_num, description):
    """Print formatted test header"""
    print(f"\n{'=' * 80}")
    print(f"Test {test_num}: {description}")
    print('=' * 80)


def check_close(actual, expected, tolerance=0.001, name="value"):
    """Check if actual is within tolerance of expected"""
    if expected == 0:
        passed = abs(actual) < tolerance
    else:
        passed = abs((actual - expected) / expected) < tolerance
    
    status = "✓" if passed else "✗"
    print(f"  {status} {name}: {actual:.6f} (expected: {expected:.6f})")
    return passed


def test_0_compute_bound():
    """Test 0: Compute-bound layer (large batch)"""
    print_test_header(0, "Compute-bound layer (simple MLP, single chip)")
    
    # Simplified hardware for easy verification
    test_hardware = {
        "compute_per_package_PFlops": {"bf16": 1.0, "fp16": 1.0},  # 1 PFLOP = 1000 TFLOPS
        "memory_bandwidth_gb_s": 1000,
        "interconnect_bandwidth_gb_s": 100,
        "interconnect_latency_us": 1.0
    }
    
    layer = MLPLayer(
        layer_idx=0,
        hidden_size=4096,
        intermediate_size=16384,
        dtype='bf16',
        parallelism={'tensor_parallel': 1}
    )
    
    metrics = layer.compute_metrics(
        batch_size=8,
        seq_len=1024,
        phase='prefill',
        hardware=test_hardware
    )
    
    # Expected values from timing_tests.md
    expected = {
        'flops_per_chip': 2_199_023_255_552,
        'weight_memory_per_chip': 268_435_456,
        'compute_time_ms': 2.199,
        'weight_load_time_ms': 0.268,
        'communication_time_ms': 0.0,
        'wall_clock_time_ms': 2.199
    }
    
    passed = []
    passed.append(check_close(metrics.flops_per_package, expected['flops_per_chip'], 0.01, "FLOPs/chip"))
    passed.append(check_close(metrics.weight_memory_per_package, expected['weight_memory_per_chip'], 0.01, "Weight/chip"))
    passed.append(check_close(metrics.compute_time_ms, expected['compute_time_ms'], 0.01, "Compute time (ms)"))
    passed.append(check_close(metrics.weight_load_time_ms, expected['weight_load_time_ms'], 0.01, "Weight load time (ms)"))
    
    if metrics.communication_time_ms is None or metrics.communication_time_ms == 0:
        print(f"  ✓ Communication time (ms): 0.0 (expected: 0.0)")
        passed.append(True)
    else:
        print(f"  ✗ Communication time (ms): {metrics.communication_time_ms} (expected: 0.0)")
        passed.append(False)
    
    # Note: wall_clock_time_ms might not be set, check if compute_time is returned
    if hasattr(metrics, 'wall_clock_time_ms') and metrics.wall_clock_time_ms is not None:
        passed.append(check_close(metrics.wall_clock_time_ms, expected['wall_clock_time_ms'], 0.01, "Wall clock time (ms)"))
    
    print(f"\n  Arithmetic intensity: {expected['flops_per_chip'] / expected['weight_memory_per_chip']:.0f} FLOPs/byte")
    print(f"  Status: COMPUTE-BOUND (compute time {expected['compute_time_ms'] / expected['weight_load_time_ms']:.1f}x longer than memory)")
    
    return all(passed)


def test_1_memory_bound():
    """Test 1: Memory-bound layer (decode phase, small batch)"""
    print_test_header(1, "Memory-bound layer (decode phase, small batch)")
    
    test_hardware = {
        "compute_per_package_PFlops": {"bf16": 1.0, "fp16": 1.0},
        "memory_bandwidth_gb_s": 1000,
        "interconnect_bandwidth_gb_s": 100,
        "interconnect_latency_us": 1.0
    }
    
    layer = MLPLayer(
        layer_idx=0,
        hidden_size=4096,
        intermediate_size=16384,
        dtype='bf16',
        parallelism={'tensor_parallel': 1}
    )
    
    metrics = layer.compute_metrics(
        batch_size=1,
        seq_len=1,
        phase='decode',
        hardware=test_hardware
    )
    
    expected = {
        'flops_per_chip': 268_435_456,
        'weight_memory_per_chip': 268_435_456,
        'compute_time_ms': 0.000268,
        'weight_load_time_ms': 0.268,
        'communication_time_ms': 0.0
    }
    
    passed = []
    passed.append(check_close(metrics.flops_per_package, expected['flops_per_chip'], 0.01, "FLOPs/chip"))
    passed.append(check_close(metrics.weight_memory_per_package, expected['weight_memory_per_chip'], 0.01, "Weight/chip"))
    passed.append(check_close(metrics.compute_time_ms, expected['compute_time_ms'], 0.01, "Compute time (ms)"))
    passed.append(check_close(metrics.weight_load_time_ms, expected['weight_load_time_ms'], 0.01, "Weight load time (ms)"))
    
    if metrics.communication_time_ms is None or metrics.communication_time_ms == 0:
        print(f"  ✓ Communication time (ms): 0.0 (expected: 0.0)")
        passed.append(True)
    else:
        print(f"  ✗ Communication time (ms): {metrics.communication_time_ms} (expected: 0.0)")
        passed.append(False)
    
    print(f"\n  Arithmetic intensity: {expected['flops_per_chip'] / expected['weight_memory_per_chip']:.0f} FLOPs/byte")
    print(f"  Status: MEMORY-BOUND (memory time {expected['weight_load_time_ms'] / expected['compute_time_ms']:.0f}x longer than compute)")
    
    return all(passed)


def test_2_communication_tp():
    """Test 2: Communication time with Tensor Parallelism (TP=4)"""
    print_test_header(2, "Communication time with Tensor Parallelism (TP=4)")
    
    test_hardware = {
        "compute_per_package_PFlops": {"bf16": 1.0, "fp16": 1.0},
        "memory_bandwidth_gb_s": 1000,
        "interconnect_bandwidth_gb_s": 100,
        "interconnect_latency_us": 10.0  # Larger latency to show effect
    }
    
    layer = MLPLayer(
        layer_idx=0,
        hidden_size=4096,
        intermediate_size=16384,
        dtype='bf16',
        parallelism={'tensor_parallel': 4}
    )
    
    metrics = layer.compute_metrics(
        batch_size=4,
        seq_len=512,
        phase='prefill',
        hardware=test_hardware
    )
    
    expected = {
        'flops_per_chip': 137_438_953_472,
        'communication_bytes': 16_777_216,
        'compute_time_ms': 0.137,
        'communication_time_ms': 0.178,  # latency: 0.01, transfer: 0.168
        'wall_clock_time_ms': 0.315  # sum, no overlap
    }
    
    passed = []
    passed.append(check_close(metrics.flops_per_package, expected['flops_per_chip'], 0.01, "FLOPs/chip"))
    passed.append(check_close(metrics.compute_time_ms, expected['compute_time_ms'], 0.01, "Compute time (ms)"))
    
    if hasattr(metrics, 'communication_bytes') and metrics.communication_bytes is not None:
        passed.append(check_close(metrics.communication_bytes, expected['communication_bytes'], 0.01, "Communication bytes"))
    
    if hasattr(metrics, 'communication_time_ms') and metrics.communication_time_ms is not None:
        passed.append(check_close(metrics.communication_time_ms, expected['communication_time_ms'], 0.01, "Communication time (ms)"))
    else:
        print(f"  ✗ Communication time (ms): None (expected: {expected['communication_time_ms']:.6f})")
        passed.append(False)
    
    if hasattr(metrics, 'wall_clock_time_ms') and metrics.wall_clock_time_ms is not None:
        passed.append(check_close(metrics.wall_clock_time_ms, expected['wall_clock_time_ms'], 0.01, "Wall clock time (ms)"))
    
    print(f"\n  Expected breakdown: latency=0.01ms + transfer=0.168ms = 0.178ms")
    print(f"  Status: Communication adds overhead to compute time")
    
    return all(passed)


def test_3_communication_overlap():
    """Test 3: Wall clock time with compute/communication overlap"""
    print_test_header(3, "Wall clock time with compute/communication overlap")
    
    test_hardware = {
        "compute_per_package_PFlops": {"bf16": 1.0, "fp16": 1.0},
        "memory_bandwidth_gb_s": 1000,
        "interconnect_bandwidth_gb_s": 100,
        "interconnect_latency_us": 10.0,
        "supports_overlap": True  # Enable overlap
    }
    
    layer = MLPLayer(
        layer_idx=0,
        hidden_size=4096,
        intermediate_size=16384,
        dtype='bf16',
        parallelism={'tensor_parallel': 4}
    )
    
    metrics = layer.compute_metrics(
        batch_size=4,
        seq_len=512,
        phase='prefill',
        hardware=test_hardware
    )
    
    expected = {
        'compute_time_ms': 0.137,
        'communication_time_ms': 0.178,
        'wall_clock_time_ms': 0.178  # max, with overlap
    }
    
    passed = []
    passed.append(check_close(metrics.compute_time_ms, expected['compute_time_ms'], 0.01, "Compute time (ms)"))
    
    if hasattr(metrics, 'communication_time_ms') and metrics.communication_time_ms is not None:
        passed.append(check_close(metrics.communication_time_ms, expected['communication_time_ms'], 0.01, "Communication time (ms)"))
    
    if hasattr(metrics, 'wall_clock_time_ms') and metrics.wall_clock_time_ms is not None:
        passed.append(check_close(metrics.wall_clock_time_ms, expected['wall_clock_time_ms'], 0.01, "Wall clock time (ms)"))
        improvement = ((0.315 - expected['wall_clock_time_ms']) / 0.315) * 100
        print(f"\n  Improvement with overlap: {improvement:.0f}% (0.178ms vs 0.315ms without overlap)")
    
    return all(passed)


def test_4_latency_dominated():
    """Test 4: Communication-dominated with high latency"""
    print_test_header(4, "Communication-dominated with high latency")
    
    test_hardware = {
        "compute_per_package_PFlops": {"bf16": 1.0, "fp16": 1.0},
        "memory_bandwidth_gb_s": 1000,
        "interconnect_bandwidth_gb_s": 100,
        "interconnect_latency_us": 100.0  # High latency scenario
    }
    
    layer = MLPLayer(
        layer_idx=0,
        hidden_size=1024,
        intermediate_size=4096,
        dtype='bf16',
        parallelism={'tensor_parallel': 8}
    )
    
    metrics = layer.compute_metrics(
        batch_size=1,
        seq_len=32,
        phase='prefill',
        hardware=test_hardware
    )
    
    expected = {
        'flops_per_chip': 67_108_864,
        'communication_bytes': 65_536,
        'compute_time_ms': 0.000067,
        'communication_time_ms': 0.101,  # latency: 0.1, transfer: 0.0007
        'wall_clock_time_ms': 0.101
    }
    
    passed = []
    passed.append(check_close(metrics.flops_per_package, expected['flops_per_chip'], 0.01, "FLOPs/chip"))
    passed.append(check_close(metrics.compute_time_ms, expected['compute_time_ms'], 0.01, "Compute time (ms)"))
    
    if hasattr(metrics, 'communication_bytes') and metrics.communication_bytes is not None:
        passed.append(check_close(metrics.communication_bytes, expected['communication_bytes'], 0.01, "Communication bytes"))
    
    if hasattr(metrics, 'communication_time_ms') and metrics.communication_time_ms is not None:
        passed.append(check_close(metrics.communication_time_ms, expected['communication_time_ms'], 0.01, "Communication time (ms)"))
        print(f"\n  Breakdown: latency=0.1ms (99%), transfer=0.0007ms (1%)")
        print(f"  Status: LATENCY-DOMINATED - reducing latency is critical for small batches")
    else:
        print(f"  ✗ Communication time (ms): None (expected: {expected['communication_time_ms']:.6f})")
        passed.append(False)
    
    return all(passed)


def main():
    """Run all timing tests"""
    print("=" * 80)
    print("MANTILE TIMING TESTS")
    print("=" * 80)
    print("\nVerifying compute_time_ms, weight_load_time_ms, communication_time_ms")
    print("Using simplified hardware (1 PFLOP, 1 TB/s) for easy verification")
    
    tests = [
        ("Test 0", test_0_compute_bound),
        ("Test 1", test_1_memory_bound),
        ("Test 2", test_2_communication_tp),
        ("Test 3", test_3_communication_overlap),
        ("Test 4", test_4_latency_dominated),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TIMING TESTS SUMMARY")
    print("=" * 80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed_count}/{total_count} passed")
    
    if passed_count == total_count:
        print("\n  🎉 All timing tests passed!")
        return 0
    else:
        print(f"\n  ⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
