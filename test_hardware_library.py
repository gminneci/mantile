#!/usr/bin/env python3
"""
Test the refactored hardware library with JSON configs.
"""

from backend.hardware_library import (
    load_hardware_config,
    list_available_configs,
    get_all_configs
)


def test_list_configs():
    """Test listing available configs"""
    print("=" * 60)
    print("Available Hardware Configurations")
    print("=" * 60)
    
    configs = list_available_configs()
    for config_name in configs:
        print(f"  - {config_name}")
    
    print(f"\nTotal: {len(configs)} configs\n")
    return len(configs) > 0


def test_load_specific_configs():
    """Test loading specific configs"""
    print("=" * 60)
    print("Loading Specific Configurations")
    print("=" * 60)
    
    results = []
    
    # Test GB200 single
    try:
        gb200 = load_hardware_config("nvidia_gb200_single")
        print(f"\n✅ {gb200.name}")
        print(f"   Description: {gb200.description}")
        print(f"   BF16: {gb200.bf16_tflops:,.0f} TFLOPs")
        print(f"   HBM: {gb200.hbm_capacity_gb:.0f} GB @ {gb200.hbm_bandwidth_gbps:,.0f} GB/s")
        print(f"   Interconnect: {gb200.interconnect_bandwidth_gbps:.0f} GB/s")
        results.append(True)
    except Exception as e:
        print(f"❌ Failed to load nvidia_gb200_single: {e}")
        results.append(False)
    
    # Test NVL-72 rack
    try:
        nvl72 = load_hardware_config("nvidia_nvl72_rack")
        print(f"\n✅ {nvl72.name}")
        print(f"   Description: {nvl72.description}")
        print(f"   BF16: {nvl72.bf16_tflops:,.0f} TFLOPs ({nvl72.bf16_tflops/1000:.1f} PFLOPs)")
        print(f"   HBM: {nvl72.hbm_capacity_gb:,.0f} GB ({nvl72.hbm_capacity_gb/1024:.1f} TB)")
        print(f"   Total chips: {nvl72.chips_per_node}")
        results.append(True)
    except Exception as e:
        print(f"❌ Failed to load nvidia_nvl72_rack: {e}")
        results.append(False)
    
    # Test H100
    try:
        h100 = load_hardware_config("nvidia_h100_80gb")
        print(f"\n✅ {h100.name}")
        print(f"   Description: {h100.description}")
        print(f"   BF16: {h100.bf16_tflops:,.0f} TFLOPs")
        print(f"   HBM: {h100.hbm_capacity_gb:.0f} GB @ {h100.hbm_bandwidth_gbps:,.0f} GB/s")
        results.append(True)
    except Exception as e:
        print(f"❌ Failed to load nvidia_h100_80gb: {e}")
        results.append(False)
    
    return all(results)


def test_get_all_configs():
    """Test loading all configs at once"""
    print("\n" + "=" * 60)
    print("Loading All Configurations")
    print("=" * 60)
    
    try:
        all_configs = get_all_configs()
        print(f"\nLoaded {len(all_configs)} configurations:")
        for name, specs in all_configs.items():
            print(f"  - {name}: {specs.name}")
        return len(all_configs) > 0
    except Exception as e:
        print(f"❌ Failed to load all configs: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility functions"""
    print("\n" + "=" * 60)
    print("Testing Backward Compatibility")
    print("=" * 60)
    
    results = []
    
    try:
        from backend.hardware_library import get_nvl72_specs
        single = get_nvl72_specs()
        print(f"\n✅ get_nvl72_specs() works: {single.name}")
        results.append(True)
    except Exception as e:
        print(f"❌ get_nvl72_specs() failed: {e}")
        results.append(False)
    
    try:
        from backend.hardware_library import get_nvl72_rack_specs
        rack = get_nvl72_rack_specs()
        print(f"✅ get_nvl72_rack_specs() works: {rack.name}")
        results.append(True)
    except Exception as e:
        print(f"❌ get_nvl72_rack_specs() failed: {e}")
        results.append(False)
    
    return all(results)


def test_error_handling():
    """Test error handling for invalid configs"""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    try:
        load_hardware_config("nonexistent_config")
        print("❌ Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"✅ Correctly raised FileNotFoundError: {str(e)[:80]}...")
        return True
    except Exception as e:
        print(f"❌ Wrong exception type: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HARDWARE LIBRARY TEST (JSON-based)")
    print("=" * 60 + "\n")
    
    results = {
        "List configs": test_list_configs(),
        "Load specific configs": test_load_specific_configs(),
        "Load all configs": test_get_all_configs(),
        "Backward compatibility": test_backward_compatibility(),
        "Error handling": test_error_handling(),
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
        print("🎉 ALL TESTS PASSED - Hardware library refactored to JSON!")
    else:
        print("❌ Some tests failed - see details above")
    print("=" * 60 + "\n")
