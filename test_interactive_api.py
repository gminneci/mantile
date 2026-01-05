#!/usr/bin/env python3
"""
Test the new interactive configuration API endpoints.
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_list_hardware():
    """Test listing hardware configurations."""
    print("=" * 60)
    print("Test: List Hardware")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/hardware")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Available configs: {data['configs']}")
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        return False


def test_load_model_and_hardware():
    """Test loading model and hardware."""
    print("\n" + "=" * 60)
    print("Test: Load Model + Hardware")
    print("=" * 60)
    
    payload = {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "hardware_config": "nvidia_gb200_single"
    }
    
    response = requests.post(f"{BASE_URL}/config/load", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Model loaded: {data['model']['id']}")
        print(f"   Layers: {data['model']['num_layers']}")
        print(f"   Hidden: {data['model']['hidden_size']}")
        print(f"   Params: {data['validation']['total_params']:,}")
        print(f"   Attention: {data['validation']['attention_type']}")
        print(f"   MLP: {data['validation']['mlp_type']}")
        print(f"\n✅ Hardware: {data['hardware']['name']}")
        print(f"   Capacity: {data['hardware']['hbm_capacity_gb']:.0f} GB")
        print(f"   BF16: {data['hardware']['bf16_tflops']:,.0f} TFLOPs")
        return True, data
    else:
        print(f"❌ Failed: {response.status_code}")
        print(response.text)
        return False, None


def test_get_layer_types():
    """Test getting layer types."""
    print("\n" + "=" * 60)
    print("Test: Get Layer Types")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/config/layer-types")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Layer types: {data['layer_types']}")
        return True, data['layer_types']
    else:
        print(f"❌ Failed: {response.status_code}")
        return False, []


def test_configure_layer_parallelism(layer_type: str):
    """Test configuring parallelism for a layer."""
    print(f"\n   Configuring {layer_type}...")
    
    payload = {
        "layer_type": layer_type,
        "tensor_parallel": 4,
        "context_parallel": 1,
        "sequence_parallel": 1
    }
    
    response = requests.post(f"{BASE_URL}/config/layer-parallelism", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ {layer_type}: TP={data['config']['parallelism']['tensor_parallel']}, "
              f"{data['config']['num_instances']} instances")
        return True
    else:
        print(f"   ❌ Failed: {response.status_code}")
        return False


def test_system_requirements():
    """Test calculating system requirements."""
    print("\n" + "=" * 60)
    print("Test: Calculate System Requirements")
    print("=" * 60)
    
    payload = {
        "batch_size": 1,
        "seq_length": 2048
    }
    
    response = requests.post(f"{BASE_URL}/config/system-requirements", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Minimum chips: {data['min_chips']}")
        print(f"   Weight memory: {data['total_weight_memory_gb']:.2f} GB")
        print(f"   Memory per chip: {data['memory_per_chip_gb']:.2f} GB")
        print(f"   HW capacity: {data['hw_capacity_gb']:.0f} GB")
        print(f"   Fits: {'✅ Yes' if data['fits_on_hardware'] else '❌ No'}")
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        print(response.text)
        return False


def test_full_flow():
    """Test the complete 5-step flow."""
    print("\n" + "=" * 60)
    print("FULL INTERACTIVE FLOW TEST")
    print("=" * 60)
    
    results = []
    
    # Step 1: Load model + hardware
    success, load_data = test_load_model_and_hardware()
    results.append(success)
    if not success:
        return False
    
    # Step 2: Get layer types
    success, layer_types = test_get_layer_types()
    results.append(success)
    if not success:
        return False
    
    # Step 3: Configure parallelism for each layer type
    print("\n" + "=" * 60)
    print("Test: Configure Layer Parallelism")
    print("=" * 60)
    for layer_type in layer_types:
        success = test_configure_layer_parallelism(layer_type)
        results.append(success)
    
    # Step 4: Calculate system requirements
    success = test_system_requirements()
    results.append(success)
    
    return all(results)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("INTERACTIVE CONFIGURATION API TEST")
    print("=" * 60)
    print("\nMake sure backend is running: python -m uvicorn backend.main:app --reload\n")
    
    try:
        # Quick health check
        response = requests.get(f"{BASE_URL}/hardware")
        if response.status_code != 200:
            print("❌ Backend not responding. Start it with:")
            print("   python -m uvicorn backend.main:app --reload")
            exit(1)
        
        # Run full test
        success = test_full_flow()
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 ALL TESTS PASSED - Interactive API working!")
        else:
            print("❌ Some tests failed - see details above")
        print("=" * 60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Start it with:")
        print("   python -m uvicorn backend.main:app --reload")
        exit(1)
