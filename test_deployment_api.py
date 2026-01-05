#!/usr/bin/env python3
"""
Quick test of the /deployment/estimate API endpoint.
This demonstrates the single-call deployment query API.
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_deployment_estimate():
    """Test the deployment estimate endpoint."""
    print("=" * 70)
    print("DEPLOYMENT API ENDPOINT TEST")
    print("=" * 70)
    
    # Complete deployment configuration
    config = {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "hardware_config": "nvidia_gb200_single",
        "batch_size": 1,
        "input_seq": 2048,
        "output_seq": 128,
        "layer_parallelism": {
            "attention": {"tensor_parallel": 2},
            "feedforward": {"tensor_parallel": 4}
        }
    }
    
    print(f"\n📤 Request:")
    print(json.dumps(config, indent=2))
    
    print(f"\n🔄 Calling POST {BASE_URL}/deployment/estimate...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/deployment/estimate",
            json=config,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n✅ Response (200 OK):")
            print(f"\n📊 Performance Metrics:")
            perf = data["performance"]
            print(f"   TTFT: {perf['ttft_ms']:.2f} ms")
            print(f"   TPOT: {perf['tpot_ms']:.4f} ms")
            print(f"   Throughput: {perf['throughput_tokens_s']:.0f} tokens/s")
            
            print(f"\n💾 Memory:")
            mem = perf["memory"]
            print(f"   Total: {mem['total_memory_gb']:.2f} GB")
            print(f"   Per chip: {mem['memory_per_chip_gb']:.2f} GB / {mem['hw_capacity_gb']:.0f} GB")
            
            print(f"\n🎯 System:")
            sys = perf["system"]
            print(f"   Chips: {sys['num_chips']}")
            print(f"   Bottleneck: {sys['bottleneck']}")
            print(f"   Fits: {'✅ Yes' if sys['fits_on_hardware'] else '❌ No'}")
            
            print(f"\n✅ Validation:")
            val = data["validation"]
            print(f"   Parameters: {val['total_params']:,}")
            print(f"   Layers: {val['num_layers']}")
            print(f"   Attention: {val['attention_type']}")
            
            print(f"\n💡 Requirements:")
            req = data["requirements"]
            print(f"   Min chips: {req['min_chips']}")
            print(f"   Memory per chip: {req['memory_per_chip_gb']:.2f} GB")
            
            return True
        
        else:
            print(f"\n❌ Error ({response.status_code}):")
            print(response.text)
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection failed!")
        print(f"   Make sure backend is running:")
        print(f"   python -m uvicorn backend.main:app --reload")
        return False
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MANTILE DEPLOYMENT API TEST")
    print("=" * 70)
    print("\nThis tests the /deployment/estimate endpoint")
    print("Make sure backend is running on port 8000\n")
    
    success = test_deployment_estimate()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 API TEST PASSED!")
        print("\nYou can now:")
        print("  • Query deployments via HTTP API")
        print("  • Integrate with React frontend")
        print("  • Use in automated scripts")
    else:
        print("❌ API TEST FAILED")
        print("\nStart backend with:")
        print("  python -m uvicorn backend.main:app --reload")
    print("=" * 70 + "\n")
