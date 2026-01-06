#!/usr/bin/env python3
"""Test the stateless system-metrics endpoint"""

import requests
import json

API_URL = "http://localhost:8000"

# Test payload
payload = {
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "hardware_config": "nvidia_gb200_single",
    "batch_size": 1,
    "input_seq": 2048,
    "output_seq": 128,
    "layers": {
        "attention": {
            "tensor_parallel": 1,
            "context_parallel": 1,
            "sequence_parallel": 1,
            "dtype": "bf16"
        },
        "feedforward": {
            "tensor_parallel": 1,
            "context_parallel": 1,
            "sequence_parallel": 1,
            "dtype": "bf16"
        },
        "norm": {
            "tensor_parallel": 1,
            "context_parallel": 1,
            "sequence_parallel": 1,
            "dtype": "bf16"
        }
    }
}

print("Testing /config/system-metrics endpoint...")
print(f"Payload: {json.dumps(payload, indent=2)}")
print()

try:
    response = requests.post(f"{API_URL}/config/system-metrics", json=payload)
    
    if response.status_code == 200:
        print("✅ SUCCESS!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ FAILED with status {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ ERROR: {e}")
