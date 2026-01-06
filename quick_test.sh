#!/bin/bash

echo "=== Quick Backend Test ==="
echo ""

# Check if backend is running
if ! curl -s http://localhost:8000/hardware > /dev/null 2>&1; then
    echo "❌ Backend is NOT running on port 8000"
    echo "   Start it with: ./run_backend.sh"
    exit 1
fi

echo "✅ Backend is running"
echo ""

# Test layer metrics endpoint
echo "Testing /config/layer-metrics..."
response=$(curl -s -X POST http://localhost:8000/config/layer-metrics \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "hardware_config": "nvidia_gb200_single",
    "layer_type": "attention",
    "batch_size": 1,
    "seq_length": 2048,
    "dtype": "bf16",
    "tensor_parallel": 1,
    "context_parallel": 1,
    "sequence_parallel": 1
  }')

if echo "$response" | grep -q "layer_type"; then
    echo "✅ Layer metrics endpoint works!"
    echo "   Sample response:"
    echo "$response" | python3 -m json.tool | head -10
else
    echo "❌ Layer metrics endpoint failed!"
    echo "   Response: $response"
fi

echo ""
echo "=== Test Complete ==="
