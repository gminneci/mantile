# Hardware Configuration Files

This directory contains JSON hardware specifications for various AI accelerators.

## Format

Each JSON file defines a hardware configuration with the following fields:

```json
{
  "name": "Hardware Name",
  "description": "Brief description of the hardware",
  "manufacturer": "Manufacturer Name",
  "compute": {
    "fp16": 0.0,  // FP16 compute in TFLOPs
    "bf16": 0.0,  // BF16 compute in TFLOPs
    "fp8": 0.0,   // FP8 compute in TFLOPs
    "int8": 0.0   // INT8 compute in TOPs
  },
  "memory": [
    {"type": "HBM", "capacity_gb": 0.0, "bandwidth_gbps": 0.0}
  ],
  "interconnect_bandwidth_gbps": 0.0, // NVLink/interconnect BW in GB/s
  "interconnect_latency_us": 0.0,     // Interconnect latency in microseconds
  "chips_per_node": 1,    // Number of chips per node
  "nodes_per_cluster": 1  // Number of nodes in cluster
}
```

## Usage

Hardware configs are loaded via the FastAPI endpoint:

```python
# GET /hardware/{config_name}
# Returns the JSON config as a dict

# Example: Load nvidia_gb200_single
hardware = load_hardware_config("nvidia_gb200_single")

# Access nested structure:
bf16_tflops = hardware['compute']['bf16']
hbm_memory = next((m for m in hardware['memory'] if 'HBM' in m['type']), hardware['memory'][0])
capacity = hbm_memory['capacity_gb']
bandwidth = hbm_memory['bandwidth_gbps']
```

## Adding New Configurations

1. Create a new JSON file in this directory
2. Follow the format above (see `template.json` for reference)
3. Use a descriptive filename: `{vendor}_{model}_{variant}.json`
4. Restart the backend to load the new config
5. Test via `GET /hardware/{config_name}` endpoint
