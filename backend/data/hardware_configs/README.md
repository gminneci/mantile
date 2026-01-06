# Hardware Configuration Files

This directory contains JSON hardware specifications for various AI accelerators.

## Format

Each JSON file defines a hardware configuration with the following fields:

```json
{
  "name": "Hardware Name",
  "description": "Brief description of the hardware",
  "fp16_tflops": 0.0,     // FP16 compute in TFLOPs
  "bf16_tflops": 0.0,     // BF16 compute in TFLOPs
  "fp8_tflops": 0.0,      // FP8 compute in TFLOPs
  "int8_tops": 0.0,       // INT8 compute in TOPs
  "hbm_capacity_gb": 0.0, // HBM memory capacity in GB
  "hbm_bandwidth_gbps": 0.0, // HBM bandwidth in GB/s
  "interconnect_bandwidth_gbps": 0.0, // NVLink/interconnect BW in GB/s
  "interconnect_latency_us": 0.0,     // Interconnect latency in microseconds
  "chips_per_node": 1,    // Number of chips per node
  "nodes_per_cluster": 1  // Number of nodes in cluster
}
```

## Available Configurations

- **nvidia_gb200_single.json**: Single GB200 package (2 Blackwell GPUs)
- **nvidia_nvl72_rack.json**: Full NVL-72 rack (72 GB200 packages, 144 GPUs)
- **nvidia_h100_80gb.json**: Single H100 SXM5 80GB GPU

## Usage

Load hardware specs using the `hardware_library.py` module:

```python
from backend.hardware_library import load_hardware_config

# Load specific config
specs = load_hardware_config("nvidia_gb200_single")

# List all available configs
configs = list_available_configs()
```

## Adding New Configurations

1. Create a new JSON file in this directory
2. Follow the format above
3. Use a descriptive filename (e.g., `vendor_model_variant.json`)
4. Test loading with `load_hardware_config()`
