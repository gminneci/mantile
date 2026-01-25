# Hardware Configuration Files

This directory contains JSON hardware specifications for various AI accelerators.

## Format

Each JSON file defines a hardware configuration with the following fields:

```json
{
  "name": "Hardware Name",
  "description": "Brief description of the hardware",
  "manufacturer": "Manufacturer Name",
  "compute_per_package_PFlops": {
    "fp16": 0.0,    // FP16 compute in PFLOPs per package (optional)
    "bf16": 0.0,    // BF16 compute in PFLOPs per package (optional)
    "fp8": 0.0,     // FP8 compute in PFLOPs per package (optional)
    "nvfp8": 0.0,   // NVIDIA FP8 compute in PFLOPs per package (optional)
    "nvfp4": 0.0,   // NVIDIA FP4 compute in PFLOPs per package (optional)
    "int8": 0.0     // INT8 compute in POPs per package (optional)
  },
  "memory_per_package": [
    {"type": "HBM", "capacity_gb": 0.0, "bandwidth_gbps": 0.0},
    {"type": "SRAM", "capacity_gb": 0.0, "bandwidth_gbps": 0.0}  // Optional on-chip memory
  ],
  "interconnect_bandwidth_gbps": 0.0,    // NVLink/interconnect BW in Gbps
  "interconnect_latency_us": 0.0,        // Interconnect latency in microseconds
  "decode_load_overlap": false,          // Can decode overlap compute with memory loads?
  "decode_comms_overlap": false,         // Can decode overlap comms with compute/memory?
  "fixed_overhead_per_kernel_us": 10.0,  // Fixed kernel launch overhead in microseconds
  "power_kw": 0.0,                       // Power consumption in kilowatts (optional)
  "tco_sec_usd": 0.0,                    // Total cost of ownership in USD per second (optional)
  "packages_per_domain": 1,              // Number of packages (GPUs/accelerators) per domain
  "domains_per_cluster": 1               // Number of domains in cluster
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
bf16_pflops = hardware['compute_per_package_PFlops']['bf16']
hbm_memory = next((m for m in hardware['memory_per_package'] if 'HBM' in m['type']), hardware['memory_per_package'][0])
capacity = hbm_memory['capacity_gb']
bandwidth = hbm_memory['bandwidth_gbps']
```

## Overlap Configuration

The `decode_load_overlap` and `decode_comms_overlap` flags control whether operations can run in parallel during decode:

- **`decode_load_overlap`**: Whether compute can overlap with memory loads during decode
  - `true`: Use `max(compute_time, load_time)` (compute-bound, rare in decode)
  - `false`: Use `compute_time + load_time` (memory-bound, typical for decode with large KV cache)
  - Default: `false` for most GPUs (decode is typically memory-bound due to KV cache streaming)

- **`decode_comms_overlap`**: Whether communication can overlap with compute/memory operations
  - `true`: Use `max(compute+load, comm_time)` (e.g., TPU with independent comm engine)
  - `false`: Use `(compute+load) + comm_time` (sequential, typical for GPUs)
  - Default: `false` for most GPUs

**Note**: Prefill always assumes full overlap (`max(compute, load)`) as it's typically compute-bound with good data reuse.

## Adding New Configurations

1. Create a new JSON file in this directory
2. Follow the format above (see `template.json` for reference)
3. Use a descriptive filename: `{vendor}_{model}_{variant}.json`
4. Restart the backend to load the new config
5. Test via `GET /hardware/{config_name}` endpoint
