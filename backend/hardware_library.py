from .models import HardwareSpecs

def get_nvl72_specs() -> HardwareSpecs:
    """
    Returns the hardware specifications for a SINGLE NVL72 package (GB200) 
    or the full rack depending on how we want to model it.
    
    The prompt says:
    Per package:
    bf16: 2.5 PFlops = 2500 TFlops
    fp8: 5.0 PFlops = 5000 TFlops
    int8: 5.0 PFlops = 5000 Tops
    HBM size: 196,608 MB = 196.608 GB  (Wait, 192GB is standard, user said 196608 which is 192GB * 1024 or just 196GB? usually 192GB. 196608MB / 1024 = 192GB exactly.)
    HBM BW: 8,192 GB/s
    
    Rack:
    72 Packages.
    
    Notes:
    The user data: "Per package ... bf16 2.50". 
    Assuming these are PFLOPS (10^15). Models use TFLOPS (10^12) often.
    2.5 PFLOPS = 2500 TFLOPS.
    
    """
    
    # User provided "196,608" for HBM size. Assuming MB? 
    # 196,608 MB = 192 GB. 
    # User provided "8,192" for HBM BW. GB/s.
    
    # "Intra Chassis BW 900" -> likely GB/s NVLink per chip? 
    # Or total? Usually NVLink is ~900GB/s bidirectional per chip for H100, GB200 is comparable or higher (1.8TB/s). 
    # User says "Intra Chassis BW 900". Let's use 900 GB/s per chip for now.
    
    return HardwareSpecs(
        name="NVIDIA GB200 (NVL72 Component)",
        fp16_tflops=2500.0, # 2.5 PFlops
        bf16_tflops=2500.0,
        fp8_tflops=5000.0,
        int8_tops=5000.0,
        
        hbm_capacity_gb=192.0, # 196,608 MB
        hbm_bandwidth_gbps=8192.0,
        
        interconnect_bandwidth_gbps=900.0, # Assuming 900 GB/s per chip
        interconnect_latency_us=0.20, # User provided 0.20
        
        chips_per_node=2, # GB200 is usually 2 Blackwell + 1 Grace? Or just treat package as unit.
        nodes_per_cluster=36 # 72 chips total in simple view?
    )

def get_nvl72_rack_specs() -> HardwareSpecs:
    """
    Returns the specs for the FULL NVL-72 Rack acting as a giant GPU.
    """
    single = get_nvl72_specs()
    n_chips = 72
    
    return HardwareSpecs(
        name="NVIDIA NVL-72 (Full Rack)",
        fp16_tflops=single.fp16_tflops * n_chips,
        bf16_tflops=single.bf16_tflops * n_chips,
        fp8_tflops=single.fp8_tflops * n_chips,
        int8_tops=single.int8_tops * n_chips,
        
        hbm_capacity_gb=single.hbm_capacity_gb * n_chips,
        hbm_bandwidth_gbps=single.hbm_bandwidth_gbps * n_chips,
        
        # Interconnect is tricky for a "full rack" view. 
        # But if we treat it as one giant chip, the "interconnect" is the rack-to-rack or simplified.
        # For now, let's keep the single chip metrics for bandwidth if we are modeling distributed execution,
        # OR scale if we are modeling "perfect parallelism".
        # Let's assume this object represents the AGGREGATE resources.
        interconnect_bandwidth_gbps=single.interconnect_bandwidth_gbps * n_chips, 
        
        chips_per_node=72,
        nodes_per_cluster=1
    )
