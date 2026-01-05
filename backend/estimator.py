from .models import HardwareSpecs, ModelIR, ParallelismConfig, EstimateResult

def estimate_performance(
    hardware: HardwareSpecs,
    model: ModelIR,
    parallel: ParallelismConfig
) -> EstimateResult:
    """
    Estimates performance for a given setup.
    Assumption: Float16/BF16 by default.
    Parallelism: Simple Tensor Parallelism (TP) logic.
    """
    
    # 1. Calculate Total Model Params
    total_params = sum(l.parameter_count for l in model.layers)
    # Add embeddings (vocab * hidden)
    total_params += (model.vocab_size * model.hidden_size)
    
    # Bytes per param (assume BF16 = 2 bytes)
    BYTES_PER_PARAM = 2
    weights_mem_gb = (total_params * BYTES_PER_PARAM) / 1e9
    
    # 2. Calculate KV Cache Size
    # KV Cache per token = 2 * n_layers * n_kv_heads * head_dim * precision
    # Total KV = KV_per_token * (batch_size * (input_seq + output_seq))
    
    # We take the first attn layer to get head config (assuming uniform)
    attn_layers = [l for l in model.layers if l.module_type == "attention"]
    if attn_layers:
        l0 = attn_layers[0]
        n_layers = model.num_layers
        n_kv_heads = l0.kv_heads or l0.num_heads
        head_dim = l0.head_dim or (model.hidden_size // l0.num_heads)
        
        kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * BYTES_PER_PARAM
        total_tokens = parallel.batch_size * (parallel.input_seq_len + parallel.output_seq_len)
        kv_cache_mem_gb = (kv_bytes_per_token * total_tokens) / 1e9
    else:
        kv_cache_mem_gb = 0.0

    # 3. Activation Memory (Rough estimate)
    # Activation per layer ~= Batch * Seq * Hidden * (Factor depending on architecture)
    # Re-materialization matters. Let's assume a static buffer.
    activation_mem_gb = 5.0 # Placeholder: 5GB constant overhead for now or use formula?
    # Simple formula: B * S * H * 2 bytes * num_layers (if storing all?) No, usually checkpointed.
    # Let's leave as fixed overhead + (B*S*H*2)/1e9
    activation_mem_gb = (parallel.batch_size * parallel.input_seq_len * model.hidden_size * 2 * model.num_layers) / 1e9 * 0.1 # Assume 10% stored
    
    total_mem_gb = weights_mem_gb + kv_cache_mem_gb + activation_mem_gb
    max_mem = hardware.hbm_capacity_gb # This is per CHIP or per SYSTEM?
    # Arguments passed: hardware is likely the SINGLE CHIP specs or AGGREGATE?
    # If using TP, we pool HBM.
    # Let's assume 'hardware' object passed here represents the TOTAL AVAILABLE RESOURCE (e.g. Rack or N chips).
    # If parallelism.tp_size > 1, total_mem is distributed.
    
    # Wait, the caller should pass the AGGREGATE hardware spec? 
    # Or we handle valid checks here?
    # Let's assume hardware spec matches the parallelism scale (e.g. if TP=72, hardware=NVL72 full).
    
    # 4. Compute Metrics (Prefill)
    # Prefill Ops ~= 2 * Total Params * Batch * Input Seq (Approx for Dense)
    prefill_flops = 2 * total_params * parallel.batch_size * parallel.input_seq_len
    
    # Compute Time = FLOPS / Peak FLOPS
    # Peak TFLOPS -> converted to FLOPS (1e12)
    peak_flops_per_sec = hardware.bf16_tflops * 1e12
    
    compute_time_prefill = prefill_flops / peak_flops_per_sec
    
    # Memory Time Prefill = Moved Bytes / Bandwidth
    # Usually Compute bound. Move weights once? Params are resident.
    # So just IO for Activations + KV write. Negligible compared to compute usually for large batch.
    
    time_to_first_token_ms = compute_time_prefill * 1000
    
    # 5. Compute Metrics (Decode)
    # Decode is per token.
    # Ops = 2 * Total Params * Batch
    decode_flops_per_token = 2 * total_params * parallel.batch_size
    
    # Memory Moved = Total Params * 2 bytes (Weights) + KV read (growing)
    # At max context:
    decode_bytes_per_token = (total_params * BYTES_PER_PARAM) + (kv_cache_mem_gb * 1e9 / total_tokens * parallel.batch_size)
    # Note: KV read is batch * current_kv_size.
    
    # Compute Time
    t_comp = decode_flops_per_token / peak_flops_per_sec
    
    # Memory Time
    hbm_bw_bytes_sec = hardware.hbm_bandwidth_gbps * 1e9
    t_mem = decode_bytes_per_token / hbm_bw_bytes_sec
    
    # Comm Time (TP)
    # Layer norm + AllReduce per layer.
    # Time = 2 * (Latency + Bytes/BW) * Layers * (AllReduces per layer)
    # Llama: AllReduce in Attn (1), MLP (1). Total 2 per layer.
    # Bytes moved is small (Hidden * Batch * 2 bytes).
    # Comm Latency dominates for small batch.
    comm_time = 0
    if parallel.tp_size > 1:
        interconnect_lat = hardware.interconnect_latency_us * 1e-6
        # 2 AllReduces per layer * Num Layers
        # Each AllReduce = 2 * (Latency + Size/BW) (Ring) or similar.
        num_comms = model.num_layers * 2
        comm_time = num_comms * (2 * interconnect_lat) # simplified latency only
        
    t_step = max(t_comp, t_mem) + comm_time
    
    time_per_output_token_ms = t_step * 1000
    total_throughput = (parallel.batch_size) / t_step
    
    # 6. Bottleneck
    is_mem_bound = t_mem > t_comp
    
    return EstimateResult(
        total_latency_ms=time_to_first_token_ms + (time_per_output_token_ms * parallel.output_seq_len),
        time_to_first_token_ms=time_to_first_token_ms,
        time_per_output_token_ms=time_per_output_token_ms,
        total_throughput_tokens_s=total_throughput,
        
        weights_mem_gb=weights_mem_gb,
        kv_cache_mem_gb=kv_cache_mem_gb,
        activation_mem_gb=activation_mem_gb,
        total_mem_gb=total_mem_gb,
        max_mem_capacity_gb=hardware.hbm_capacity_gb,
        
        compute_bound_percent=100.0 if not is_mem_bound else (t_comp/t_mem)*100,
        memory_bound_percent=100.0 if is_mem_bound else (t_mem/t_comp)*100,
        comm_bound_percent=(comm_time / t_step) * 100
    )
