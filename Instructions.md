# Competitor perf estimator

### App overview

**Working title:**

Mantile

**Purpose:**

Estimate *achievable* performance (latency, throughput, memory, cost, efficiency) of LLM inference workloads on modern AI accelerators **given a concrete model, system configuration, and parallelisation strategy**, with a **human-in-the-loop** to resolve ambiguities and override assumptions.

**Target users:**

- AI systems researchers
- Infra / performance engineers
- Hardware strategy & product teams
- Advanced analysts (not casual users)

**Core philosophy:**

> Deterministic where possible, explicit assumptions where not, and never “magic black box” outputs.
> 

### Backend structure

1. **Model → Intermediate Representation (IR) Builder**
    
    Parse HuggingFace (HF) config + code to produce a static IR: full tensor list, shapes, parameter groups, and execution order (attention, FFN, MoE, norms, embeddings, etc.), with explicit flags for non-standard variants (GQA/MQA/MLA, MoE routing, hybrids).
    
2. **IR Annotation & Functional Decomposition**
    
    Transform the raw IR into semantically tagged modules (e.g. AttentionBlock[MLA], FFN[Dense/MoE], KVCache, RoPE), attaching model-level metadata (head dims, expert counts, sharing, reuse).
    
3. **Parallelism Planner**
    
    Given hardware + user choices (TP/PP/DP/SP, KV sharding), derive per-module sharded tensor shapes, replication factors, and communication patterns (all-reduce, all-to-all, pipeline bubbles), producing a parallelised IR.
    
4. **Module-Level Performance Estimators**
    
    For each annotated + sharded module, estimate FLOPs, memory traffic, KV/cache footprint, communication volume, and achievable efficiency as a function of (batch, seq_in/out, dtype, chip count, interconnect). Roofline level estimation is fine: it should be possible to validate the calculations with a spreadsheet. Also add editable ‘efficiency factors’ (e.g., GEMM efficiency, all-reduce efficiency) to allow for calibration from a small set of reference benchmarks.
    
5. **Global Composer & Bottleneck Analyser**
    
    Interactive configuration system allowing per-layer parallelism choices, automatic minimum hardware calculation, and system-level metric aggregation with bottleneck attribution.
    

### App structure

1. **Hardware & System Library (editable)** ✅
    - JSON-based configs in `backend/data/hardware_configs/` directory
    - Per chip + system profile: peak FLOPs by dtype, HBM size/bw, interconnect bw/lat
    - Available configs: GB200 single, NVL-72 rack, H100 80GB
    - Easily portable to other projects (standard JSON format)
    - TODO: Versioned profiles + diff view, power/cost fields
2. **Model Library + IR Inspector (auditable/editable)**
    - HF link → generated model IR (layers/modules + full tensor list + shapes + KV layout).
    - Editable tables, with provenance tags (from config / from code / user override) and versioning + diffs.
3. **Run Config & Results Dashboard (main page)**
    - Inputs: batch, seq_in/out, dtype, parallelism (TP/PP/DP/SP), KV sharding, chips/nodes.
    - Explicitly separate prefill and decode (different batch size, parallelism etc)
    - Outputs: TTFT, TBOT, tokens/s, memory headroom, comm overhead, utilisation, bottleneck attribution.
    - Built-in validation warnings (OOM, invalid splits, unsupported settings).
4. **Layer/Module Drilldown**
    - From dashboard → per-layer/module breakdown (FLOPs, bytes, comm, efficiency, timeline prefill/decode).
    - Click any number → show formula + assumptions used.
5. **Assumptions & Audit Log (global)**
    - Always-accessible panel listing all non-default assumptions/overrides, provenance, and estimated impact.
    - One-click export of the full audit trail.
    - Constraint checker early (divisibility, TP vs heads, MoE routing vs EP, KV sharding feasibility, memory limits)
6. **Parallelism & Placement View (optional initially)**
    - Visualise PP stages / tensor sharding / KV placement and "what runs where".
    - Useful once multi-node + pipeline modeling is solid.
7. **Run Comparison + Parameter Sweeps (optional initially)**
    - Compare pinned runs (delta TTFT/TBOT/throughput/memory/bottlenecks).
    - Sweep mode (vary selected params) with charts + export.
8. **Reproducible Import/Export**
    - Export/import a single bundle: (model IR + hardware profile + run params + overrides) as JSON/YAML to reproduce/share results.

---

## Implementation Status (January 2026)

### Steps 1-4: Core Backend ✅

**Step 1: IR Builder** ✅
- `ir_builder.py`: Parses HF configs, creates ModelIR
- Works with TinyLlama, ready for Llama 3.3 70B

**Step 2: Layer Implementations** ✅
- `backend/layers/attention.py`:
  - AttentionLayer (MHA) with TP, CP support
  - GroupedQueryAttentionLayer (GQA) with TP, CP support
  - Comprehensive test suite: MHA-1/2/3, CP-1/2/3/4/5, GQA-1/2/3
- `backend/layers/mlp.py`:
  - MLPLayer (2-proj), GatedMLPLayer (3-proj SwiGLU)
  - TP, SP support
  - Test suite: docs/mlp_tests.md
- `backend/layers/norm.py`:
  - NormLayer (LayerNorm, RMSNorm)
- `backend/layers/embedding.py`:
  - EmbeddingLayer (vocab × hidden, replicated)

**Step 3: Parallelism** ⚠️
- Each layer handles its own parallelism (TP, CP, SP)
- Need: Model-level parallelism planner to coordinate

**Step 4: Module Performance Estimators** ✅
- All layers implement: `compute_flops()`, `compute_weight_memory()`, `compute_activation_memory()`, `compute_kv_cache()`, `_compute_communication_bytes()`
- Phase-aware (prefill vs decode)
- Parallelism-aware (TP, CP, SP)
- Extensively tested

### Step 5: Global Composer 🔲

**Interactive UX Flow:**

1. **Model + Hardware Selection**
  - Frontend selects a `model_id` (e.g., llama_3.3_70b) and `hardware_config` (e.g., nvidia_nvl72_rack)
  - Discover valid IDs via `GET /models` and `GET /hardware`
  - Stateless: all context is provided on each request; no server-side session

2. **Validation & Parameter Check**
   - Auto-populate all layers from config (attention, MLP, norm, embedding)
   - Run validation: total parameter count matches expected, layer configs valid
   - Display model summary: num layers, hidden size, attention heads, vocab, etc.

3. **Per-Layer Parallelism Configuration** (Interactive)
   - For each layer type (embedding, attention, MLP, norm), user chooses:
     - Parallelism strategy: TP, CP, SP, or combinations (TP×CP, TP×SP)
     - Number of chips: How many chips to use (must divide evenly)
   - Example: TP=4 for attention, TP=8 for MLP, TP=1 for norm/embedding
   - UI shows current memory and compute per chip for each choice

4. **Minimum System Calculation** (Automatic)
   - Given per-layer parallelism choices, calculate minimum chips needed:
     - Same chips can run different layers (weight sharing)
     - Example: TP=4 attention + TP=8 MLP → need max(4, 8) = 8 chips if weights fit
     - Check: Do all layer weights fit on the chips simultaneously?
     - If not, calculate minimum chips needed (memory-bound)
   - Display: "Your config requires minimum X chips (Y GB memory per chip)"

5. **System-Level Metrics** (Display)
   - Aggregate metrics across all layers:
     - **Prefill**: Total FLOPs, latency (ms), throughput (tokens/s)
     - **Decode**: Total FLOPs/token, latency (ms), throughput (tokens/s)
     - **Memory**: Total weights, activations, KV cache per chip
     - **Communication**: Total bytes per iteration (all-reduce, all-to-all)
     - **Bottleneck**: Compute-bound vs memory-bound vs communication-bound
   - Per-layer breakdown: Click any layer to see detailed metrics

**Future Enhancements:**

- **Parameter Sweep Mode**: Automatically sweep across different deployment topologies
  - Vary TP/CP/SP values, find Pareto frontier (latency vs throughput vs cost)
  - Generate comparison charts, export results

- **Layer-Level Bottleneck Analysis**: Identify which layers are bottlenecks
  - Show compute utilization, memory bandwidth utilization per layer
  - Suggest optimizations (increase TP, enable SP, use CP for long context)

- **System-Level Bottleneck Analysis**: Identify global bottlenecks
  - Compute-bound: GPU utilization high, suggest more efficient ops
  - Memory-bound: BW saturation, suggest activation checkpointing or quantization
  - Communication-bound: Interconnect saturation, suggest different parallelism

- **LLM-Assisted Troubleshooting**: Integration with LLM to debug issues
  - User describes problem ("OOM on chip 3", "low throughput")
  - LLM analyzes config, suggests fixes (reduce batch, increase TP, enable CP)

**Implementation Status:**
 
 - Backend provides stateless endpoints:
   - `GET /hardware`, `GET /hardware/{config_name}`
   - `GET /models`, `GET /models/{model_id}`
   - `GET /api/layers?model_id=...` (layer categories + counts)
  - Removed: `POST /config/load` (use stateless endpoints instead)
   - `POST /config/layer-metrics` (per-layer metrics for a representative layer)
   - `POST /config/system-metrics` (aggregate system metrics)
 - Legacy `estimator.py` and `/estimate` endpoint removed; replaced by the stateless API above.

### Llama 3.3 70B Support ✅

**Architecture:**
- 80 layers, 8192 hidden, 64 heads, 8 KV heads (GQA 8:1)
- Head dim: 128, intermediate: 28672
- Vocab: 128256, SwiGLU activation, RMSNorm, RoPE (500K theta)

**Status:**
- ✅ GQA attention layer
- ✅ Gated MLP (SwiGLU)
- ✅ RMSNorm
- ✅ Embedding layer
- ✅ Full model construction test: `test_llama_33_70b.py`
- ✅ Metrics computed: 70.6B params, 141.11 GB weights, 291.38 TFLOPs (prefill)

**Test Results:**
```
Embedding:         2.10 GB weights
Per-layer:         ~302 MB attention, ~1.4 GB MLP
Total (80 layers): 70.6B params, 186.84 GB total memory
```

**Next steps:**
- Expand communication modeling and bottleneck attribution
- Calibration against public benchmarks (inferencemax.semianalysis.com)
- RoPE modeling (document as minimal overhead)


