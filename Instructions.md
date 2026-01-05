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
    
    Aggregate module costs into end-to-end metrics (prefill/decode latency, throughput, memory headroom), model overlap and scheduling, and attribute bottlenecks (compute vs memory vs comm) with sensitivity to assumptions.
    

### App structure

1. **Hardware & System Library (editable)**
    - Per chip + system profile: peak FLOPs by dtype, HBM size/bw, interconnect bw/lat, collective efficiency knobs, power/cost (optional).
    - Versioned profiles + diff view.
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
