# How to Create Model Configs

> **Note**: This document is a structured prompt for an AI agent to follow step-by-step. It is NOT a deterministic script. The agent should use judgment, adapt to edge cases, and ask clarifying questions when needed.

---

## ⚠️ CRITICAL: File Boundary Rules

**READ THIS BEFORE DOING ANYTHING ELSE.**

You are operating in a restricted workspace. The following rules are **non-negotiable**:

### Allowed Actions
| Action | Allowed Location |
|--------|------------------|
| **Create** new files | `model_builder/agent_scratchpad/` **ONLY** |
| **Modify** existing files | `model_builder/agent_scratchpad/` **ONLY** |
| **Read** files | Anywhere in the repo (read-only) |

### Forbidden Actions
- ❌ **DO NOT** create files outside `model_builder/agent_scratchpad/`
- ❌ **DO NOT** modify any file outside `model_builder/agent_scratchpad/`
- ❌ **DO NOT** create test scripts, helper scripts, or temp files in any other directory
- ❌ **DO NOT** modify `backend/`, `frontend/`, or root-level files without explicit user approval

### Exception: Final Deliverables (User Approval Required)
The following files may be created/modified **ONLY after explicit user approval**:
- `backend/data/model_configs/{model_id}.json` — Final validated model config
- `model_builder/gaps/{layer_type}.md` — Gap reports for unsupported layers

**Before creating any deliverable**, you MUST:
1. Show the user the complete file content you intend to write
2. Wait for explicit approval (e.g., "yes", "approved", "go ahead")
3. Only then create/modify the file

### Scratchpad Usage
Use `model_builder/agent_scratchpad/` for:
- Test scripts (`test_*.py`)
- Exploratory code
- Temporary CSV/JSON exports
- Debug outputs
- Validation scripts

Example:
```bash
# ✅ Correct
model_builder/agent_scratchpad/test_attention.py
model_builder/agent_scratchpad/tensors_debug.csv

# ❌ Wrong - will pollute the repo
test_attention.py
backend/test_model.py
scripts/temp_validate.py
```

---

## Quick Start (Read This First)

**Your task**: Create a validated model config for a HuggingFace model.

**Key files**:
| Path | Purpose |
|------|---------|
| `model_builder/utils.py` | HF inspection & validation tools (run these first) |
| `backend/data/model_configs/meta-llama_Llama-3.3-70B-Instruct.json` | Output template to follow |
| `backend/layers/` | Supported layer implementations (inspect dynamically) |
| `model_builder/gaps/` | Store gap reports for unsupported layers |

**Workflow**:
1. Run `utils.py` functions to extract HF model info
2. Validate against public sources (web search) and use `validate_config()`
3. Use `get_supported_layers()` then inspect `backend/layers/` for mapping
4. Generate config JSON (include ALL layers, mark unsupported)
5. Create gap reports for any unsupported layers
6. **Produce final deliverables summary** (mandatory)

**Gap Tracking** (CRITICAL):
- Maintain a running gap list in `model_builder/agent_scratchpad/gap_tracker.md`
- Add each gap **immediately** when identified (don't wait until the end)
- Format: `| Gap Type | Feature | Severity | Report File |`

**Checkpoints** (pause and confirm with user):
- After Step 2: Share validation findings before proceeding
- After Step 3: Share **all identified gaps** before proceeding to config generation
- After Step 4: Review config before saving
- After Step 5: Review gap reports before saving
- After Step 6: Present final deliverables summary for confirmation

---

## Section 1: Objectives

### Primary Goal
Enable an AI agent to create a complete, validated model configuration file for any HuggingFace transformer model that Mantile can use for performance estimation.

### Deliverables

1. **Model Configuration File** (`backend/data/model_configs/{model_id}.json`)
   - Complete architectural parameters extracted from HuggingFace
   - Layer type mappings to Mantile's layer classes
   - Validated parameter counts and dimensions
   - Support status for each layer type (supported/unsupported)

2. **Validation Report**
   - Cross-reference with public sources (papers, HF cards, blogs)
   - Parameter count verification
   - Architectural consistency checks

3. **Layer Gap Analysis** (when unsupported layers are detected)
   - Example tensor output from dry run
   - **Tensor analysis**: Explanation of EVERY unusual tensor (not just listing)
   - **Dimension mismatch analysis**: Architectural interpretation of any mismatches
   - **Implementation hints**: Base class, methods to override, special handling
   - **FLOP/memory impact summary**: Quantified impact of each gap
   - Documentation references (PyTorch, vLLM, SGLang, papers)
   - Parallelism strategies for that layer type
   - Everything needed to implement the layer later WITHOUT additional context

4. **Test Suite for Gaps** (`tests/{layer_type}_tests.md`) — **MANDATORY**
   - **3-5 complete tests per gap** with step-by-step calculations
   - Must follow exact format from existing test files
   - Covers single chip + parallelism scenarios
   - All expected values derived and shown
   - See Section 4.4 for detailed requirements

---

## Section 2: Plan

### Step 1: Extract HuggingFace Model Information

**Tools**: `model_builder/utils.py`

The agent should use the utilities in `utils.py` to extract model information:

```python
from model_builder.utils import (
    get_model_config,
    inspect_model_structure,
    save_tensor_inspection,
    analyze_layer_structure,
    count_parameters,
    estimate_memory,
    validate_config,
    validate_model_config_file,
    get_supported_layers
)

# 1. Get high-level config (hidden_size, num_layers, etc.)
config = get_model_config("meta-llama/Llama-3.3-70B-Instruct")

# 2. Dry-run tensor inspection (no weight download)
tensors = inspect_model_structure("meta-llama/Llama-3.3-70B-Instruct")

# 3. Save for reference
save_tensor_inspection(tensors, "model_builder/agent_scratchpad/tensors.csv")

# 4. Analyze layer patterns
structure = analyze_layer_structure(tensors)

# 5. Compute totals for validation
params = count_parameters(tensors)
memory = estimate_memory(tensors, dtype="float16")

print(f"Parameters: {params['total_formatted']} ({params['total']:,})")
print(f"Memory (fp16): {memory['gb']:.1f} GB")
```

**Expected Output**:
- HF config dictionary with all architecture parameters
- Complete tensor list with names, shapes, and dtypes
- Layer structure analysis (attention type, MLP type, etc.)
- Parameter count and memory estimates for validation

---

### Step 2: Validate Against Public Information

**Method**: Manual web search + programmatic validation

The agent should:

1. **Search for authoritative sources**:
   - ArXiv papers (original model paper)
   - HuggingFace model cards
   - Technical blogs (official company blogs preferred)
   - Technical reports

2. **Verify key parameters**:
   - Total parameter count (match HF config vs public claims)
   - Architecture dimensions (hidden_size, num_layers, num_heads)
   - Attention type (MHA, GQA, MQA) and head counts
   - MLP architecture (standard, gated/SwiGLU, MOE)
   - Normalization type (LayerNorm, RMSNorm)
   - Special features (rope, alibi, tied embeddings, etc.)

3. **Run programmatic validation**:
   ```python
   # Validate against expected values from public sources
   result = validate_config(
       tensors,
       expected_params=70_000_000_000,  # From model card
       expected_layers=80                # From paper
   )
   
   if result['valid']:
       print("✓ Validation passed")
   else:
       for disc in result['discrepancies']:
           print(f"⚠️ {disc}")
   ```

4. **Document findings**:
   - If HF config differs from public info, note the discrepancy
   - Flag for human review if critical parameters don't match

**Example validation notes**:
```
Source: meta-llama/Llama-3.3-70B-Instruct model card
- Confirmed: 70B parameters (69.5B actual, 0.7% diff ✓)
- Confirmed: 80 layers, 8192 hidden size
- Confirmed: GQA with 64 heads, 8 KV heads
- Confirmed: SwiGLU MLP with 28672 intermediate size
```

---

### Step 3: Map to Mantile Layer Types

**Reference**: `backend/layers/` directory

#### Discovering Supported Layers

Use `get_supported_layers()` to dynamically discover what Mantile supports:

```python
from model_builder.utils import get_supported_layers

# Get current supported layers (reads from backend/layers/)
supported = get_supported_layers()

for class_name, info in supported.items():
    print(f"{class_name}: {info['docstring']}")
    print(f"  Source: {info['module']}")
```

Then inspect the actual layer files in `backend/layers/` to understand:
- Required constructor parameters for each layer class
- What tensor patterns each class expects
- How to extract parameters from tensor shapes

#### Mapping Process

For each component identified in Step 1:

1. **Identify the tensor pattern** (from dry-run output)
2. **Match to Mantile layer class**
3. **Extract required parameters**
4. **Mark support status**

**Example tensor-to-layer mapping**:
```
Tensor: model.layers.0.self_attn.q_proj.weight [8192, 8192]
        model.layers.0.self_attn.k_proj.weight [1024, 8192]  # num_kv_heads * head_dim
        model.layers.0.self_attn.v_proj.weight [1024, 8192]
        model.layers.0.self_attn.o_proj.weight [8192, 8192]

Maps to: GroupedQueryAttentionLayer
  - hidden_size: 8192 (from q_proj shape)
  - num_heads: 64 (8192 / 128)
  - num_kv_heads: 8 (1024 / 128)
  - head_dim: 128 (inferred)
```

---

### Step 4: Layer Gap Analysis (for unsupported layers)

When a tensor pattern doesn't match any supported Mantile layer, create a **Layer Gap Report**. The gap report serves as the complete specification for a future agent to implement the layer.

**⚠️ Gap reports must be COMPREHENSIVE. A future agent will use this to implement the layer without access to the model or additional context.**

#### 4.1 Example Tensors

Document the exact tensor names, shapes, and dtypes from the dry run:

```
UNSUPPORTED LAYER: Mixture of Experts (MoE) Router

Tensors detected:
  model.layers.0.block_sparse_moe.gate.weight: [8, 4096] (torch.float16)
  model.layers.0.block_sparse_moe.experts.0.w1.weight: [14336, 4096] (torch.float16)
  model.layers.0.block_sparse_moe.experts.0.w2.weight: [4096, 14336] (torch.float16)
  model.layers.0.block_sparse_moe.experts.0.w3.weight: [14336, 4096] (torch.float16)
  ... (8 experts total)
```

#### 4.2 Tensor Analysis (REQUIRED for each unusual tensor)

**⚠️ DO NOT just list tensors - EXPLAIN each one that differs from standard patterns.**

For EVERY tensor that is unusual or unexpected, provide:

```markdown
### Tensor: {tensor_name}

**Shape**: {shape}
**What it is**: {explanation of what this tensor represents}
**Why it's unusual**: {how it differs from standard patterns}
**Architectural interpretation**: {what architecture feature this indicates}
**Impact on implementation**:
  - FLOPs: {how this affects FLOP calculation, e.g., "+2*M*d for bias add"}
  - Memory: {how this affects memory, e.g., "+d*bytes for bias storage"}
  - KV Cache: {if applicable}
```

**Example analyses**:

```markdown
### Tensor: model.layers.0.self_attn.sinks (64,)

**What it is**: Attention sink tokens - learned vectors that act as "garbage collectors"
for attention in streaming/infinite context scenarios.
**Why it's unusual**: Standard attention has no sink tensors.
**Architectural interpretation**: This model uses StreamingLLM-style attention sinks
to maintain quality during very long generation without full KV cache.
**Reference**: https://arxiv.org/abs/2309.17453 (Efficient Streaming Language Models)
**Impact on implementation**:
  - FLOPs: Minimal - sink attention is computed but small (64 positions)
  - Memory: +64 * bytes_per_elem per layer for sink vectors
  - KV Cache: Sink tokens are always retained even with sliding window

### Tensor: model.layers.0.self_attn.q_proj.bias (4096,)

**What it is**: Bias term for query projection.
**Why it's unusual**: Many modern LLMs (LLaMA, Mistral) omit biases for efficiency.
**Impact on implementation**:
  - FLOPs: +M additions per projection (4 projections = +4*M ops, negligible vs matmul)
  - Memory: +hidden_size * bytes_per_elem per projection with bias
  - Total bias memory per attention layer: 4 * hidden_size * bytes = {calculate}
```

#### 4.3 Dimension Mismatch Analysis

**⚠️ If any dimensions don't match expected patterns, EXPLAIN the architecture.**

Common mismatches and what they mean:

| Mismatch | Likely Architecture | Explanation |
|----------|---------------------|-------------|
| `q_proj output > hidden_size` | Multi-head Latent Attention (MLA) | Q projects to larger latent space |
| `kv_proj << q_proj` | Grouped Query Attention (GQA) | Fewer KV heads than query heads |
| `o_proj input != hidden_size` | Non-standard attention width | Attention operates in different dimension |
| `intermediate_size != 4*hidden` | Custom MLP ratio | Model uses non-standard expansion |

**Required format**:

```markdown
### Dimension Mismatch: {description}

**Observed**: {what you see, e.g., "q_proj: (4096, 2880), hidden_size: 2880"}
**Expected (standard)**: {what standard pattern would be, e.g., "q_proj: (2880, 2880)"}
**Interpretation**: {what architecture this indicates}

**Derived parameters**:
- actual_query_dim = 4096
- num_heads = 32
- head_dim = 4096 / 32 = 128
- hidden_size = 2880 (smaller than query dim)

**Impact**:
- Q/K/V projections are NOT square matrices
- Output projection maps attention_dim → hidden_size, not hidden → hidden
- FLOP calculation must use actual dimensions, not assume hidden_size throughout
```

#### 4.4 Implementation Hints (REQUIRED)

Provide guidance for the implementing agent:

```markdown
## Implementation Guidance

### Recommended Base Class
- **Extend**: `{ClassName}` from `backend/layers/{module}.py`
- **Reason**: {why this base class is appropriate}
- **Alternative**: {if applicable, another option and why}

### Methods to Override
| Method | Reason |
|--------|--------|
| `compute_flops()` | {what changes from base} |
| `compute_weight_memory()` | {what changes from base} |
| `compute_activation_memory()` | {what changes from base} |
| `compute_kv_cache()` | {what changes from base, or "N/A"} |
| `_validate_parallelism()` | {if custom validation needed} |

### New Constructor Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `{param}` | `{type}` | {description} |

### Special Handling Required
- [ ] {Special case 1, e.g., "Circular KV cache buffer for sliding window"}
- [ ] {Special case 2, e.g., "All-to-all communication for expert routing"}
- [ ] {Special case 3}

### KV Cache Considerations (if attention layer)
- **Standard behavior**: {how KV cache normally works}
- **Modified behavior**: {how this layer changes it}
- **Implementation note**: {specific guidance, e.g., "Use min(seq_len, window_size) for cache length"}
```

#### 4.5 FLOP and Memory Impact Summary

**⚠️ REQUIRED: Quantify the impact of each gap on calculations.**

```markdown
## Impact Summary

### Additional FLOPs (per layer, per token)
| Component | Formula | Example (d=2880, E=128) |
|-----------|---------|-------------------------|
| Bias adds (attention) | 4 * M | 4 * 2048 = 8,192 |
| Router | 2 * M * d * E | 2 * 2048 * 2880 * 128 = 1.5B |
| {other} | {formula} | {value} |

### Additional Memory (per layer)
| Component | Formula | Example (d=2880) |
|-----------|---------|------------------|
| Attention biases | 4 * d * bytes | 4 * 2880 * 2 = 23KB |
| Sink vectors | sink_size * bytes | 64 * 2 = 128B |
| {other} | {formula} | {value} |

### Communication Changes
| Pattern | When | Payload |
|---------|------|---------|
| All-to-All | MoE routing | M * d * bytes per chip |
| {other} | {when} | {payload} |
```

#### 4.6 Documentation References

Provide links to implementation references:

- **PyTorch**: Core tensor operations and autograd
  - `torch.nn.Linear`, `torch.topk`, `torch.scatter_add`
  
- **vLLM**: Optimized inference implementations
  - `vllm/model_executor/layers/fused_moe/` 
  - https://docs.vllm.ai/en/latest/

- **SGLang**: Alternative inference framework
  - Router implementation patterns
  - https://sgl-project.github.io/

- **Papers** (if applicable):
  - {Paper title}: {arxiv link}

#### 4.7 Parallelism Strategies

Document how this layer is typically parallelized:

```
MoE Parallelism Strategies:
1. Expert Parallelism (EP): Distribute experts across devices
   - Each device holds subset of experts
   - Requires All-to-All communication

2. Tensor Parallelism (TP): Split expert weights across devices
   - Each expert is tensor-parallel
   - Same as standard TP for dense layers

3. Hybrid EP+TP: Combine both strategies
   - EP across nodes, TP within nodes
   - Common for large-scale deployments
```

#### 4.8 Test Suite Specification

**⚠️ MANDATORY: Create 3-5 complete tests for each gap identified.**

Tests in Mantile are written in **Markdown format** with step-by-step calculations. Before writing tests, study the existing test files to understand the expected format:

- **Reference files** (MUST READ before writing tests):
  - `tests/README.md` - Overview and conventions
  - `tests/attention_tests.md` - Full examples for attention layers
  - `tests/mlp_tests.md` - Full examples for MLP layers

##### Test Requirements

| Requirement | Details |
|-------------|---------|
| **Minimum tests per gap** | 3-5 tests covering different scenarios |
| **Test coverage** | Single chip, TP, and other relevant parallelism modes |
| **Calculation detail** | Every number must be derived step-by-step |
| **Exact format** | Follow the structure in existing test files exactly |
| **Output file** | `tests/{layer_type}_tests.md` (add to repo, not scratchpad) |

##### Required Test Scenarios (pick 3-5 from these)

1. **Single chip baseline** - No parallelism, establishes ground truth
2. **Tensor Parallel (TP=4 or TP=8)** - Tests weight/computation sharding
3. **Different batch/sequence sizes** - Tests scaling behavior
4. **Decode phase** (if applicable) - Tests KV cache behavior
5. **Layer-specific variants** - E.g., for MoE: different top_k, expert counts

##### Test File Structure (STRICT FORMAT)

Your test file MUST follow this exact structure. See `tests/attention_tests.md` for a complete example.

```markdown
# [Layer Type] Tests

## Overview

These tests verify the correctness of:
- **FLOPs computation**: [specific operations for this layer]
- **Weight memory**: Parameter storage per chip and in aggregate
- **Activation memory**: Intermediate tensor buffers during forward pass
- **Communication**: Inter-chip data transfer requirements

## Test Summary

| Test | Config | Key Params | FLOPs/chip | Weight/chip | Activation/chip |
|------|--------|------------|------------|-------------|-----------------|
| 1 | No parallelism | ... | ... | ... | ... |
| 2 | TP=4 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |

## Conventions Used in All Tests

* `bytes_per_elem = 2` (FP16/BF16)
* `B=batch_size`, `S=seq_len`, `M=B*S`
* **FLOPs for GEMM** `(M×K)@(K×N)` = `2*M*K*N`
* **communication_bytes** = payload bytes (logical tensor size)
* [Add layer-specific conventions here]

---

# Test [PREFIX]-1 — [Description], single chip

### Test case parameters

* [param_1] = **[value]**
* [param_2] = **[value]**
* ...
* num_chips = **1**
* tensor_parallel `tp` = **1**

Derived:
* [derived_param] = [formula] = [value]

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) [Operation name]**
[Shape notation]: `X[M,K] @ W[K,N]`
* FLOPs = `2 * M * K * N`
* = `2 * [value] * [value] * [value] = [result]`

**(b) [Next operation]**
...

**Total FLOPs**
* `flops_total = [component_1] + [component_2] + ...`
* = `[value] + [value] = [total]`

Per chip (1 chip):
* **flops_per_chip = [total]**

#### 2) Weight memory

* [weight_1]: `[shape]` → `[elements] * bytes_per_elem = [bytes]`
* [weight_2]: `[shape]` → `[elements] * bytes_per_elem = [bytes]`
* Total weight elems = [sum] → bytes = `[sum] * 2 = [total_bytes]`

Per chip:
* **weight_memory_per_chip = [bytes]**
* **weight_memory_total = [bytes]**

#### 3) Activation memory

* [tensor_1]: `[shape]` → `[elements] * bytes_per_elem = [bytes]`
* [tensor_2]: `[shape]` → `[elements] * bytes_per_elem = [bytes]`
* Total activation bytes = **[sum]**

Per chip:
* **activation_memory_per_chip = [bytes]**
* **activation_memory_total = [bytes]**

#### 4) KV cache (if applicable)

* [calculation or "Not applicable for this layer: **0**"]

#### 5) Communication

* Single chip: **0**

### Expected results

**Per-chip metrics**
* flops_per_chip: **[value]**
* weight_memory_per_chip: **[value]**
* activation_memory_per_chip: **[value]**
* kv_cache_per_chip: **[value]**

**Aggregate metrics**
* flops_total: **[value]**
* weight_memory_total: **[value]**
* activation_memory_total: **[value]**
* kv_cache_total: **[value]**

**Hardware-dependent (optional)**
* communication_bytes: **[value]**

---

# Test [PREFIX]-2 — [Description], TP = 4

[Repeat full structure with TP=4 calculations...]

---

# Test [PREFIX]-3 — [Description], [variant]

[Repeat full structure...]
```

##### Calculation Standards (NON-NEGOTIABLE)

1. **Every intermediate value must be shown**
   - ❌ Wrong: `FLOPs = 2,281,701,376`
   - ✅ Correct: `FLOPs = 2 * 256 * 1024 * 1024 = 536,870,912`

2. **Use consistent variable names**
   - Always use: `d` (hidden), `M` (tokens), `B` (batch), `S` (seq_len)
   - Define layer-specific: `E` (experts), `di` (intermediate), etc.

3. **Show per-chip AND total values**
   - Both must be calculated and listed in Expected results

4. **Communication must specify pattern**
   - State: all-reduce, all-gather, all-to-all, or reduce-scatter
   - Show payload shape and bytes

##### Example: MoE Test Suite (save to `tests/moe_tests.md`)

```markdown
## Test MOE-1: MoE Router + Experts, single chip

### Test case parameters

* hidden_size `d` = **4096**
* num_experts `E` = **8**
* top_k = **2**
* expert_intermediate `di` = **14336**
* batch_size `B` = **1**
* seq_len `S` = **2048**
* bytes_per_elem = **2**
* num_chips = **1**

Derived:
* `M = B*S = 2048` tokens

### Expected calculations (step-by-step)

#### 1) FLOPs

**(a) Router gating**
`X[M,d] @ W_gate[d,E]`:
* FLOPs = `2 * M * d * E`
* = `2 * 2048 * 4096 * 8 = 134,217,728`

**(b) Expert MLP (per active expert)**
Each token routed to top_k=2 experts.
Per expert (gated MLP): `3 * 2 * M * d * di`
* = `3 * 2 * 2048 * 4096 * 14336 = 722,204,057,600` per expert
* Total for top_k=2: `2 * 722,204,057,600 = 1,444,408,115,200`

**Total FLOPs**
* = `134,217,728 + 1,444,408,115,200 = 1,444,542,332,928`

#### 2) Memory

**(a) Router weights**
* `W_gate[d,E]`: `4096 * 8 * 2 = 65,536 bytes`

**(b) Expert weights (all 8 experts)**
* Per expert: `3 * d * di * 2 = 3 * 4096 * 14336 * 2 = 352,321,536 bytes`
* All experts: `8 * 352,321,536 = 2,818,572,288 bytes = 2.62 GB`

**Total Memory**
* = `65,536 + 2,818,572,288 = 2,818,637,824 bytes ≈ 2.62 GB`

### Expected Behavior

* Memory per chip: ~2.62 GB (weights only)
* Compute time: depends on hardware FLOP/s

### Rationale

MoE routes each token to top_k experts. Router is a small linear layer.
Only top_k experts compute per token, but all expert weights must be stored.
```

---

### Step 5: Generate Model Configuration

**Output**: `backend/data/model_configs/{model_id}.json`

Create the final configuration file following this schema:

```json
{
  "model_id": "model_name_here",
  "hf_model_id": "org/Model-Name",
  "name": "Human Readable Name",
  
  "hidden_size": 8192,
  "num_layers": 80,
  "vocab_size": 128256,
  "total_params": 69502369792,
  "total_params_formatted": "69.5B",
  
  "layer_types": [
    {
      "name": "attention",
      "class": "GroupedQueryAttentionLayer",
      "count": 80,
      "supported": true,
      "specs": {
        "layer_idx": 0,
        "input_dim": 8192,
        "output_dim": 8192,
        "parameter_count": 150994944,
        "hidden_size": 8192,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128
      }
    },
    {
      "name": "moe_router",
      "class": "MoERouterLayer",
      "count": 80,
      "supported": false,
      "gap_report": "See model_builder/gaps/moe_router.md",
      "specs": {
        "num_experts": 8,
        "top_k": 2,
        "expert_hidden_size": 14336
      }
    }
  ],
  
  "validated": true,
  "validation_notes": "Cross-referenced with official paper and HF model card",
  "validation_sources": [
    "https://arxiv.org/abs/...",
    "https://huggingface.co/org/model"
  ]
}
```

**Key requirements**:
- Include ALL layer types, even unsupported ones
- Mark `supported: false` for layers without Mantile implementations
- Link to gap reports for unsupported layers
- Include validation sources
- **Follow standard layer naming**: Use `attention`, `feedforward`, `norm`, `embedding`
- **Layer order should be**: attention, feedforward, norm, embedding
- **model_id must match filename** (without `.json` extension)

**Before saving, validate the config**:
```python
from pathlib import Path
from model_builder.utils import validate_model_config_file

# Validate the config file
result = validate_model_config_file(Path("backend/data/model_configs/model_id.json"))

if not result['valid']:
    print("Validation errors:")
    for error in result['errors']:
        print(f"  - {error}")
    # Fix errors before proceeding

if result['warnings']:
    print("Warnings:")
    for warning in result['warnings']:
        print(f"  - {warning}")
```

**CLI validation** (before committing):
```bash
# Validate all configs
python -m model_builder.utils validate backend/data/model_configs/

# Validate specific file
python -m model_builder.utils validate backend/data/model_configs/model_id.json
```

---

### Step 6: Final Deliverables Summary (MANDATORY)

**⚠️ DO NOT SKIP THIS STEP**

Before concluding your work, you MUST provide the user with a structured summary of ALL deliverables. This ensures nothing is missed.

#### 6.1 Gap Registry

Enumerate **every gap identified** during the process, even if minor:

```markdown
## Gap Registry

| # | Gap Type | Feature | Affected Layer(s) | Severity | Gap Report File | Status |
|---|----------|---------|-------------------|----------|-----------------|--------|
| 1 | MoE | 128 experts | feedforward | HIGH | gaps/moe_*.md | Created |
| 2 | Attention | Sliding Window (SWA) | attention | MEDIUM | gaps/attention_*.md | Created |
| 3 | Attention | YaRN RoPE scaling | attention | LOW | gaps/attention_*.md | Documented |
| 4 | General | Bias tensors | attention, feedforward | LOW | gaps/attention_*.md | Documented |

**Total gaps identified: 4**
**Gap reports created: 2**
```

Severity levels:
- **HIGH**: Core layer type unsupported (MoE, cross-attention, etc.)
- **MEDIUM**: Variant of supported layer (SWA, different normalization)
- **LOW**: Minor feature difference (biases, scaling factors)

#### 6.2 Test Suite Registry

For each gap, list the tests created:

```markdown
## Test Suite Registry

| Gap | Test File | # Tests | Test IDs | Status |
|-----|-----------|---------|----------|--------|
| MoE | tests/moe_tests.md | 4 | MOE-1, MOE-2, MOE-3, MOE-4 | Created |
| SWA | tests/swa_attention_tests.md | 3 | SWA-1, SWA-2, SWA-3 | Created |

**Total: 7 tests across 2 test files**
```

**Test requirements reminder:**
- Minimum **3-5 tests per gap**
- Must include single-chip baseline + parallelism variants
- All calculations shown step-by-step
- Follows exact format from `tests/attention_tests.md`

#### 6.3 Deliverables Checklist

Present this completed checklist to the user:

```markdown
## Deliverables Summary

### Config File
- [ ] Path: `backend/data/model_configs/{model_id}.json`
- [ ] Validation: PASSED / FAILED
- [ ] All layers mapped: YES / NO

### Gap Reports Created
- [ ] `model_builder/gaps/{gap_1}.md`
- [ ] `model_builder/gaps/{gap_2}.md`
- [ ] (list all)

### Test Suites Created (MANDATORY for each gap)
- [ ] `tests/{layer_1}_tests.md` - {N} tests (IDs: ...)
- [ ] `tests/{layer_2}_tests.md` - {N} tests (IDs: ...)
- [ ] (list all)

### Validation Summary
- [ ] Parameter count matches: YES / NO (X% difference)
- [ ] Architecture verified: YES / NO
- [ ] Sources documented: YES / NO

### Gap Count Verification
- Gaps identified during analysis: **N**
- Gap reports created: **M**
- Test suites created: **T** (must equal number of distinct gap types)
- Total tests written: **X** (minimum 3 per gap)
- Gaps documented in existing reports: **P**
- Unaccounted gaps: **N - M - P** (MUST BE 0)
```

#### 6.4 Final Message Template

Your completion message to the user MUST follow this structure:

```markdown
## ✅ Model Config Complete: {model_name}

### Deliverables
1. **Config file**: `backend/data/model_configs/{model_id}.json`
2. **Gap reports**: 
   - `gaps/{report_1}.md` - {brief description}
   - `gaps/{report_2}.md` - {brief description}
3. **Test suites**:
   - `tests/{layer_1}_tests.md` - {N} tests covering {scenarios}
   - `tests/{layer_2}_tests.md` - {N} tests covering {scenarios}

### Gap Summary
| Gap | Severity | Gap Report | Test Suite | # Tests |
|-----|----------|------------|------------|---------|
| {gap_1} | HIGH/MED/LOW | Created | Created | N |
| {gap_2} | HIGH/MED/LOW | Created | Created | N |

**Total: {N} gaps, {M} reports, {T} test suites, {X} tests**

### Validation
- Parameter count: {X}B (matches public claim of {Y}B, {Z}% diff)
- Architecture: Verified against {source}

### Next Steps
- Review gap reports for implementation priorities
- Run test suites after implementing layers
- {any model-specific notes}
```

---

## Appendix A: Example Tensor Output

Example dry-run output for Llama-style model:

```
kind,name,shape,dtype
param,model.embed_tokens.weight,"(128256, 8192)",torch.float16
param,model.layers.0.self_attn.q_proj.weight,"(8192, 8192)",torch.float16
param,model.layers.0.self_attn.k_proj.weight,"(1024, 8192)",torch.float16
param,model.layers.0.self_attn.v_proj.weight,"(1024, 8192)",torch.float16
param,model.layers.0.self_attn.o_proj.weight,"(8192, 8192)",torch.float16
param,model.layers.0.mlp.gate_proj.weight,"(28672, 8192)",torch.float16
param,model.layers.0.mlp.up_proj.weight,"(28672, 8192)",torch.float16
param,model.layers.0.mlp.down_proj.weight,"(8192, 28672)",torch.float16
param,model.layers.0.input_layernorm.weight,"(8192,)",torch.float16
param,model.layers.0.post_attention_layernorm.weight,"(8192,)",torch.float16
... (repeats for all layers)
param,model.norm.weight,"(8192,)",torch.float16
param,lm_head.weight,"(128256, 8192)",torch.float16
```

---

## Appendix B: Checklist

Before finalizing a model config, verify:

**Extraction & Validation**
- [ ] HF config successfully fetched
- [ ] Dry-run tensor inspection completed
- [ ] Parameter count matches public claims (within 1%)
- [ ] All layer types identified and mapped

**Config File**
- [ ] **Layer names follow standard convention** (attention, feedforward, norm, embedding)
- [ ] **No duplicate layer names**
- [ ] **model_id matches filename** (without .json)
- [ ] **Config passes validation**: `python -m model_builder.utils validate <config_file>`
- [ ] Validation sources documented
- [ ] JSON schema validates
- [ ] Config loads in Mantile without errors

**Gap Tracking (CRITICAL)**
- [ ] **Gap tracker maintained** during analysis (`agent_scratchpad/gap_tracker.md`)
- [ ] **All identified gaps enumerated** in final summary
- [ ] **Gap count verified**: gaps identified = reports created + documented in existing reports
- [ ] Unsupported layers have gap reports

**Gap Report Completeness (CRITICAL)**
- [ ] **Every unusual tensor explained** (not just listed) with:
  - What it is and why it's unusual
  - Architectural interpretation
  - FLOP and memory impact
- [ ] **Dimension mismatches analyzed** with architectural interpretation
- [ ] **Implementation hints provided**:
  - Recommended base class to extend
  - Methods that need overriding
  - New constructor parameters
  - Special handling required (e.g., circular KV cache)
- [ ] **FLOP/memory impact summary** with formulas and example calculations
- [ ] **Documentation references** include relevant papers (not just code repos)
- [ ] **Parallelism strategies** documented with communication patterns

**Test Suites (MANDATORY)**
- [ ] **3-5 tests created per gap** in `tests/{layer_type}_tests.md`
- [ ] **Test format matches** existing tests (`tests/attention_tests.md`, `tests/mlp_tests.md`)
- [ ] **Single-chip baseline test** included for each gap
- [ ] **Parallelism test (TP=4 or similar)** included for each gap
- [ ] **All calculations shown step-by-step** (no unexplained numbers)
- [ ] **Expected results section** with per-chip and aggregate metrics
- [ ] **Test IDs follow convention** (PREFIX-1, PREFIX-2, etc.)
- [ ] **Test summary table** at top of each test file

**Final Deliverables**
- [ ] **Final deliverables summary provided** (Step 6)
- [ ] **Test suite registry** included in summary
- [ ] All test files listed with test counts
