# Model Builder Plan

> **Note**: This document is a structured prompt for an AI agent to follow step-by-step. It is NOT a deterministic script. The agent should use judgment, adapt to edge cases, and ask clarifying questions when needed.

---

## Quick Start (Read This First)

**Your task**: Create a validated model config for a HuggingFace model.

**Key files**:
| Path | Purpose |
|------|---------|
| `model_builder/utils.py` | HF inspection & validation tools (run these first) |
| `backend/data/model_configs/llama_3.3_70b.json` | Output template to follow |
| `backend/layers/` | Supported layer implementations (inspect dynamically) |
| `model_builder/gaps/` | Store gap reports for unsupported layers |

**Workflow**:
1. Run `utils.py` functions to extract HF model info
2. Validate against public sources (web search) and use `validate_config()`
3. Use `get_supported_layers()` then inspect `backend/layers/` for mapping
4. Generate config JSON (include ALL layers, mark unsupported)
5. Create gap reports for any unsupported layers

**Checkpoints** (pause and confirm with user):
- After Step 2: Share validation findings before proceeding
- After Step 4: Review config before saving
- After Step 5: Review gap reports before saving

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
   - Documentation references (PyTorch, vLLM, SGLang)
   - Parallelism strategies for that layer type
   - Minimal test suite specification
   - Everything needed to implement the layer later

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
    get_supported_layers
)

# 1. Get high-level config (hidden_size, num_layers, etc.)
config = get_model_config("meta-llama/Llama-3.3-70B-Instruct")

# 2. Dry-run tensor inspection (no weight download)
tensors = inspect_model_structure("meta-llama/Llama-3.3-70B-Instruct")

# 3. Save for reference
save_tensor_inspection(tensors, "output/tensors.csv")

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

When a tensor pattern doesn't match any supported Mantile layer, create a **Layer Gap Report**:

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

#### 4.2 Documentation References

Provide links to implementation references:

- **PyTorch**: Core tensor operations and autograd
  - `torch.nn.Linear`, `torch.topk`, `torch.scatter_add`
  
- **vLLM**: Optimized inference implementations
  - `vllm/model_executor/layers/fused_moe/` 
  - https://docs.vllm.ai/en/latest/

- **SGLang**: Alternative inference framework
  - Router implementation patterns
  - https://sgl-project.github.io/

#### 4.3 Parallelism Strategies

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

#### 4.4 Test Suite Specification

Specify minimal but comprehensive tests:

```python
# Tests needed for MoERouterLayer

def test_moe_router_output_shape():
    """Router should output (batch, seq_len, num_experts) routing weights"""
    pass

def test_moe_router_top_k():
    """Only top-k experts should have non-zero weights"""
    pass

def test_moe_flops_calculation():
    """FLOPs should account for: router + (top_k * expert_flops)"""
    pass

def test_moe_expert_parallelism():
    """Memory and compute should split correctly across expert-parallel chips"""
    pass

def test_moe_communication():
    """All-to-All communication volume should be calculated correctly"""
    pass
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

- [ ] HF config successfully fetched
- [ ] Dry-run tensor inspection completed
- [ ] Parameter count matches public claims (within 1%)
- [ ] All layer types identified and mapped
- [ ] Unsupported layers have gap reports
- [ ] Validation sources documented
- [ ] JSON schema validates
- [ ] Config loads in Mantile without errors
