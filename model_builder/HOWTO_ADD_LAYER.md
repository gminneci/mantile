# How to Add a New Layer Implementation

> **Note**: This document is a structured prompt for an AI agent to implement new layer types in Mantile. It is NOT a deterministic script. The agent should use judgment, adapt to edge cases, and ask clarifying questions when needed.

---

## ⚠️ CRITICAL: File Boundary Rules

**READ THIS BEFORE DOING ANYTHING ELSE.**

You are operating in a restricted workspace. The following rules are **non-negotiable**:

### Allowed Actions
| Action | Allowed Location |
|--------|------------------|
| **Create** new layer files | `backend/layers/` (new layer classes) |
| **Modify** existing layer files | `backend/layers/*.py` (extend existing layers) |
| **Create/Modify** test code | `tests/run_tests.py` |
| **Create** scratchpad files | `model_builder/agent_scratchpad/` |
| **Read** files | Anywhere in the repo |

### Forbidden Actions
- ❌ **DO NOT** modify `backend/layers/base.py` without explicit user approval
- ❌ **DO NOT** modify model configs until implementation is complete
- ❌ **DO NOT** delete existing tests
- ❌ **DO NOT** modify frontend code

### Checkpoints (pause and confirm with user)
1. After Step 2: Share understanding of the layer before implementation
2. After Step 4: Review implementation before writing tests
3. After Step 6: Review test results before updating configs
4. After Step 7: Final review before completion

---

## Prerequisites

Before starting, you MUST have:

1. **Gap Report** (`model_builder/gaps/{layer_type}_{model}.md`)
   - Example tensors with shapes and dtypes
   - Identified gaps/differences from existing layers
   - Parallelism strategies documented
   - Documentation references

2. **Test Specifications** (`tests/{layer_type}_tests.md`)
   - 3-5 complete test cases with step-by-step calculations
   - Expected values for FLOPs, memory, communication
   - Single-chip and parallel configurations

If these don't exist, stop and ask the user to provide them or run the model config process first.

---

## Quick Start

**Your task**: Implement a new layer type in Mantile based on a gap report.

**Key files**:
| Path | Purpose |
|------|---------|
| `backend/layers/base.py` | Base `Layer` class and `LayerMetrics` dataclass |
| `backend/layers/attention.py` | Reference attention implementations |
| `backend/layers/mlp.py` | Reference MLP implementations |
| `backend/layers/__init__.py` | Export new layer classes here |
| `tests/run_tests.py` | Add test functions here |

**Workflow**:
1. Study the gap report and understand what's different
2. Research reference implementations (PyTorch, vLLM, SGLang)
3. Choose base class and design the new layer
4. Implement the layer class
5. Convert markdown tests to Python test code
6. Run tests and fix discrepancies
7. Update model config and validate

---

## Section 1: Study the Gap Report

### 1.1 Read the Gap Report Thoroughly

Open the gap report and extract:

```markdown
## Gap Report Analysis

### Layer Type
- [ ] Is this a NEW layer type (e.g., MoE) or a VARIANT of existing (e.g., SWA attention)?

### Tensor Patterns
- [ ] List all tensor names and shapes
- [ ] Identify any unexpected tensors (biases, sinks, etc.)
- [ ] Note any shape mismatches from standard patterns

### Key Differences from Existing Layers
- [ ] List each gap identified
- [ ] For each gap, note:
  - Impact on FLOPs calculation
  - Impact on memory calculation
  - Impact on communication pattern

### Parallelism Requirements
- [ ] Which parallelism strategies are needed?
- [ ] Any special communication patterns (all-to-all for MoE)?
```

### 1.2 Identify Implementation Approach

| Scenario | Approach |
|----------|----------|
| **Variant of existing layer** (e.g., SWA) | Extend existing class, override specific methods |
| **New layer type** (e.g., MoE) | Create new class extending `Layer` base |
| **Minor modification** (e.g., add biases) | Modify existing class with optional parameter |

**Decision tree**:
```
Is this a completely new layer type?
├── YES → Create new file in backend/layers/ extending Layer
└── NO → Is it a variant of Attention?
    ├── YES → Extend AttentionLayer or GroupedQueryAttentionLayer
    └── NO → Is it a variant of MLP?
        ├── YES → Extend MLPLayer or GatedMLPLayer
        └── NO → Extend Layer base class
```

---

## Section 2: Research Reference Implementations

### 2.1 Required References

Before implementing, study these sources in order:

1. **PyTorch Reference** (understand the math)
   - Search for the layer type in PyTorch source
   - Understand tensor shapes and operations
   - Note any fused operations

2. **vLLM Implementation** (understand inference optimizations)
   - Repository: `vllm-project/vllm`
   - Key paths:
     - `vllm/model_executor/layers/` - Layer implementations
     - `vllm/attention/` - Attention variants
     - `vllm/model_executor/layers/fused_moe/` - MoE implementations
   
3. **SGLang Implementation** (alternative patterns)
   - Repository: `sgl-project/sglang`
   - Often has cleaner, more readable implementations

### 2.2 Key Questions to Answer

```markdown
## Reference Implementation Notes

### PyTorch
- [ ] What are the core tensor operations?
- [ ] What are the input/output shapes?
- [ ] Are there any fused operations?

### vLLM
- [ ] How is KV cache handled?
- [ ] What parallelism is supported?
- [ ] Any custom kernels or optimizations?

### SGLang
- [ ] Any different design choices?
- [ ] Cleaner implementation patterns?
```

### 2.3 Example: MoE Research

```python
# Key operations in MoE:
# 1. Router: X @ W_router → scores for each expert
# 2. Top-K selection: select top_k experts per token
# 3. Expert computation: each expert is a full MLP
# 4. Combine: weighted sum of expert outputs

# vLLM reference: vllm/model_executor/layers/fused_moe/fused_moe.py
# - Uses custom Triton kernels for fused routing
# - Supports expert parallelism (EP) and tensor parallelism (TP)
```

---

## Section 3: Design the Layer Class

### 3.1 Study the Base Class

Read `backend/layers/base.py` to understand:

```python
class Layer(ABC):
    """
    Subclasses must implement:
        - compute_flops(): Calculate FLOPs for given batch/seq_len/phase
        - compute_weight_memory(): Calculate parameter memory footprint
        - compute_activation_memory(): Calculate activation memory footprint
        - compute_kv_cache(): Calculate KV cache size (if applicable)
        - _get_num_packages(): Return number of packages this layer uses
    
    Subclasses should define:
        - SUPPORTED_PARALLELISM: Set of supported parallelism types
    """
```

### 3.2 Design Document

Before coding, write a brief design:

```markdown
## Layer Design: {LayerName}

### Class Hierarchy
- Extends: {BaseClass}
- New class name: {ClassName}

### Constructor Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| ... | ... | ... |

### SUPPORTED_PARALLELISM
- {parallelism_1}: {description}
- {parallelism_2}: {description}

### Method Overrides
| Method | Changes from Base |
|--------|-------------------|
| compute_flops | {description} |
| compute_weight_memory | {description} |
| compute_activation_memory | {description} |
| compute_kv_cache | {description or "N/A"} |
| _get_num_packages | {description} |

### FLOPs Formula
```
{step-by-step formula matching test spec}
```

### Memory Formula
```
{step-by-step formula matching test spec}
```

### Communication Pattern
- {pattern}: {description}
```

---

## Section 4: Implement the Layer

### 4.1 Implementation Checklist

- [ ] Create/modify the layer file
- [ ] Add docstring with architecture description
- [ ] Define `SUPPORTED_PARALLELISM` class attribute
- [ ] Implement `__init__` with all required parameters
- [ ] Implement `_get_num_packages()`
- [ ] Implement `compute_flops()`
- [ ] Implement `compute_weight_memory()`
- [ ] Implement `compute_activation_memory()`
- [ ] Implement `compute_kv_cache()` (if applicable)
- [ ] Override `_validate_parallelism()` if needed
- [ ] Export from `backend/layers/__init__.py`

### 4.2 Code Template

```python
"""
{Layer Type} Implementation

{Brief description of what this layer models}
"""

from typing import Optional
from .base import Layer, Phase, DataType


class {ClassName}(Layer):  # or extend specific layer class
    """
    {Detailed architecture description}
    
    Architecture:
        {Description of the computation}
    
    Parameters:
        {List of weight tensors and their shapes}
    
    Supported parallelism:
        - {parallelism_type}: {description}
    """
    
    SUPPORTED_PARALLELISM = {"{parallelism_1}", "{parallelism_2}"}
    
    def __init__(
        self,
        layer_idx: int,
        # Layer-specific parameters
        hidden_size: int,
        # ... other params
        dtype: DataType | str = "bf16",
        parallelism: Optional[dict] = None
    ):
        """
        Args:
            layer_idx: Layer index
            hidden_size: Model hidden dimension
            # ... document all params
            dtype: Numerical precision
            parallelism: Parallelism config
        """
        self.hidden_size = hidden_size
        # Store all layer-specific params
        
        # Calculate param count
        self.param_count = ...
        
        super().__init__(layer_idx, dtype, parallelism)
    
    def _get_num_packages(self) -> int:
        """Calculate number of packages based on parallelism config."""
        tp = self.parallelism.get("tensor_parallel", 1)
        # Add other parallelism dimensions as needed
        return tp
    
    def compute_flops(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        Compute FLOPs per package.
        
        {Document the formula from test spec}
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        B = batch_size
        # ... implement matching test spec formulas
        
        flops = 0
        # Step 1: ...
        flops += ...
        # Step 2: ...
        flops += ...
        
        return int(flops)
    
    def compute_weight_memory(self) -> int:
        """
        Compute weight memory per package in bytes.
        
        {Document which weights and sharding}
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        params_per_package = self.param_count // tp
        return int(params_per_package * self.dtype.bytes_per_element)
    
    def compute_activation_memory(self, batch_size: int, seq_len: int, phase: Phase) -> int:
        """
        Compute activation memory per package in bytes.
        
        {Document which tensors are counted}
        """
        tp = self.parallelism.get("tensor_parallel", 1)
        B = batch_size
        # ... implement
        
        elems = 0
        # Tensor 1: shape [...]
        elems += ...
        
        return int(elems * self.dtype.bytes_per_element)
    
    def compute_kv_cache(self, batch_size: int, seq_len: int) -> int:
        """Compute KV cache size (return 0 if not applicable)."""
        return 0  # or implement for attention layers
```

### 4.3 Export the Layer

Add to `backend/layers/__init__.py`:

```python
from .{module} import {ClassName}

__all__ = [
    # ... existing exports
    "{ClassName}",
]
```

---

## Section 5: Write Test Code

### 5.1 Test Code Structure

Each test in the markdown file becomes a test case in `tests/run_tests.py`:

```python
def test_{layer_type}_layers():
    """Run {LayerType} layer tests from tests/{layer_type}_tests.md"""
    print("=" * 80)
    print("{LAYER_TYPE} LAYER TESTS")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test {PREFIX}-1: {Description}
    print("\n--- Test {PREFIX}-1: {Description} ---")
    try:
        layer = {ClassName}(
            layer_idx=0,
            # ... params from test spec
            dtype='bf16',
            parallelism={'tensor_parallel': 1}
        )
        metrics = layer.compute_metrics(batch_size=..., seq_len=..., phase='prefill')
        
        expected = {
            'flops_per_chip': ...,  # From test spec
            'weight_memory_per_chip': ...,
            'activation_memory_per_chip': ...,
            'kv_cache_per_chip': ...,  # if applicable
        }
        
        print(f"  FLOPs/chip: {metrics.flops_per_package:,} (expected: {expected['flops_per_chip']:,})")
        print(f"  Weight/chip: {metrics.weight_memory_per_package:,} (expected: {expected['weight_memory_per_chip']:,})")
        print(f"  Activation/chip: {metrics.activation_memory_per_package:,} (expected: {expected['activation_memory_per_chip']:,})")
        
        if (metrics.flops_per_package == expected['flops_per_chip'] and
            metrics.weight_memory_per_package == expected['weight_memory_per_chip'] and
            metrics.activation_memory_per_package == expected['activation_memory_per_chip']):
            print("  ✓ PASSED")
            tests_passed += 1
        else:
            print("  ✗ FAILED")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        tests_failed += 1
    
    # ... repeat for all tests in the spec
    
    print(f"\n--- {LayerType} Tests Summary: {tests_passed} passed, {tests_failed} failed ---")
    return tests_passed, tests_failed
```

### 5.2 Add to Main Test Runner

At the bottom of `tests/run_tests.py`, add your test function to `main()`:

```python
def main():
    # ... existing tests
    
    # New layer tests
    {layer}_passed, {layer}_failed = test_{layer_type}_layers()
    
    # Update totals
    total_passed += {layer}_passed
    total_failed += {layer}_failed
```

---

## Section 6: Run Tests and Debug

### 6.1 Run Tests

```bash
cd /path/to/Mantile
python tests/run_tests.py
```

### 6.2 Debug Failures

When a test fails, systematically check:

1. **FLOPs mismatch**:
   - Print intermediate values in `compute_flops()`
   - Compare step-by-step with test spec
   - Check parallelism divisors

2. **Memory mismatch**:
   - Verify tensor shapes match spec
   - Check bytes_per_element calculation
   - Verify parallelism sharding

3. **Common issues**:
   - Integer vs float division
   - Off-by-one in sequence lengths
   - Missing parallelism handling

### 6.3 Document Discrepancies

If the test spec appears wrong:

```markdown
## Discrepancy Report

### Test: {TEST_ID}

**Expected (from spec)**: {value}
**Actual (from implementation)**: {value}

**Analysis**:
{Explain why there's a difference}

**Recommendation**:
- [ ] Fix implementation
- [ ] Fix test spec (with justification)
```

Present discrepancies to the user for resolution before proceeding.

### 6.4 Validate Test Specifications (Critical!)

**Do NOT assume test specifications are 100% correct.** Test specs are written by humans and can contain errors. Always verify:

1. **Formula consistency across tests**:
   - Check that similar tests use the same formulas
   - If Test A uses `(k*M/ep) × (d_ff/tp)` and Test B uses `(k*M)/(ep*tp) × d_ff`, that's a red flag
   - Parallelism should affect calculations consistently

2. **Physical plausibility**:
   - Does the formula make sense physically?
   - TP shards weights/activations along hidden dimensions, NOT tokens
   - EP shards experts across chips, distributing token-expert pairs
   - CP shards sequences, reducing tokens per chip

3. **Cross-reference with other tests**:
   - Compare formulas between single-chip and parallel tests
   - The parallel version should be a clean division of the single-chip version
   - Watch for inconsistent divisors

4. **When you find an inconsistency**:
   - Do NOT write workaround code to match a suspicious test
   - Flag it immediately to the user
   - Explain the inconsistency with specific calculations
   - Propose the correct formula with justification

**Example red flag (real case)**:
```
MOE-5 (EP=4, TP=2): routed_intermediate = (k*M/ep) × (d_ff/tp) = 128 × 2048
MOE-6 (EP=8, TP=4): routed_intermediate = (k*M)/(ep×tp) × d_ff_local = 16 × 1024

These use DIFFERENT formulas! MOE-5 divides tokens by EP only, MOE-6 divides by EP×TP.
The consistent formula should be: (k*M/ep) × (d_ff/tp) for both.
```

**Never implement special-case workarounds** just to make tests pass. If you need an `if` statement that changes the fundamental formula based on a config option (e.g., `if shared_experts > 0: use different formula`), that's a sign the test spec has a bug, not that you need special logic.

---

## Section 7: Update Model Config

### 7.1 Once All Tests Pass

Update the model config that required this layer:

```json
{
  "layer_types": [
    {
      "name": "...",
      "class": "{ClassName}",  // Use new class name
      "count": ...,
      "supported": true,       // Change to true
      "gap_report": null,      // Remove gap report reference
      "specs": {
        // ... specs unchanged
      }
    }
  ]
}
```

### 7.2 Validate the Config

```bash
python -m model_builder.utils validate backend/data/model_configs/{model_id}.json
```

### 7.3 Update Gap Report Status

If the gap is fully resolved, update `model_builder/gaps/README.md`:

```markdown
| Layer Type | Model | Severity | Status | File |
|------------|-------|----------|--------|------|
| {type} | {model} | {severity} | **Implemented** | [{file}]({file}) |
```

### 7.4 Archive Test Specifications

Once implementation is complete and all tests pass, move the sanitized test specifications from the gap report to the canonical test file in `/tests/`:

1. **Copy tests from gap file** (`model_builder/gaps/{layer_type}_gaps_{model}.md`)
2. **Append to canonical test file** (`tests/{layer_type}_tests.md`)
3. **Sanitize if needed**: Remove any model-specific notes, implementation hints, or debug comments
4. **Add section header**: Group the new tests under a descriptive heading

Example structure in `tests/attention_tests.md`:
```markdown
---

# Sliding Window Attention (SWA) Tests

These tests cover sliding window attention with:
- Configurable window size for limited attention span
- Attention sinks (preserved initial tokens)
- ...

---

# Test SWA-1 — ...
```

This ensures:
- All test specifications are in one authoritative location
- Gap files can be archived or removed once implemented
- Future developers can reference all tests in `/tests/`

---

## Section 8: Final Checklist

Before completing, verify:

**Implementation**
- [ ] Layer class created/modified in `backend/layers/`
- [ ] All abstract methods implemented
- [ ] `SUPPORTED_PARALLELISM` defined correctly
- [ ] Exported from `backend/layers/__init__.py`
- [ ] Docstrings complete with architecture description

**Tests**
- [ ] All tests from spec implemented in `tests/run_tests.py`
- [ ] All tests passing
- [ ] Test function added to `main()`

**Model Config**
- [ ] `supported: true` for the layer
- [ ] `class` updated to new class name
- [ ] `gap_report` removed or set to null
- [ ] Config validates successfully

**Documentation**
- [ ] Gap report status updated in `model_builder/gaps/README.md`
- [ ] Test specs archived to `tests/{layer_type}_tests.md`

---

## Appendix A: Example Layer Variants

### A.1 Sliding Window Attention (SWA)

Extends `GroupedQueryAttentionLayer`:

```python
class SlidingWindowAttentionLayer(GroupedQueryAttentionLayer):
    """GQA with sliding window - limits KV cache to window_size tokens."""
    
    def __init__(self, ..., sliding_window: int, ...):
        self.sliding_window = sliding_window
        super().__init__(...)
    
    def compute_kv_cache(self, batch_size: int, seq_len: int) -> int:
        # KV cache limited to sliding_window tokens
        cache_len = min(seq_len, self.sliding_window)
        # ... rest of calculation
```

### A.2 Mixture of Experts (MoE)

New layer type extending `Layer`:

```python
class MoELayer(Layer):
    """Mixture of Experts - routes tokens to top_k experts."""
    
    SUPPORTED_PARALLELISM = {"tensor_parallel", "expert_parallel"}
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        dtype: DataType | str = "bf16",
        parallelism: Optional[dict] = None
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        # Router params + expert params
        self.router_params = hidden_size * num_experts
        self.expert_params = num_experts * 3 * hidden_size * intermediate_size
        self.param_count = self.router_params + self.expert_params
        super().__init__(layer_idx, dtype, parallelism)
```

### A.3 Adding Biases to Existing Layer

Modify existing class:

```python
class GroupedQueryAttentionLayer(Layer):
    def __init__(
        self,
        ...,
        has_bias: bool = False,  # New parameter
        ...
    ):
        self.has_bias = has_bias
        # Adjust param count if biases present
        if has_bias:
            self.param_count += self.hidden_size + ...  # bias terms
```

---

## Appendix B: Common Mistakes

1. **Forgetting to handle decode phase differently**
   - Decode processes 1 token, not seq_len tokens
   - KV cache grows by 1 each step

2. **Wrong parallelism divisors**
   - TP divides by tp for weights AND some activations
   - SP divides sequence length, not hidden size
   - EP divides experts, not hidden size

3. **Missing communication**
   - TP requires all-reduce on outputs
   - EP requires all-to-all for token routing
   - CP requires softmax stats reduction

4. **Integer overflow**
   - Use `int()` wrapper on final results
   - Be careful with large models (100B+ params)

5. **Not exporting the layer**
   - Must add to `__init__.py` or tests won't find it
