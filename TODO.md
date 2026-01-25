# Mantile TODO

This document consolidates all TODOs across the Mantile codebase, organized by priority and category.

---

## High Priority

### Validation & Calibration

#### Calibrate MFU from Benchmark Data
**Location**: Roadmap item  
**Description**: Calibrate model FLOPs utilization (MFU) based on reconciliation with InferenceMAX benchmark data.  
**Current State**: Reconciliation infrastructure complete, predictions vs actuals comparison working  
**Next Steps**:
- Analyze error patterns across different configurations
- Extract MFU calibration factors from throughput comparisons
- Apply calibrated MFU to model predictions

**Code Reference**: `backend/main.py:407`
```python
# MFU placeholder (TODO: compute at layer level)
```

#### Expand to Additional Benchmark Sources
**Location**: Roadmap item  
**Description**: Expand reconciliation to additional benchmark sources beyond InferenceMAX  
**Current State**: InferenceMAX extraction and comparison working  
**Target Sources**:
- vLLM official benchmarks
- TensorRT-LLM performance data
- MLPerf inference results

---

## Medium Priority

### Layer Architecture Improvements

#### Sophisticated Kernel Count Modeling
**Location**: `backend/layers/base.py:146`  
**Description**: Make kernel count more sophisticated to better model GPU execution patterns  
**Current State**: Using conservative static estimates per layer type  
**Improvements Needed**:
- Different fusion strategies (e.g., flash attention vs naive)
- Hardware-specific fusion capabilities (some GPUs fuse better than others)
- Batch size effects (some kernels only launch once regardless of batch)
- Dynamic fusion decisions based on input shapes

**Code Reference**:
```python
# TODO: Make kernel count more sophisticated - account for:
# - Different fusion strategies (e.g., flash attention vs naive)
# - Hardware-specific fusion (some GPUs fuse better than others)
# - Batch size effects (some kernels only launch once regardless of batch)
# - Dynamic fusion decisions based on input shapes
# For now, use a conservative static estimate per layer type
default_kernel_count: int = 0
```

#### Pipeline Parallelism Modeling
**Location**: `backend/layers/mlp.py:74`  
**Description**: Model pipeline-parallel stages explicitly with per-stage metrics  
**Current State**: Only tensor parallelism (TP) modeled; PP is TODO  
**Implementation Plan**:
- Add pipeline stage assignment to layer configs
- Model inter-stage communication and bubble overhead
- Track per-stage timing and memory separately

**Code Reference**:
```python
def _get_num_packages(self) -> int:
    """Number of chips used for this layer's shard. TP modeled; PP TODO."""
    # TODO: Model pipeline-parallel stages explicitly with per-stage metrics
    # Note: Sequence parallelism typically shares devices with TP (doesn't add chips)
    tp = self.parallelism.get("tensor_parallel", 1)
    return tp
```

#### Complex Memory Hierarchies
**Location**: `backend/layers/base.py:371`  
**Description**: Add support for complex memory hierarchies (e.g., TPUs)  
**Current State**: Defaulting to HBM for all memory operations  
**Requirements**:
- Specify where weights live vs where KV cache lives
- Multi-tier memory (SRAM, HBM, DRAM)
- Memory hierarchy-aware bandwidth calculations
- Support for disaggregated memory architectures

**Code Reference**:
```python
# TODO: Future support for complex memory hierarchies (e.g., TPUs)
# Will need to specify where weights live vs where KV cache lives
# For now, default to HBM for all memory operations
hbm_memory = next(
    (m for m in hardware['memory_per_package'] if 'HBM' in m['type']),
    hardware['memory_per_package'][0]
)
```

### Collective Operations & Networking

#### Model Specific Collective Operations
**Location**: `backend/layers/__init__.py:8`  
**Description**: Model specific collective ops (All-Reduce, All-Gather, Reduce-Scatter) with hardware-aware performance  
**Current State**: Using generic communication bytes/time  
**Implementation Needed**:
- Different hardware has different performance characteristics for each collective
- Model NVLink vs PCIe vs inter-node bandwidth differences
- Consider message size effects on collective performance

#### Hardware Topology Modeling
**Location**: `backend/layers/__init__.py:8`  
**Description**: Implement detailed network topology modeling  
**Components**:
- NVLink vs PCIe vs inter-node bandwidth
- Switch fabric topology
- NUMA effects
- Multi-tier memory hierarchy (HBM, DRAM, NVMe)

---

## Low Priority

### UI/UX Improvements

#### Implicit Layer/System View Switching
**Location**: `frontend/src/App.jsx`  
**Description**: Replace tab-based switching with implicit mode detection based on layer selection  
**Current State**: Users manually switch between "System" and "Layer" tabs  
**Proposed Behavior**:
- Default view: System metrics (no layers selected)
- Automatic switch: When any layer is selected, automatically show Layer metrics view
- Automatic return: When all layers are deselected, return to System metrics view
- Visual clarity: Title changes from "System Metrics" to "Layer Metrics" with appropriate icon (Server vs Layers)
**Benefits**:
- More intuitive workflow - mode follows selection state
- Reduces cognitive load - one less thing to remember to switch
- Cleaner UI - removes tab navigation element

### Layer-Level Metrics Dashboard
**Location**: Roadmap item  
**Description**: Add detailed per-layer metrics view showing compute, memory, and communication breakdown  
**Features**:
- Visualize bottlenecks at the layer level (attention vs MLP compute/memory trade-offs)
- Support layer-by-layer profiling to identify optimization opportunities
- Interactive layer comparison across configurations

### Agentic IR Builder
**Location**: Roadmap item  
**Description**: Make IR builder autonomous to automatically infer model architectures  
**Features**:
- Automatically infer model architectures from HuggingFace configs
- Support automatic layer detection and parallelism strategy suggestions
- Enable AI-assisted model configuration generation from model cards
- Implement automated testing for newly added models

### Contribution Automation
**Location**: Roadmap item  
**Description**: Create automated workflows for validating contributions  
**Components**:
- Automated workflows for validating new hardware configurations
- CI/CD pipelines for testing model configurations against known benchmarks
- Enable community contributions through automated validation and testing
- Develop tooling for semi-automated addition of new hardware accelerators

---

## Data Collection (Historical)

### Extract October/November 2025 InferenceMAX Data
**Location**: `reconcile/sources/inferencemax/TODO.md`  
**Priority**: Low-Medium  
**Description**: Extract historical benchmark data from pre-aggregated format runs  
**Current State**: Script extracts aggregated data from December 9, 2025 onwards  
**Date Gap**: October 24 - December 9, 2025  

#### Background
InferenceMAX launched October 9, 2025, but the aggregated benchmark format (`results_bmk` artifact with `agg_bmk.json`) only started on December 9, 2025. Earlier benchmarks exist in a different, non-aggregated format where each configuration is stored as a separate artifact.

#### Workflows Available
1. **Full Sweep Scheduler - 1k1k**
   - Workflow ID: 200724779
   - Date Range: October 24 - October 31, 2025
   - Path: `.github/workflows/1k1k-sweep.yml`

2. **Test Sweep**
   - Workflow ID: 195698854
   - Date Range: October 7 - October 24, 2025
   - Path: `.github/workflows/test.yml`

#### Artifact Format
Each benchmark is a separate artifact with naming pattern:
```
{model}_{isl}{osl}_{precision}_{framework}_tp{tp}_ep{ep}_dpa_{dpa}_conc{conc}_{gpu}_{index}
```

Example: `gptoss_1k1k_fp4_vllm_tp8_ep1_dpa_false_conc8_h100-cw_1`

#### Implementation Plan
1. **Phase 1: Investigation** (30 min)
   - Download 3-5 sample artifacts from different runs
   - Examine internal format and available metrics
   - Verify it matches aggregated format

2. **Phase 2: Parser** (1-2 hours)
   - Create `parse_individual_artifact()` function
   - Extract config from filename
   - Parse benchmark results from file content
   - Convert to standard CSV format

3. **Phase 3: Workflow Integration** (1 hour)
   - Add `--workflow` parameter to script
   - Support both aggregated and individual artifact formats
   - Add `--date-range` to target specific months

4. **Phase 4: Testing** (30 min)
   - Extract October data and verify metrics
   - Merge with December data and validate

#### Expected Outcome
- Add ~50-100 additional configurations from October-November 2025
- Extend date range back to October 24, 2025
- More comprehensive GPU coverage from early benchmarks

#### Notes
- Current December+ data (203-235 configs) is sufficient for initial reconciliation
- Historical data nice-to-have for trend analysis
- GitHub Actions artifacts retained for 90 days by default (may be expired)

---

## Completed

### Reconciliation Infrastructure
- ✅ Build reconciliation infrastructure for comparing predictions with benchmarks
- ✅ Extract and standardize data from InferenceMAX
- ✅ Automated batch prediction and comparison workflows
- ✅ Model-agnostic utilities with CLI parameters
- ✅ Eliminated DRY violations in reconciliation code
- ✅ Created shared constants module for config columns and metrics
- ✅ Updated documentation for reconciliation workflow

---

## Contributing

When adding items to this TODO list:
1. Use clear, descriptive headers
2. Include file path and line number references
3. Provide context about current state
4. Outline implementation plan if available
5. Categorize by priority (High/Medium/Low)
6. Mark as completed when done

**To search for TODOs in code:**
```bash
# Find all TODO comments
grep -r "TODO\|todo" --include="*.py" --include="*.md" .

# Find TODO comments in Python files only
grep -r "# TODO" --include="*.py" .
```
