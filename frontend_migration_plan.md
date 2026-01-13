# Frontend Migration Plan: Two-Phase Architecture

## Executive Summary
The backend has been refactored to explicitly separate prefill and decode phases. Each phase now has its own configuration, metrics, and layer settings. The frontend must be updated to support configuring and displaying metrics for both phases independently.

---

## 1. Backend API Changes

### Old API (Single Phase)
```typescript
POST /api/system-metrics
{
  model_config: string,
  hardware_config: string,
  seq_len: number,
  output_seq: number,
  batch_size: number,
  phase: "prefill" | "decode",
  dtype: string,
  layer_configs: Array<{
    layer_index: number,
    tp_degree: number,
    cp_degree: number,
    sp_degree: number
  }>
}
```

### New API (Dual Phase)
```typescript
POST /api/system-metrics
{
  prefill_req: {
    model_config: string,
    hardware_config: string,
    seq_len: number,        // Input sequence length
    batch_size: number,
    dtype: string,
    layer_configs: Array<{...}>
  },
  decode_req: {
    model_config: string,
    hardware_config: string,
    seq_len: number,        // Output sequence length
    batch_size: number,
    dtype: string,
    layer_configs: Array<{...}>
  }
}
```

**Key Changes:**
- No more `phase` field - phases are separated at the request level
- No more `output_seq` field - decode phase uses its own `seq_len`
- Each phase has independent layer configurations
- Model and hardware configs can theoretically differ per phase (though typically same)

---

## 2. UI/UX Design Changes

### Current Layout
```
┌────────────────────────────────────────────────┐
│ [Single Config Panel]                          │
│  - Model: llama_3.3_70b                        │
│  - Hardware: nvidia_gb200                      │
│  - Phase: [Prefill ▼]                          │
│  - Seq Len: 8192                               │
│  - Output Seq: 256                             │
│  - Batch: 1                                    │
│  - Dtype: bf16                                 │
│                                                │
│ [Layer Configs Table]                          │
│  Layer | TP | CP | SP                          │
│  ─────┼────┼────┼────                          │
│    0   │ 2  │ 4  │ 1                           │
│   ...                                          │
└────────────────────────────────────────────────┘
```

### Target Layout
```
┌────────────────────────────────────────────────┐
│ [Shared Config]                                │
│  - Model: llama_3.3_70b                        │
│  - Dtype: bf16                                 │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ PREFILL PHASE                                  │
│  - Hardware: nvidia_gb200                      │
│  - Batch Size: 1                               │
│  - Seq Len: 8192 (input prompt length)         │
│                                                │
│ [Layer Configs Table]                          │
│  Layer | TP | CP | SP                          │
│  ──────┼────┼────┼────                         │
│    0   │ 2  │ 4  │ 1                           │
│   ...                                          │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ DECODE PHASE                                   │
│  - Hardware: nvidia_gb200                      │
│  - Batch Size: 1                               │
│  - Seq Len: 256 (output generation length)     │
│                                                │
│ [Layer Configs Table]                          │
│  Layer | TP | CP | SP                          │
│  ──────┼────┼────┼────                         │
│    0   │ 8  │ 1  │ 1                           │
│   ...                                          │
└────────────────────────────────────────────────┘
```

**Design Principles:**
1. **Minimal Shared Configuration**: Only model and dtype are shared to prevent unexpected metric calculations
2. **Phase-Specific Resources**: Hardware and batch size are configured independently per phase
3. **Phase Separation**: Clear visual distinction between prefill and decode configurations
4. **Independent Layer Configs**: Each phase can have different parallelism strategies per layer
5. **Contextual Labeling**: Make it clear what `seq_len` means for each phase
6. **No Default Values**: All fields must be explicitly set to avoid silent computation of unexpected metrics

---

## 3. Component Architecture Changes

### 3.1 State Management

**Current State Structure:**
```typescript
{
  model: string,
  hardware: string,
  phase: "prefill" | "decode",
  seqLen: number,
  outputSeq: number,
  batchSize: number,
  dtype: string,
  layerConfigs: Array<LayerConfig>
}
```

**Target State Structure:**
```typescript
{
  // Shared config (minimal to prevent unexpected computations)
  shared: {
    model: string,
    dtype: string
  },
  
  // Prefill phase
  prefill: {
    hardware: string,
    batchSize: number,
    seqLen: number,
    layerConfigs: Array<LayerConfig>
  },
  
  // Decode phase
  decode: {
    hardware: string,
    batchSize: number,
    seqLen: number,
    layerConfigs: Array<LayerConfig>
  }
}
```

### 3.2 Component Breakdown

**New/Modified Components:**
1. **`SharedConfigPanel`**: Model and dtype selectors only (minimal shared config)
2. **`PhaseConfigPanel`**: Reusable component for prefill/decode configuration
   - Props: `phase` ("prefill" | "decode"), `hardware`, `batchSize`, `seqLen`, `layerConfigs`, `onUpdate`
   - Renders: Phase header, hardware selector, batch size input, seq len input, layer configs table
3. **`LayerConfigTable`**: Existing component (reusable for both phases)
4. **`MetricsDisplay`**: Updated to show prefill metrics, decode metrics, AND system-level aggregated metrics
5. **`ComparePanel`**: Duplicate the entire structure for comparison

### 3.3 API Integration

**Request Builder Function:**
```typescript
function buildSystemMetricsRequest(config) {
  return {
    prefill_req: {
      model_config: config.shared.model,
      hardware_config: config.prefill.hardware,
      seq_len: config.prefill.seqLen,
      batch_size: config.prefill.batchSize,
      dtype: config.shared.dtype,
      layer_configs: config.prefill.layerConfigs
    },
    decode_req: {
      model_config: config.shared.model,
      hardware_config: config.decode.hardware,
      seq_len: config.decode.seqLen,
      batch_size: config.decode.batchSize,
      dtype: config.shared.dtype,
      layer_configs: config.decode.layerConfigs
    }
  };
}
```

---

## 4. Implementation Phases

### Phase 1: State & Data Flow (Foundation)
**Goal:** Update internal state management without changing UI

**Tasks:**
- [ ] Refactor state structure to separate shared/prefill/decode
- [ ] Update state initialization to populate both phase configs
- [ ] Create utility functions to build API requests
- [ ] Update API call function signature
- [ ] Test API integration with new request format

**Files to Modify:**
- `App.jsx`: State structure, API calls

**Validation:**
- Backend responds successfully to new request format
- Console logs show correct request structure

---

### Phase 2: UI Components (Visual Update)
**Goal:** Update UI to show both phases

**Tasks:**
- [ ] Create `SharedConfigPanel` component (model and dtype only)
- [ ] Create `PhaseConfigPanel` component (or refactor existing config panel)
  - Include hardware selector per phase
  - Include batch size input per phase
  - Include seq len input
  - Include layer configs table
- [ ] Update main layout to show shared + prefill + decode sections
- [ ] Add visual indicators (icons/colors) for phase distinction
- [ ] Update metrics display to show prefill metrics, decode metrics, AND system-level metrics
- [ ] Ensure no default values are set - all inputs must be explicitly filled by user

**Files to Modify:**
- `App.jsx`: Layout restructure
- New files: `SharedConfigPanel.jsx`, `PhaseConfigPanel.jsx` (or refactor existing)

**Validation:**
- UI shows all configuration fields without defaults
- Changes to shared config (model/dtype) affect both phases
- Changes to phase-specific config (hardware/batch/seq/layers) only affect that phase
- System metrics are prominently displayed alongside phase-specific metrics

---

### Phase 3: Layer Configuration (Complex Interactions)
**Goal:** Enable independent layer configuration per phase

**Tasks:**
- [ ] Update layer config table to work with phase-specific data
- [ ] Implement "Copy from Prefill" button for decode phase
- [ ] Add validation for layer configs (parallelism constraints)
- [ ] Update bulk operations (apply to all layers) per phase
- [ ] Handle model changes (reset both phase configs)

**Files to Modify:**
- Layer configuration components
- Validation logic

**Validation:**
- Can set different TP/CP/SP per layer per phase
- Bulk operations work correctly
- Validation errors show appropriately

---

### Phase 4: Compare Panel (Duplication)
**Goal:** Extend compare functionality to support two-phase architecture

**Tasks:**
- [ ] Duplicate shared/prefill/decode structure for compare panel
- [ ] Update comparison display to show differences per phase
- [ ] Ensure both configurations can be computed independently
- [ ] Update diff visualization for phase-specific configs

**Files to Modify:**
- Compare panel components
- Diff/comparison utilities

**Validation:**
- Can compare two configurations with different phase settings
- Metrics show for both phases in both configs
- Diff highlights work for phase-specific changes

---

### Phase 5: Polish & Edge Cases
**Goal:** Handle edge cases and improve UX

**Tasks:**
- [ ] Add loading states per phase
- [ ] Handle errors gracefully (one phase fails, other succeeds)
- [ ] Add tooltips explaining prefill vs decode
- [ ] Implement preset configurations (e.g., "Default", "Memory Optimized")
- [ ] Add export/import configuration feature
- [ ] Responsive design testing

**Files to Modify:**
- All components (polish pass)
- New utility components (tooltips, presets)

**Validation:**
- Error handling works correctly
- UI is intuitive and self-explanatory
- Mobile/tablet layouts work

---

## 5. Data Flow Diagram

```
User Input (Shared)          User Input (Prefill)              User Input (Decode)
   │                              │                                  │
   ├─ Model                       ├─ Hardware: nvidia_gb200          ├─ Hardware: nvidia_gb200
   └─ Dtype                       ├─ Batch: 1                        ├─ Batch: 1
                                  ├─ Seq Len: 8192                   ├─ Seq Len: 256
                                  └─ Layer Configs                   └─ Layer Configs
                                         │                                  │
                                         │                                  │
      ┌──────────────────────────────────┴──────────────────────────────────┘
      │
      ▼
buildSystemMetricsRequest()
      │
      ▼
┌──────────────────────────────┐
│  POST /api/system-metrics    │
│  {                           │
│    prefill_req: {            │
│      model_config,           │
│      hardware_config,        │  ⚠️  No defaults - all fields
│      batch_size,             │      must be explicitly set
│      seq_len,                │      by user to prevent
│      dtype,                  │      unexpected computations
│      layer_configs           │
│    },                        │
│    decode_req: {             │
│      model_config,           │
│      hardware_config,        │
│      batch_size,             │
│      seq_len,                │
│      dtype,                  │
│      layer_configs           │
│    }                         │
│  }                           │
└──────────────────────────────┘
      │
      ▼
┌──────────────────────────────┐
│  Backend Response:           │
│  {                           │
│    prefill: {                │
│      latency_ms,             │
│      memory_gb,              │
│      flops,                  │
│      communication_time_ms,  │
│      ...                     │
│    },                        │
│    decode: {                 │
│      latency_ms,             │
│      memory_gb,              │
│      flops,                  │
│      communication_time_ms,  │
│      ...                     │
│    },                        │
│    system: {                 │
│      total_time_ms,          │  ← System-level metrics
│      throughput_tokens_s,    │    (aggregated from both phases)
│      total_memory_gb,        │
│      utilization_percent,    │
│      ...                     │
│    }                         │
│  }                           │
└──────────────────────────────┘
      │
      ▼
┌─────┴────────────────┐
│                      │
▼                      ▼                      ▼
Prefill Metrics    Decode Metrics      System Metrics
Display            Display              Display
- Latency          - Latency            - Total Time
- Memory           - Memory             - Throughput
- FLOPs            - FLOPs              - Total Memory
- Comm Time        - Comm Time          - Utilization
```

---

## 6. Migration Checklist

### Pre-Implementation
- [x] Backend refactoring complete
- [x] Backend API tested and validated
- [ ] Frontend team review of this plan
- [ ] Agree on UI/UX design
- [ ] Identify breaking changes for existing users

### Implementation
- [x] Phase 1: State & Data Flow ✅ COMPLETED (2026-01-13)
  - State structure refactored (shared/prefill/decode)
  - API request builder implemented
  - New two-phase API integration complete
  - Backend integration tested and working
- [x] Phase 2: UI Components ✅ COMPLETED (2026-01-13)
  - SharedConfigPanel component created (model + dtype)
  - PhaseConfigPanel component created (hardware + batch + seq + layers)
  - MetricsDisplay component shows system + prefill + decode metrics
  - LayerConfigCard component extracted and reusable
  - Visual indicators added (🔹 prefill blue, 🔸 decode green)
  - No default values - all fields require explicit user input
  - "Copy from Prefill" button for decode phase
- [ ] Phase 3: Layer Configuration (IN PROGRESS)
- [ ] Phase 4: Compare Panel
- [ ] Phase 5: Polish & Edge Cases

### Testing
- [ ] Unit tests for state management
- [ ] Integration tests for API calls
- [ ] UI tests for component rendering
- [ ] E2E tests for complete user flows
- [ ] Cross-browser testing
- [ ] Performance testing (large layer counts)

### Deployment
- [ ] Update documentation
- [ ] Create migration guide for users
- [ ] Deploy to staging
- [ ] User acceptance testing
- [ ] Deploy to production
- [ ] Monitor for issues

---

## 7. Open Questions & Considerations

### Technical Decisions
1. **Should model/hardware/dtype/batch be truly shared, or configurable per phase?**
   - ✅ **DECISION**: Hardware and batch are now configurable per phase
   - Rationale: Different phases may benefit from different hardware configurations or batch sizes
   - Model and dtype remain shared (same model architecture, same precision)
   - **Implementation**: Hardware and batch selectors appear in each PhaseConfigPanel

2. **Should we use default values for configuration fields?**
   - ✅ **DECISION**: No defaults - all fields must be explicitly set by user
   - Rationale: Prevent silent computation of unexpected metrics
   - UI should clearly indicate which fields are required
   - **Implementation**: Disable "Compute Metrics" button until all fields are filled

3. **How to handle layer config initialization for new models?**
   - Copy prefill configs to decode by default?
   - Use different defaults (e.g., higher TP for decode)?
   - **Recommendation**: Provide "Copy from Prefill to Decode" button, but no automatic copying

4. **Should we support "phase-only" mode (compute just prefill or just decode)?**
   - Current: Always compute both phases
   - Alternative: Checkboxes to enable/disable phases
   - **Recommendation**: Always compute both (backend expects both)

### UX Considerations
1. **Page layout: Vertical stacking vs. side-by-side?**
   - Vertical: Easier to see all configs, but requires scrolling
   - Side-by-side: More compact, but harder on small screens
   - **Recommendation**: Vertical on mobile, side-by-side on desktop (responsive)

2. **Required field indication:**
   - All fields must be explicitly set by user (no defaults)
   - Use clear visual indicators (asterisks, borders, disabled compute button)
   - Show helpful placeholder text (e.g., "Select hardware...")
   - **Recommendation**: Red border for empty required fields, disable compute until all filled

3. **Visual distinction between phases:**
   - Color coding (blue for prefill, green for decode)?
   - Icons (🔹 prefill, 🔸 decode)?
   - **Recommendation**: Both (color + icon for accessibility)

4. **System metrics prominence:**
   - System-level metrics should be highly visible (larger text, highlighted section)
   - Show relationship between phase metrics and system metrics
   - **Recommendation**: Dedicated "System Metrics" panel above or between phase panels

### Future Enhancements
1. **Phase ratio analysis**: Show prefill vs decode time proportion
2. **Optimization suggestions**: "Prefill is bottleneck, try increasing CP"
3. **Configuration templates**: Save/load common configurations
4. **Export to deployment scripts**: Generate kubernetes/slurm configs

---

## 8. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Breaking existing frontend | High | High | Implement feature flag, gradual rollout |
| Complex state management bugs | Medium | Medium | Thorough testing, TypeScript validation |
| UX confusion (two configs) | Medium | Medium | Clear labeling, tooltips, documentation |
| Performance (double layer tables) | Low | Low | Virtual scrolling, lazy rendering |
| API compatibility issues | High | Low | Backend already stable, well-tested |

---

## 9. Success Criteria

### Functional Requirements
- [ ] Users can configure prefill and decode phases independently
- [ ] API requests are correctly formatted and accepted by backend
- [ ] Metrics display correctly for both phases
- [ ] Compare panel works with two-phase configs
- [ ] No regressions in existing functionality

### Non-Functional Requirements
- [ ] Page load time < 2s
- [ ] Smooth scrolling with 100+ layers
- [ ] Mobile-responsive design
- [ ] WCAG 2.1 AA accessibility compliance
- [ ] No console errors or warnings

### User Satisfaction
- [ ] Users understand the difference between prefill and decode
- [ ] Configuration workflow is intuitive
- [ ] Feature adds value to existing workflows
- [ ] Documentation is clear and helpful

---

## 10. Next Steps

1. **Review this plan** with team and stakeholders
2. **Create UI mockups** (Figma/wireframes) for final approval
3. **Set up feature branch** (`feature/two-phase-frontend`)
4. **Begin Phase 1 implementation** (state management refactor)
5. **Iterate with regular demos** (show progress after each phase)

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-12  
**Author:** GitHub Copilot  
**Status:** Draft - Awaiting Review
