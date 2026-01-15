# Comparison Systems Feature - Implementation Plan

## Overview
Add ability to compare two different system configurations side-by-side with overlaid metrics visualization.

## Design Decisions
1. **Metrics Display**: Overlay both systems in same visualization (green vs orange bars/values)
2. **Copy Behavior**: "Copy from Prefill" in comparison decode copies from comparison prefill
3. **Orange Accent**: `#f96c56` for all comparison system elements

## Implementation Phases

### **Phase 1: Add Compare Button to Header**
- **Location**: Header bar, right side (after model dropdown)
- **Text**: "Compare Systems" (toggle to "Exit Comparison" when active)
- **State**: Add `comparisonMode` boolean (default: false)
- **Styling**: Match header styling, use `ml-auto` to push right

### **Phase 2: State Management**
Add new state variables:
- `comparisonMode`: boolean to control visibility
- `comparisonConfig`: object containing:
  - `model`: selected model for comparison
  - `prefill`: prefill phase config
  - `decode`: decode phase config
  - `layerConfigs`: layer-specific configurations
- `comparisonMetrics`: stores computed metrics for comparison system
- Initialize comparison config with same defaults as primary config

### **Phase 3: Layout Adjustments**
- **Current structure**: `<div className="flex h-screen">` with left sidebars + main content
- **New structure** (when `comparisonMode === true`):
  - Left sidebars: Primary Prefill (300px) + Primary Decode (300px)
  - Main content: Flex-grows to fill space
  - Right sidebars: Comparison Prefill (300px) + Comparison Decode (300px)
- Use conditional rendering: `{comparisonMode && <RightSidebars />}`

### **Phase 4: Right Sidebars Implementation**

#### Comparison Prefill Sidebar
- **Title**: "Comparison - Prefill"
- **Width**: 300px
- **Accent color**: #f96c56 (orange)
- **Components**:
  - Shared config panel (model dropdown, hardware config if needed)
  - Layer configuration cards with:
    - Context Parallel slider
    - Tensor Parallel slider
    - Data Type dropdown
  - Orange styling for all interactive elements

#### Comparison Decode Sidebar
- **Title**: "Comparison - Decode"
- **Width**: 300px
- **Accent color**: #f96c56 (orange)
- **Components**:
  - "Copy from Prefill" button (copies from comparison prefill, NOT primary)
  - Layer configuration cards (same as comparison prefill)
  - Orange styling throughout

### **Phase 5: Metrics Display - Overlay Mode**
Update MetricsDisplay component to handle dual datasets:

#### Props
- `metrics`: primary system metrics (existing)
- `comparisonMetrics`: comparison system metrics (new)

#### Visualization Strategy
- **System Metrics Boxes**: Show two values side-by-side or stacked
  - Primary: Green accent (#10B981) + label "Primary"
  - Comparison: Orange accent (#f96c56) + label "Comparison"
  
- **Bar Charts**: Render 2 bars per metric
  - Green bar: Primary system
  - Orange bar: Comparison system
  
- **Text Metrics**: Display both values with color-coded labels
  - Example: "Primary: 45.2ms | Comparison: 52.1ms"

#### Layout
- Keep single MetricsDisplay component in main content area
- When comparison mode inactive: show only primary metrics (current behavior)
- When comparison mode active: overlay both datasets in same visualization

### **Phase 6: API Calls & Data Flow**
- **Primary system**: Existing flow (compute metrics on config change)
- **Comparison system**: 
  - When `comparisonMode === true`, make second API call
  - Send `comparisonConfig` to backend
  - Store results in `comparisonMetrics` state
  - Trigger on: comparison config changes, comparison mode toggle

#### API Request Structure
```javascript
// Primary (existing)
buildSystemMetricsRequest(config)

// Comparison (new)
buildSystemMetricsRequest(comparisonConfig)
```

### **Phase 7: Color Scheme**

#### Primary System (Left)
- **Sidebar borders**: Blue/Teal theme
- **Sliders**: #29AF83 (teal)
- **Metrics data**: #10B981 (green)
- **Icons/accents**: #3B82F6 (blue)

#### Comparison System (Right)
- **All elements**: #f96c56 (orange)
- **Sidebar borders**: Orange left border
- **Sliders**: Orange
- **Metrics data**: Orange
- **Icons/accents**: Orange

#### Shared Elements
- **Card backgrounds**: #e7e3da (ivory) for both
- **Text colors**: #1F2937 (dark) for readability
- **Labels**: #6B7280 (gray)

## Execution Checklist

### Step 1: Header Button
- [ ] Add "Compare Systems" button to header
- [ ] Add `comparisonMode` state
- [ ] Implement toggle behavior
- [ ] Style button to match header

### Step 2: State Setup
- [ ] Add `comparisonConfig` state with structure matching primary config
- [ ] Add `comparisonMetrics` state
- [ ] Initialize comparison config with defaults

### Step 3: Right Sidebars
- [ ] Create comparison prefill sidebar with orange theme
- [ ] Create comparison decode sidebar with orange theme
- [ ] Add conditional rendering based on `comparisonMode`
- [ ] Wire up state handlers for comparison config

### Step 4: Copy from Prefill
- [ ] Implement copy handler for comparison decode
- [ ] Ensure it copies from comparison prefill (not primary)

### Step 5: API Integration
- [ ] Add second API call for comparison metrics
- [ ] Trigger on comparison config changes
- [ ] Store in `comparisonMetrics` state

### Step 6: Metrics Overlay
- [ ] Update MetricsDisplay to accept `comparisonMetrics` prop
- [ ] Implement side-by-side value display
- [ ] Add color coding (green vs orange)
- [ ] Add "Primary" / "Comparison" labels
- [ ] Ensure graceful handling when comparison metrics unavailable

### Step 7: Styling Polish
- [ ] Apply orange theme throughout comparison sidebars
- [ ] Add colored borders (blue left, orange right)
- [ ] Test responsive behavior with 4 sidebars
- [ ] Verify text readability

### Step 8: Testing
- [ ] Test toggle on/off behavior
- [ ] Verify independent config changes (primary vs comparison)
- [ ] Test copy from prefill in both systems
- [ ] Verify metrics compute correctly for both
- [ ] Test with different model selections
- [ ] Check layout on different screen sizes

## Technical Notes

### Component Structure
```
App.jsx
├── Header (with Compare button)
├── Primary Prefill Sidebar (300px)
├── Primary Decode Sidebar (300px)
├── Main Content
│   └── MetricsDisplay (overlay mode)
├── [Conditional] Comparison Prefill Sidebar (300px)
└── [Conditional] Comparison Decode Sidebar (300px)
```

### State Flow
```
User changes comparison config
  → comparisonConfig updated
  → Triggers useEffect
  → API call with comparison config
  → comparisonMetrics updated
  → MetricsDisplay re-renders with overlay
```

## Future Enhancements (Out of Scope)
- Save/load comparison configurations
- Export comparison results
- Diff view showing deltas between systems
- More than 2 systems comparison
- Comparison presets
