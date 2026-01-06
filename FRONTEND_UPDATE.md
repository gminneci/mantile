# Frontend Update for Stateless API

## Key Changes Required in App.jsx

### 1. Remove Model Loading

**Before:**
```javascript
const loadModelAndHardware = async () => {
  const response = await axios.post(`${API_URL}/config/load`, {
    model_id: modelId,
    hardware_config: hwPreset
  });
  const layersResponse = await axios.get(`${API_URL}/config/layers`);
  setLayers(layersResponse.data.layers);
};
```

**After:**
```javascript
// No loading needed! Just fetch layers when model/hardware changes
const fetchLayers = async (model_id) => {
  const response = await axios.get(`${API_URL}/api/layers`, {
    params: { model_id }
  });
  setLayers(response.data.layers);
};

useEffect(() => {
  if (modelId) {
    fetchLayers(modelId);
  }
}, [modelId]);
```

### 2. Update Layer Metrics Fetch

**In LayerConfigCard component:**

```javascript
const fetchMetrics = async () => {
  setLoadingMetrics(true);
  try {
    const response = await axios.post(`${API_URL}/config/layer-metrics`, {
      model_id: modelId,          // ADD
      hardware_config: hwPreset,   // ADD
      layer_type: layer.type,
      batch_size: batchSize,
      seq_length: seqLength,
      dtype: config.dtype || 'bf16',
      tensor_parallel: config.parallelism?.tensor_parallel || 1,
      context_parallel: config.parallelism?.context_parallel || 1,
      sequence_parallel: config.parallelism?.sequence_parallel || 1
    });
    setLayerMetrics(response.data);
  } catch (err) {
    console.error('Failed to fetch metrics:', err);
  } finally {
    setLoadingMetrics(false);
  }
};
```

### 3. Update System Metrics Computation

```javascript
const computeMetrics = async () => {
  setLoading(true);
  try {
    // Build layers object from layerConfigs
    const layersPayload = {};
    for (const [layerType, config] of Object.entries(layerConfigs)) {
      layersPayload[layerType] = {
        tensor_parallel: config.parallelism?.tensor_parallel || 1,
        context_parallel: config.parallelism?.context_parallel || 1,
        sequence_parallel: config.parallelism?.sequence_parallel || 1,
        dtype: config.dtype || 'bf16'
      };
    }
    
    const response = await axios.post(`${API_URL}/config/system-metrics`, {
      model_id: modelId,
      hardware_config: hwPreset,
      batch_size: batchSize,
      input_seq: seqLength,
      output_seq: 128,
      layers: layersPayload
    });
    
    setSystemMetrics(response.data);
  } catch (err) {
    console.error('Failed to compute metrics:', err);
    setError(err.response?.data?.detail || 'Failed to compute metrics');
  } finally {
    setLoading(false);
  }
};
```

### 4. Comparison Mode (Right Panel)

For config2, use the same approach but with config2's state:

```javascript
const computeConfig2Metrics = async () => {
  // Build layers payload from config2.layerConfigs
  const layersPayload = {};
  for (const [layerType, config] of Object.entries(config2.layerConfigs)) {
    layersPayload[layerType] = {
      tensor_parallel: config.parallelism?.tensor_parallel || 1,
      context_parallel: config.parallelism?.context_parallel || 1,
      sequence_parallel: config.parallelism?.sequence_parallel || 1,
      dtype: config.dtype || 'bf16'
    };
  }
  
  const response = await axios.post(`${API_URL}/config/system-metrics`, {
    model_id: config2.modelId,
    hardware_config: config2.hwPreset,
    batch_size: batchSize,
    input_seq: seqLength,
    output_seq: 128,
    layers: layersPayload
  });
  
  setConfig2(prev => ({
    ...prev,
    systemMetrics: response.data
  }));
};

// Watch config2 changes
useEffect(() => {
  if (config2.modelId) {
    fetchLayers(config2.modelId).then(layersData => {
      setConfig2(prev => ({ ...prev, layers: layersData.layers }));
    });
  }
}, [config2.modelId]);

useEffect(() => {
  if (comparisonMode && config2.layerConfigs && Object.keys(config2.layerConfigs).length > 0) {
    computeConfig2Metrics();
  }
}, [config2.layerConfigs, config2.modelId, config2.hwPreset, batchSize, seqLength]);
```

## Summary of Changes

1. **Remove**: `loadModelAndHardware()` function and its calls
2. **Add**: `fetchLayers()` function that uses `/api/layers?model_id=...`
3. **Update**: LayerConfigCard to include model_id and hardware_config in metrics requests
4. **Update**: System metrics to include full configuration (model_id, hardware_config, layers)
5. **Add**: Separate metrics computation for config2 in comparison mode
6. **Remove**: All references to `/config/load` and `/config/layer-parallelism` endpoints

## Testing

After implementing these changes:

1. Change model → should fetch layers automatically
2. Change parallelism → should recompute metrics with new config
3. Enable comparison mode → both configs should work independently
4. Change config2 model → should fetch its layers and recompute
