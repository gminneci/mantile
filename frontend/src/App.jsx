import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Server, Cpu, Box, Layers, CheckCircle, AlertCircle, Copy } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

// --- API Client ---
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// --- Components ---

function StatCard({ label, value, unit, icon: Icon, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="card metrics-card flex flex-col gap-2"
    >
      <div className="flex justify-between items-start">
        <span className="text-dark text-sm uppercase tracking-wider">{label}</span>
        {Icon && <Icon size={18} className="text-highlight-blue" style={{ opacity: 0.7 }} />}
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-3xl font-bold text-dark">{value}</span>
        {unit && <span className="text-dark text-sm" style={{ opacity: 0.85 }}>{unit}</span>}
      </div>
    </motion.div>
  );
}

function LayerConfigCard({ layer, config, onConfigChange, batchSize, seqLength, modelId, hwPreset }) {
  const [metrics, setMetrics] = useState(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  useEffect(() => {
    if (config && modelId && hwPreset) {
      fetchMetrics();
    }
  }, [config, batchSize, seqLength, modelId, hwPreset]);

  const fetchMetrics = async () => {
    if (!config || !modelId || !hwPreset) return;
    setLoadingMetrics(true);
    try {
      const response = await axios.post(`${API_URL}/config/layer-metrics`, {
        model_id: modelId,
        hardware_config: hwPreset,
        layer_type: layer.type,
        batch_size: batchSize,
        seq_length: seqLength,
        dtype: config.dtype || 'bf16',
        tensor_parallel: config.parallelism?.tensor_parallel || 1,
        context_parallel: config.parallelism?.context_parallel || 1,
        sequence_parallel: config.parallelism?.sequence_parallel || 1
      });
      setMetrics(response.data);
    } catch (err) {
      console.error('Failed to fetch layer metrics:', err);
      console.error('Request details:', {
        model_id: modelId,
        hardware_config: hwPreset,
        layer_type: layer.type,
        batch_size: batchSize,
        seq_length: seqLength,
        dtype: config.dtype || 'bf16',
        parallelism: config.parallelism
      });
      if (err.response) {
        console.error('Response status:', err.response.status);
        console.error('Response data:', err.response.data);
      }
    } finally {
      setLoadingMetrics(false);
    }
  };

  const availableStrategies = layer.available_parallelism || [];
  const maxChips = availableStrategies.length > 0 ? 8 : 1; // simplified for now

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="card"
    >
      <div className="flex justify-between items-start mb-4">
        <div>
          <h4 className="text-white font-semibold capitalize flex items-baseline gap-2">
            {layer.type} <span className="text-dim text-sm font-mono">({layer.count}x)</span>
          </h4>
        </div>
        {config && <CheckCircle size={20} className="text-accent" />}
      </div>

      {availableStrategies.length > 0 ? (
        <div className="flex flex-col gap-3 mt-4">
          {availableStrategies.map(strategy => (
            <div key={strategy}>
              <label className="text-dim text-xs uppercase tracking-wider">
                {strategy.replace('_', ' ')}
              </label>
              <div className="flex items-center gap-4 mt-2">
                <input
                  type="range"
                  min="1"
                  max={maxChips}
                  value={config?.parallelism[strategy] || 1}
                  onChange={(e) => onConfigChange(layer.type, strategy, parseInt(e.target.value))}
                  className="flex-1 w-full h-2 bg-surface rounded-lg appearance-none cursor-pointer accent-accent"
                />
                <span className="text-accent font-mono text-sm w-10 text-right">
                  {config?.parallelism[strategy] || 1}
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-dim text-sm italic">Replicated (no parallelism)</div>
      )}

      {/* Format Selection */}
      <div className="flex items-center gap-3 mt-4">
        <label className="text-dim text-xs uppercase tracking-wider whitespace-nowrap">Format</label>
        <select
          className="input-field text-xs dtype-select"
          style={{ padding: '0.25rem 0.5rem', minWidth: '72px', width: 'auto' }}
          value={config?.dtype || 'bf16'}
          onChange={(e) => onConfigChange(layer.type, 'dtype', e.target.value)}
        >
          {(layer.available_dtypes || ['fp32', 'fp16', 'bf16', 'fp8', 'int8']).map(dtype => (
            <option key={dtype} value={dtype}>{dtype.toUpperCase()}</option>
          ))}
        </select>
      </div>

      {/* Layer Metrics */}
      {metrics && (
        <div className="mt-5 pt-4">
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div>
              <span className="text-dim">Chips:</span>
              <span className="text-white ml-4 font-mono">{metrics.num_chips}</span>
            </div>
            <div>
              <span className="text-dim">Weight Mem:</span>
              <span className="text-white ml-4 font-mono">{metrics.memory.total_weights_gb.toFixed(2)} GB</span>
            </div>
            <div>
              <span className="text-dim">Act Mem:</span>
              <span className="text-white ml-4 font-mono">{metrics.memory.total_activation_gb.toFixed(2)} GB</span>
            </div>
            <div>
              <span className="text-dim">FLOPs:</span>
              <span className="text-white ml-4 font-mono">{metrics.compute.total_flops_tflops.toFixed(1)} TF</span>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}

function ConfigPanel({ title, modelId, hwPreset, layers, layerConfigs, onModelChange, onHwChange, onLayerConfigChange, batchSize, seqLength, onBatchChange, onSeqChange }) {
  return (
    <aside style={{
      width: '340px',
      minWidth: '340px',
      backgroundColor: '#1D2F61',
      padding: '32px',
      overflowY: 'auto',
      borderRight: '1px solid #29AF83',
      boxSizing: 'border-box'
    }}>
      <div className="flex items-center" style={{ gap: '12px', marginBottom: '24px' }}>
        <div className="w-8 h-8 rounded bg-accent flex items-center justify-center">
          <Activity size={20} className="text-bg" />
        </div>
        <h1 className="text-xl font-bold tracking-tight text-white">{title}</h1>
      </div>

      {/* Hardware Selection */}
      <div className="flex flex-col" style={{ gap: '16px', marginBottom: '24px' }}>
        <h3 className="text-accent font-semibold flex items-center" style={{ gap: '8px' }}>
          <Server size={18} /> Hardware
        </h3>
        <select
          className="input-field"
          value={hwPreset}
          onChange={(e) => onHwChange(e.target.value)}
        >
          <option value="nvidia_gb200_single">NVIDIA GB200 (Single Package)</option>
          <option value="nvidia_nvl72_rack">NVIDIA NVL-72 (Full Rack)</option>
        </select>
      </div>

      {/* Model Selection */}
      <div className="flex flex-col" style={{ gap: '16px', marginBottom: '24px' }}>
        <h3 className="text-accent font-semibold flex items-center" style={{ gap: '8px' }}>
          <Box size={18} /> Model
        </h3>
        <select
          className="input-field"
          value={modelId}
          onChange={(e) => onModelChange(e.target.value)}
        >
          <option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama 1.1B</option>
          <option value="meta-llama/Llama-3.3-70B-Instruct">Llama 3.3 70B</option>
        </select>
      </div>

      {/* Runtime Parameters */}
      <div className="flex flex-col" style={{ gap: '16px', marginBottom: '24px' }}>
        <h3 className="text-accent font-semibold flex items-center" style={{ gap: '8px' }}>
          <Cpu size={18} /> Runtime Parameters
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Batch Size</label>
            <input
              type="number"
              className="input-field"
              value={batchSize}
              onChange={(e) => onBatchChange?.(parseInt(e.target.value))}
              min="1"
            />
          </div>
          <div>
            <label className="label">Sequence Length</label>
            <input
              type="number"
              className="input-field"
              value={seqLength}
              onChange={(e) => onSeqChange?.(parseInt(e.target.value))}
              min="128"
            />
          </div>
        </div>
      </div>

      {/* Layer Configurations */}
      {layers && layers.length > 0 && (
        <div className="flex flex-col" style={{ gap: '16px' }}>
          <h3 className="text-accent font-semibold flex items-center" style={{ gap: '8px' }}>
            <Layers size={18} /> Layer Configuration
          </h3>
          <div className="flex flex-col" style={{ gap: '12px' }}>
            {layers.map(layer => (
              <LayerConfigCard
                key={layer.type}
                layer={layer}
                config={layerConfigs[layer.type]}
                onConfigChange={onLayerConfigChange}
                batchSize={batchSize}
                seqLength={seqLength}
                modelId={modelId}
                hwPreset={hwPreset}
              />
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}

// --- Main App ---

export default function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Config State
  const [modelId, setModelId] = useState('TinyLlama/TinyLlama-1.1B-Chat-v1.0');
  const [hwPreset, setHwPreset] = useState('nvidia_gb200_single');
  const [batchSize, setBatchSize] = useState(1);
  const [seqLength, setSeqLength] = useState(2048);
  
  // Layer State
  const [layers, setLayers] = useState([]);
  const [layerConfigs, setLayerConfigs] = useState({});
  
  // Metrics State
  const [systemMetrics, setSystemMetrics] = useState(null);
  
  // Comparison Mode
  const [comparisonMode, setComparisonMode] = useState(false);
  const [config2, setConfig2] = useState({
    modelId: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    hwPreset: 'nvidia_gb200_single',
    layerConfigs: {},
    layers: [],
    systemMetrics: null
  });

  // Initialize config2 layers when comparison mode is enabled
  useEffect(() => {
    if (comparisonMode && layers.length > 0 && Object.keys(config2.layerConfigs).length === 0) {
      const defaultConfigs = {};
      layers.forEach(layer => {
        defaultConfigs[layer.type] = {
          parallelism: {},
          dtype: 'bf16'
        };
        layer.available_parallelism.forEach(strategy => {
          defaultConfigs[layer.type].parallelism[strategy] = 1;
        });
      });
      setConfig2(prev => ({
        ...prev,
        layers: layers,
        layerConfigs: defaultConfigs
      }));
    }
  }, [comparisonMode, layers]);

  // Load config2 layers when its model changes
  useEffect(() => {
    if (comparisonMode && config2.modelId) {
      const loadConfig2Layers = async () => {
        try {
          const layersResponse = await axios.get(`${API_URL}/api/layers`, {
            params: { model_id: config2.modelId }
          });
          
          const defaultConfigs = {};
          layersResponse.data.layers.forEach(layer => {
            defaultConfigs[layer.type] = {
              parallelism: {},
              dtype: 'bf16'
            };
            layer.available_parallelism.forEach(strategy => {
              defaultConfigs[layer.type].parallelism[strategy] = 1;
            });
          });
          
          setConfig2(prev => ({
            ...prev,
            layers: layersResponse.data.layers,
            layerConfigs: defaultConfigs
          }));
        } catch (err) {
          console.error('Failed to load config2 layers:', err);
        }
      };
      loadConfig2Layers();
    }
  }, [comparisonMode, config2.modelId]);

  // Load layers (stateless - no server-side state)
  const loadLayers = async () => {
    setLoading(true);
    setError(null);
    try {
      // Get layer information using stateless endpoint
      const layersResponse = await axios.get(`${API_URL}/api/layers`, {
        params: { model_id: modelId }
      });
      setLayers(layersResponse.data.layers);
      
      // Initialize default configs (all layers with minimal parallelism)
      const defaultConfigs = {};
      layersResponse.data.layers.forEach(layer => {
        defaultConfigs[layer.type] = {
          parallelism: {},
          dtype: 'bf16'
        };
        layer.available_parallelism.forEach(strategy => {
          defaultConfigs[layer.type].parallelism[strategy] = 1;
        });
      });
      setLayerConfigs(defaultConfigs);
      
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to load model layers");
    } finally {
      setLoading(false);
    }
  };

  // Load on mount and when model changes
  useEffect(() => {
    if (modelId) {
      loadLayers();
    }
  }, [modelId]);

  // Handle layer config change
  const handleLayerConfigChange = async (layerType, strategy, value) => {
    let newConfig;
    
    if (strategy === 'dtype') {
      // Handle dtype change
      newConfig = {
        ...layerConfigs[layerType],
        dtype: value
      };
    } else {
      // Handle parallelism change
      newConfig = {
        ...layerConfigs[layerType],
        parallelism: {
          ...layerConfigs[layerType].parallelism,
          [strategy]: value
        }
      };
    }
    
    setLayerConfigs({
      ...layerConfigs,
      [layerType]: newConfig
    });
    // No backend call needed - stateless API!
  };

  // Compute aggregate metrics
  const computeMetrics = async () => {
    if (!modelId || !hwPreset || Object.keys(layerConfigs).length === 0) return;
    
    try {
      // Build layers payload from layerConfigs
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
      console.error('Request payload:', {
        model_id: modelId,
        hardware_config: hwPreset,
        batch_size: batchSize,
        input_seq: seqLength,
        output_seq: 128,
        layers: layersPayload
      });
      if (err.response?.data) {
        console.error('Backend error:', err.response.data);
      }
    }
  };

  useEffect(() => {
    if (layers.length > 0 && Object.keys(layerConfigs).length > 0 && modelId && hwPreset) {
      computeMetrics();
    }
  }, [layerConfigs, batchSize, seqLength, modelId, hwPreset]);

  return (
    <div className="app-container">
      {/* Left Panel - Configuration */}
      <ConfigPanel
        title="Configuration"
        modelId={modelId}
        hwPreset={hwPreset}
        layers={layers}
        layerConfigs={layerConfigs}
        onModelChange={setModelId}
        onHwChange={setHwPreset}
        onLayerConfigChange={handleLayerConfigChange}
        batchSize={batchSize}
        seqLength={seqLength}
        onBatchChange={setBatchSize}
        onSeqChange={setSeqLength}
      />

      {/* Center Panel - Metrics */}
      <main 
        className={comparisonMode ? "main-content with-comparison" : "main-content"}
        style={comparisonMode ? { marginRight: '340px' } : {}}
      >
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-500/10 border border-red-500/50 rounded-lg p-4 mb-6 flex items-start gap-3"
          >
            <AlertCircle size={20} className="text-red-500 mt-0.5" />
            <div className="flex-1">
              <p className="text-red-500 font-semibold">Error</p>
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          </motion.div>
        )}

        {loading && (
          <div className="flex items-center justify-center h-64">
            <div className="text-highlight-blue">Loading...</div>
          </div>
        )}

        {systemMetrics && !loading && (
          <>
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-white">System Metrics</h2>
              <div
                onClick={() => setComparisonMode(!comparisonMode)}
                className="cursor-pointer text-white hover:text-highlight-blue transition-colors flex items-center gap-2"
              >
                <Copy size={16} />
                <span>{comparisonMode ? 'Exit Comparison' : 'Compare Configs'}</span>
              </div>
            </div>

            {/* Aggregate Metrics */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <StatCard
                label="TTFT"
                value={systemMetrics.ttft_ms?.toFixed(3) || 'N/A'}
                unit="ms"
                icon={Activity}
                delay={0.1}
              />
              <StatCard
                label="TPOT"
                value={systemMetrics.tpot_ms?.toFixed(3) || 'N/A'}
                unit="ms"
                icon={Activity}
                delay={0.2}
              />
              <StatCard
                label="Throughput"
                value={systemMetrics.throughput_tokens_s?.toFixed(0) || 'N/A'}
                unit="tok/s"
                icon={Activity}
                delay={0.3}
              />
              <StatCard
                label="Total Memory"
                value={(
                  (systemMetrics.memory?.weight_memory_gb || 0) +
                  (systemMetrics.memory?.activation_memory_gb || 0) +
                  (systemMetrics.memory?.kv_cache_gb || 0)
                ).toFixed(1)}
                unit="GB"
                icon={Server}
                delay={0.4}
              />
            </div>

            {/* Runtime Controls moved to side panels */}
          </>
        )}
      </main>

      {/* Right Panel - Comparison (Optional) */}
      {comparisonMode && (
        <aside 
          className="coral-theme"
          style={{ 
            position: 'fixed',
            right: 0,
            top: 0,
            width: '340px',
            minWidth: '340px',
            height: '100vh',
            backgroundColor: '#1D2F61',
            zIndex: 10,
            overflowY: 'auto',
            padding: '32px',
            borderLeft: '1px solid #F48471',
            boxSizing: 'border-box'
          }}
        >
          <div className="flex items-center gap-3" style={{ marginBottom: '24px' }}>
            <div className="w-8 h-8 rounded bg-accent flex items-center justify-center">
              <Activity size={20} className="text-bg" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-white">Configuration 2</h1>
          </div>

          {/* Hardware Selection */}
          <div className="flex flex-col" style={{ gap: '16px', marginBottom: '24px' }}>
            <h3 className="text-accent font-semibold flex items-center" style={{ gap: '8px' }}>
              <Server size={18} /> Hardware
            </h3>
            <select
              className="input-field"
              value={config2.hwPreset}
              onChange={(e) => setConfig2({ ...config2, hwPreset: e.target.value })}
            >
              <option value="nvidia_gb200_single">NVIDIA GB200 (Single Package)</option>
              <option value="nvidia_nvl72_rack">NVIDIA NVL-72 (Full Rack)</option>
            </select>
          </div>

          {/* Model Selection */}
          <div className="flex flex-col" style={{ gap: '16px', marginBottom: '24px' }}>
            <h3 className="text-accent font-semibold flex items-center" style={{ gap: '8px' }}>
              <Box size={18} /> Model
            </h3>
            <select
              className="input-field"
              value={config2.modelId}
              onChange={(e) => setConfig2({ ...config2, modelId: e.target.value })}
            >
              <option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama 1.1B</option>
              <option value="meta-llama/Llama-3.3-70B-Instruct">Llama 3.3 70B</option>
            </select>
          </div>

          {/* Runtime Parameters (Right Panel) */}
          <div className="flex flex-col" style={{ gap: '16px', marginBottom: '24px' }}>
            <h3 className="text-accent font-semibold flex items-center" style={{ gap: '8px' }}>
              <Cpu size={18} /> Runtime Parameters
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">Batch Size</label>
                <input
                  type="number"
                  className="input-field"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  min="1"
                />
              </div>
              <div>
                <label className="label">Sequence Length</label>
                <input
                  type="number"
                  className="input-field"
                  value={seqLength}
                  onChange={(e) => setSeqLength(parseInt(e.target.value))}
                  min="128"
                />
              </div>
            </div>
          </div>

          {/* Layer Configurations */}
          {config2.layers && config2.layers.length > 0 && (
            <div className="flex flex-col" style={{ gap: '16px' }}>
              <h3 className="text-accent font-semibold flex items-center" style={{ gap: '8px' }}>
                <Layers size={18} /> Layer Configuration
              </h3>
              <div className="flex flex-col" style={{ gap: '12px' }}>
                {config2.layers.map(layer => (
                  <LayerConfigCard
                    key={layer.type}
                    layer={layer}
                    config={config2.layerConfigs[layer.type]}
                    onConfigChange={(layerType, strategy, value) => {
                      if (strategy === 'dtype') {
                        // Handle dtype change
                        setConfig2({
                          ...config2,
                          layerConfigs: {
                            ...config2.layerConfigs,
                            [layerType]: {
                              ...config2.layerConfigs[layerType],
                              dtype: value
                            }
                          }
                        });
                      } else {
                        // Handle parallelism change
                        setConfig2({
                          ...config2,
                          layerConfigs: {
                            ...config2.layerConfigs,
                            [layerType]: {
                              ...config2.layerConfigs[layerType],
                              parallelism: {
                                ...config2.layerConfigs[layerType]?.parallelism,
                                [strategy]: value
                              }
                            }
                          }
                        });
                      }
                    }}
                    batchSize={batchSize}
                    seqLength={seqLength}
                    modelId={config2.modelId}
                    hwPreset={config2.hwPreset}
                  />
                ))}
              </div>
            </div>
          )}
        </aside>
      )}
    </div>
  );
}
