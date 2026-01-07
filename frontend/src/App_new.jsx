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
      className="card flex flex-col gap-2"
    >
      <div className="flex justify-between items-start">
        <span className="text-dim text-sm uppercase tracking-wider">{label}</span>
        {Icon && <Icon size={18} className="text-accent" style={{ opacity: 0.7 }} />}
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-3xl font-bold text-white">{value}</span>
        {unit && <span className="text-dim text-sm">{unit}</span>}
      </div>
    </motion.div>
  );
}

function LayerConfigCard({ layer, config, onConfigChange, batchSize, seqLength }) {
  const [metrics, setMetrics] = useState(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  useEffect(() => {
    if (config) {
      fetchMetrics();
    }
  }, [config, batchSize, seqLength]);

  const fetchMetrics = async () => {
    if (!config) return;
    setLoadingMetrics(true);
    try {
      const response = await axios.post(`${API_URL}/config/layer-metrics`, {
        layer_type: layer.type,
        batch_size: batchSize,
        seq_length: seqLength
      });
      setMetrics(response.data);
    } catch (err) {
      console.error('Failed to fetch layer metrics:', err);
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
          <h4 className="text-white font-semibold capitalize">{layer.type}</h4>
          <p className="text-dim text-sm">{layer.count} instances</p>
        </div>
        {config && <CheckCircle size={20} className="text-accent" />}
      </div>

      {availableStrategies.length > 0 ? (
        <div className="flex flex-col gap-3">
          {availableStrategies.map(strategy => (
            <div key={strategy}>
              <label className="text-dim text-xs uppercase tracking-wider">
                {strategy.replace('_', ' ')}
              </label>
              <div className="flex items-center gap-2 mt-1">
                <input
                  type="range"
                  min="1"
                  max={maxChips}
                  value={config?.parallelism[strategy] || 1}
                  onChange={(e) => onConfigChange(layer.type, strategy, parseInt(e.target.value))}
                  className="flex-1 h-2 bg-surface rounded-lg appearance-none cursor-pointer accent-accent"
                />
                <span className="text-accent font-mono text-sm w-8 text-right">
                  {config?.parallelism[strategy] || 1}
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-dim text-sm italic">Replicated (no parallelism)</div>
      )}

      {/* Layer Metrics */}
      {metrics && (
        <div className="mt-4 pt-4 border-t border-surface">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-dim">Chips:</span>
              <span className="text-white ml-2 font-mono">{metrics.num_chips}</span>
            </div>
            <div>
              <span className="text-dim">Weight Mem:</span>
              <span className="text-white ml-2 font-mono">{metrics.memory.total_weights_gb.toFixed(2)} GB</span>
            </div>
            <div>
              <span className="text-dim">Act Mem:</span>
              <span className="text-white ml-2 font-mono">{metrics.memory.total_activation_gb.toFixed(2)} GB</span>
            </div>
            <div>
              <span className="text-dim">FLOPs:</span>
              <span className="text-white ml-2 font-mono">{metrics.compute.total_flops_tflops.toFixed(1)} TF</span>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}

function ConfigPanel({ title, modelId, hwPreset, layers, layerConfigs, onModelChange, onHwChange, onLayerConfigChange, batchSize, seqLength }) {
  return (
    <aside className="sidebar">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-8 h-8 rounded bg-accent flex items-center justify-center">
          <Activity size={20} className="text-bg" />
        </div>
        <h1 className="text-xl font-bold tracking-tight text-white">{title}</h1>
      </div>

      {/* Hardware Selection */}
      <div className="flex flex-col gap-4 mb-6">
        <h3 className="text-accent font-semibold flex items-center gap-2">
          <Server size={18} /> Hardware
        </h3>
        <select
          className="input-field"
          value={hwPreset}
          onChange={(e) => onHwChange(e.target.value)}
        >
          <option value="nvl72_single">NVIDIA GB200 (Single Package)</option>
          <option value="nvl72_rack">NVIDIA NVL-72 (Full Rack)</option>
        </select>
      </div>

      {/* Model Selection */}
      <div className="flex flex-col gap-4 mb-6">
        <h3 className="text-accent font-semibold flex items-center gap-2">
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

      {/* Layer Configurations */}
      {layers && layers.length > 0 && (
        <div className="flex flex-col gap-4">
          <h3 className="text-accent font-semibold flex items-center gap-2">
            <Layers size={18} /> Layer Configuration
          </h3>
          <div className="flex flex-col gap-3">
            {layers.map(layer => (
              <LayerConfigCard
                key={layer.type}
                layer={layer}
                config={layerConfigs[layer.type]}
                onConfigChange={onLayerConfigChange}
                batchSize={batchSize}
                seqLength={seqLength}
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
  const [hwPreset, setHwPreset] = useState('nvl72_single');
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
    hwPreset: 'nvl72_single',
    layerConfigs: {}
  });

  // Load model and hardware
  const loadModelAndHardware = async () => {
    setLoading(true);
    setError(null);
    try {
      // Step 1: Load model and hardware
      const response = await axios.post(`${API_URL}/config/load`, {
        model_id: modelId,
        hardware_config: hwPreset
      });
      
      // Step 2: Get layer information
      const layersResponse = await axios.get(`${API_URL}/config/layers`);
      setLayers(layersResponse.data.layers);
      
      // Initialize default configs (all layers with minimal parallelism)
      const defaultConfigs = {};
      layersResponse.data.layers.forEach(layer => {
        defaultConfigs[layer.type] = {
          parallelism: {}
        };
        layer.available_parallelism.forEach(strategy => {
          defaultConfigs[layer.type].parallelism[strategy] = 1;
        });
      });
      setLayerConfigs(defaultConfigs);
      
      // Configure all layers on backend
      for (const layer of layersResponse.data.layers) {
        await axios.post(`${API_URL}/config/layer-parallelism`, {
          layer_type: layer.type,
          ...defaultConfigs[layer.type].parallelism
        });
      }
      
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to load model and hardware");
    } finally {
      setLoading(false);
    }
  };
        // Removed: legacy /config/load call
  }, [modelId, hwPreset]);

  // Handle layer config change
  const handleLayerConfigChange = async (layerType, strategy, value) => {
    const newConfig = {
      ...layerConfigs[layerType],
      parallelism: {
        ...layerConfigs[layerType].parallelism,
        [strategy]: value
      }
    };
    
    setLayerConfigs({
      ...layerConfigs,
      [layerType]: newConfig
    });
    
    // Update backend
    try {
      await axios.post(`${API_URL}/config/layer-parallelism`, {
        layer_type: layerType,
        ...newConfig.parallelism
      });
    } catch (err) {
      console.error('Failed to update layer config:', err);
    }
  };

  // Compute aggregate metrics
  const computeMetrics = async () => {
    try {
      const response = await axios.post(`${API_URL}/config/system-metrics`, {
        batch_size: batchSize,
        input_seq: seqLength,
        output_seq: 128
      });
      setSystemMetrics(response.data);
    } catch (err) {
      console.error('Failed to compute metrics:', err);
    }
  };

  useEffect(() => {
    if (layers.length > 0 && Object.keys(layerConfigs).length > 0) {
      computeMetrics();
    }
  }, [layerConfigs, batchSize, seqLength]);

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
      />

      {/* Center Panel - Metrics */}
      <main className="main-content">
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
            <div className="text-accent">Loading...</div>
          </div>
        )}

        {systemMetrics && !loading && (
          <>
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-white">System Metrics</h2>
              <button
                onClick={() => setComparisonMode(!comparisonMode)}
                className="px-4 py-2 bg-surface hover:bg-accent/20 text-accent rounded-lg flex items-center gap-2 transition-colors"
              >
                <Copy size={16} />
                {comparisonMode ? 'Exit' : 'Compare'}
              </button>
            </div>

            {/* Aggregate Metrics */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <StatCard
                label="TTFT"
                value={systemMetrics.time_to_first_token_ms?.toFixed(3) || 'N/A'}
                unit="ms"
                icon={Activity}
                delay={0.1}
              />
              <StatCard
                label="TPOT"
                value={systemMetrics.time_per_output_token_ms?.toFixed(3) || 'N/A'}
                unit="ms"
                icon={Activity}
                delay={0.2}
              />
              <StatCard
                label="Throughput"
                value={systemMetrics.total_throughput_tokens_s?.toFixed(0) || 'N/A'}
                unit="tok/s"
                icon={Activity}
                delay={0.3}
              />
              <StatCard
                label="Total Memory"
                value={((systemMetrics.weights_mem_gb || 0) + (systemMetrics.activation_mem_gb || 0) + (systemMetrics.kv_cache_mem_gb || 0)).toFixed(1)}
                unit="GB"
                icon={Server}
                delay={0.4}
              />
            </div>

            {/* Runtime Controls */}
            <div className="card mb-6">
              <h3 className="text-lg font-semibold mb-4 text-white">Runtime Parameters</h3>
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
          </>
        )}
      </main>

      {/* Right Panel - Comparison (Optional) */}
      {comparisonMode && (
        <ConfigPanel
          title="Configuration 2"
          modelId={config2.modelId}
          hwPreset={config2.hwPreset}
          layers={layers}
          layerConfigs={config2.layerConfigs}
          onModelChange={(val) => setConfig2({ ...config2, modelId: val })}
          onHwChange={(val) => setConfig2({ ...config2, hwPreset: val })}
          onLayerConfigChange={(layerType, strategy, value) => {
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
          }}
          batchSize={batchSize}
          seqLength={seqLength}
        />
      )}
    </div>
  );
}
