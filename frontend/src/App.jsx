import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, AlertCircle, Copy, CheckCircle2, Server, Cpu, Layers } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import LayerConfigCard from './components/LayerConfigCard';
import MetricsDisplay from './components/MetricsDisplay';

// --- API Client ---
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // STATE STRUCTURE: model (system), prefill/decode (phase), layerConfigs (per-type)
  const [config, setConfig] = useState({
    model: 'llama_3.3_70b', // System level: model ID only
    prefill: {
      hardware: 'nvidia_nvl72_rack',
      batchSize: 128,
      seqLen: 1024
    },
    decode: {
      hardware: 'nvidia_nvl72_rack',
      batchSize: 128,
      seqLen: 1024
    },
    layerConfigs: {} // Layer TYPE configs: {[type]: {prefill: {tp, cp, sp}, decode: {tp, cp, sp}, dtype}}
  });
  
  // Layer metadata (from backend)
  const [layers, setLayers] = useState([]);
  
  // Metrics State (response from backend)
  const [metrics, setMetrics] = useState(null);
  
  // Available dtypes per hardware
  const [availableDtypes, setAvailableDtypes] = useState(['bf16', 'fp16', 'fp8', 'int8']);
  const [availableDtypesPrefill, setAvailableDtypesPrefill] = useState(['bf16', 'fp16', 'fp8', 'int8']);
  const [availableDtypesDecode, setAvailableDtypesDecode] = useState(['bf16', 'fp16', 'fp8', 'int8']);
  
  // Comparison Mode
  const [comparisonMode, setComparisonMode] = useState(false);
  const [comparisonConfig, setComparisonConfig] = useState({
    model: '',
    prefill: {
      hardware: '',
      batchSize: null,
      seqLen: null
    },
    decode: {
      hardware: '',
      batchSize: null,
      seqLen: null
    },
    layerConfigs: {}
  });
  const [comparisonMetrics, setComparisonMetrics] = useState(null);
  
  // Helper: derive dtype list from hardware capabilities
  const deriveDtypes = (hw) => {
    const dtypes = [];
    if (hw?.compute?.bf16) dtypes.push('bf16');
    if (hw?.compute?.fp16) dtypes.push('fp16');
    if (hw?.compute?.fp8) dtypes.push('fp8');
    if (hw?.compute?.int8) dtypes.push('int8');
    return dtypes.length ? dtypes : ['bf16', 'fp16', 'fp8', 'int8'];
  };

  // Helper: Build API request from current config state
  const buildSystemMetricsRequest = () => {
    const buildPhaseRequest = (phase) => {
      const phaseConfig = config[phase];
      
      // Build layers dict keyed by layer type name (not array)
      const layersDict = {};
      
      layers.forEach(layerMeta => {
        const layerTypeConfig = config.layerConfigs[layerMeta.name];
        const phaseParallelism = layerTypeConfig[phase];
        
        // Backend expects keys: tensor_parallel, context_parallel, sequence_parallel
        layersDict[layerMeta.name] = {
          tensor_parallel: phaseParallelism.tp_degree,
          context_parallel: phaseParallelism.cp_degree,
          sequence_parallel: phaseParallelism.sp_degree,
          dtype: layerTypeConfig.dtype
        };
      });
      
      return {
        model_id: config.model,
        hardware_id: phaseConfig.hardware,
        batch_size: Number(phaseConfig.batchSize),
        seq_len: Number(phaseConfig.seqLen),
        layers: layersDict
      };
    };

    return {
      prefill_req: buildPhaseRequest('prefill'),
      decode_req: buildPhaseRequest('decode')
    };
  };

  // Helper: Check if all required fields are filled (no defaults!)
  const isConfigComplete = () => {
    const hasAllLayerConfigs = layers.every(layer => {
      const cfg = config.layerConfigs[layer.name];
      return cfg && cfg.dtype; // Dtype is required for each layer type
    });
    
    return (
      config.model &&
      config.prefill.hardware &&
      config.prefill.batchSize !== null &&
      config.prefill.seqLen !== null &&
      config.decode.hardware &&
      config.decode.batchSize !== null &&
      config.decode.seqLen !== null &&
      Object.keys(config.layerConfigs).length > 0 &&
      hasAllLayerConfigs
    );
  };

  // Helper: Build comparison metrics request
  const buildComparisonMetricsRequest = () => {
    const buildPhaseRequest = (phase) => {
      const phaseConfig = comparisonConfig[phase];
      
      const layersDict = {};
      
      layers.forEach(layerMeta => {
        const layerTypeConfig = comparisonConfig.layerConfigs[layerMeta.name];
        const phaseParallelism = layerTypeConfig[phase];
        
        layersDict[layerMeta.name] = {
          tensor_parallel: phaseParallelism.tp_degree,
          context_parallel: phaseParallelism.cp_degree,
          sequence_parallel: phaseParallelism.sp_degree,
          dtype: layerTypeConfig.dtype
        };
      });

      return {
        model_id: comparisonConfig.model,
        hardware_config: phaseConfig.hardware,
        batch_size: Number(phaseConfig.batchSize),
        seq_len: Number(phaseConfig.seqLen),
        layers: layersDict
      };
    };

    return {
      prefill_req: buildPhaseRequest('prefill'),
      decode_req: buildPhaseRequest('decode')
    };
  };

  // Helper: Check if comparison config is complete
  const isComparisonConfigComplete = () => {
    const hasAllLayerConfigs = layers.every(layer => {
      const cfg = comparisonConfig.layerConfigs[layer.name];
      return cfg && cfg.dtype;
    });
    
    return (
      comparisonConfig.model &&
      comparisonConfig.prefill.hardware &&
      comparisonConfig.prefill.batchSize !== null &&
      comparisonConfig.prefill.seqLen !== null &&
      comparisonConfig.decode.hardware &&
      comparisonConfig.decode.batchSize !== null &&
      comparisonConfig.decode.seqLen !== null &&
      Object.keys(comparisonConfig.layerConfigs).length > 0 &&
      hasAllLayerConfigs
    );
  };



  // Load layers (stateless - no server-side state)
  const loadLayers = async () => {
    setLoading(true);
    setError(null);
    try {
      // Get layer information using stateless endpoint
      const layersResponse = await axios.get(`${API_URL}/api/layers`, {
        params: { model_id: config.model }
      });
      setLayers(layersResponse.data.layers);
      
      // Initialize layer configs by TYPE
      // Each type has: {prefill: {tp, cp, sp}, decode: {tp, cp, sp}, dtype}
      const defaultLayerConfigs = {};
      
      layersResponse.data.layers.forEach(layer => {
        defaultLayerConfigs[layer.name] = {
          prefill: {
            tp_degree: 1,
            cp_degree: 1,
            sp_degree: 1
          },
          decode: {
            tp_degree: 1,
            cp_degree: 1,
            sp_degree: 1
          },
          dtype: 'fp8' // Default to FP8
        };
      });
      
      setConfig(prev => ({
        ...prev,
        layerConfigs: defaultLayerConfigs
      }));
      
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to load model layers");
    } finally {
      setLoading(false);
    }
  };
  
  // Update available dtypes when hardware presets change
  useEffect(() => {
    const loadHardwareDtypes = async () => {
      // Only load if both hardware configs are set
      if (!config.prefill.hardware || !config.decode.hardware) {
        return;
      }
      
      try {
        // Load dtypes for prefill and decode hardware
        const resPrefill = await axios.get(`${API_URL}/hardware/${config.prefill.hardware}`);
        const resDecode = await axios.get(`${API_URL}/hardware/${config.decode.hardware}`);
        setAvailableDtypesPrefill(deriveDtypes(resPrefill.data));
        setAvailableDtypesDecode(deriveDtypes(resDecode.data));
        // Combined dtypes for shared selector (intersection)
        const commonDtypes = deriveDtypes(resPrefill.data).filter(dt => 
          deriveDtypes(resDecode.data).includes(dt)
        );
        setAvailableDtypes(commonDtypes.length ? commonDtypes : ['bf16', 'fp16', 'fp8', 'int8']);
      } catch (e) {
        console.error('Failed to load hardware dtypes:', e);
        setAvailableDtypes(['bf16', 'fp16', 'fp8', 'int8']);
        setAvailableDtypesPrefill(['bf16', 'fp16', 'fp8', 'int8']);
        setAvailableDtypesDecode(['bf16', 'fp16', 'fp8', 'int8']);
      }
    };
    loadHardwareDtypes();
  }, [config.prefill.hardware, config.decode.hardware]);

  // Load layers when model changes (auto-load)
  useEffect(() => {
    if (config.model && config.model !== '') {
      loadLayers();
    }
  }, [config.model]);

  // Compute aggregate metrics using new two-phase API
  const computeMetrics = async () => {
    // Check if config is complete
    if (!isConfigComplete()) {
      console.log('Config incomplete, skipping metrics computation');
      return;
    }
    
    setLoading(true);
    try {
      const requestPayload = buildSystemMetricsRequest();
      console.log('Computing metrics with payload:', requestPayload);
      
      const response = await axios.post(`${API_URL}/config/system-metrics`, requestPayload);
      setMetrics(response.data);
      console.log('Metrics response:', response.data);
    } catch (err) {
      console.error('Failed to compute metrics:', err);
      if (err.response?.data) {
        console.error('Backend error:', err.response.data);
        setError(err.response.data.detail || 'Failed to compute metrics');
      } else {
        setError('Failed to compute metrics');
      }
    } finally {
      setLoading(false);
    }
  };

  // Compute comparison metrics
  const computeComparisonMetrics = async () => {
    if (!isComparisonConfigComplete()) {
      console.log('Comparison config incomplete, skipping metrics computation');
      return;
    }
    
    try {
      const requestPayload = buildComparisonMetricsRequest();
      console.log('Computing comparison metrics with payload:', requestPayload);
      
      const response = await axios.post(`${API_URL}/config/system-metrics`, requestPayload);
      setComparisonMetrics(response.data);
      console.log('Comparison metrics response:', response.data);
    } catch (err) {
      console.error('Failed to compute comparison metrics:', err);
      if (err.response?.data) {
        console.error('Backend error:', err.response.data);
      }
    }
  };

  // Auto-compute metrics when config changes (only if complete)
  useEffect(() => {
    if (isConfigComplete()) {
      computeMetrics();
    }
  }, [config]);

  // Auto-compute comparison metrics when comparison config changes
  useEffect(() => {
    if (comparisonMode && isComparisonConfigComplete()) {
      computeComparisonMetrics();
    }
  }, [comparisonConfig, comparisonMode]);

  // Helper: Copy prefill config to decode (phase level only)
  const copyPrefillToDecodePhase = () => {
    setConfig(prev => ({
      ...prev,
      decode: {
        hardware: prev.prefill.hardware,
        batchSize: prev.prefill.batchSize,
        seqLen: prev.prefill.seqLen
      }
    }));
  };

  // Helper: Copy all layer configs from prefill to decode
  const copyPrefillToDecodeLayers = () => {
    setConfig(prev => ({
      ...prev,
      layerConfigs: Object.fromEntries(
        Object.entries(prev.layerConfigs).map(([type, cfg]) => [
          type,
          {
            ...cfg,
            decode: { ...cfg.prefill } // Copy prefill parallelism to decode
          }
        ])
      )
    }));
  };

  // Handler: Layer config change (layerType, phase, field, value)
  const handleLayerConfigChange = (layerType, phase, field, value) => {
    setConfig(prev => ({
      ...prev,
      layerConfigs: {
        ...prev.layerConfigs,
        [layerType]: {
          ...prev.layerConfigs[layerType],
          ...(phase ? {
            [phase]: {
              ...prev.layerConfigs[layerType][phase],
              [field]: value
            }
          } : {
            [field]: value // For dtype (shared across phases)
          })
        }
      }
    }));
  };

  return (
    <div className="app-container" style={{ display: 'flex', flexDirection: 'column', height: '100vh', backgroundColor: '#0F1729' }}>
      {/* Top Bar */}
      <header style={{
        backgroundColor: '#1D2F61',
        padding: '1rem 2rem',
        borderBottom: '1px solid #29AF83',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1rem'
      }}>
        <div className="flex items-center gap-2">
          <img src="/mantile-logo.png" alt="Mantile" style={{ height: '38px' }} />
        </div>
        
        <button
          onClick={() => {
            if (!comparisonMode) {
              // Initialize comparison config with same values as primary
              setComparisonConfig({
                model: config.model,
                prefill: { ...config.prefill },
                decode: { ...config.decode },
                layerConfigs: JSON.parse(JSON.stringify(config.layerConfigs))
              });
            }
            setComparisonMode(!comparisonMode);
          }}
          style={{
            fontSize: '0.9rem',
            marginLeft: 'auto',
            color: '#D1D5DB',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: '0.5rem 1rem'
          }}
        >
          {comparisonMode ? 'Exit Comparison' : 'Compare Systems'}
        </button>
      </header>

      {/* Main Layout: Two Sidebars + Content */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Primary Sidebars Container */}
        <div style={{ position: 'relative', display: 'flex' }}>
          {/* Model Selection - Spans Both Primary Phases */}
          <div style={{ 
            position: 'absolute',
            left: '0',
            top: '0',
            width: '600px',
            backgroundColor: '#1D2F61',
            padding: '1.5rem',
            borderBottom: '1px solid #29AF83',
            zIndex: 100
          }}>
            <label className="label">Model</label>
            <select
              value={config.model}
              onChange={(e) => {
                const newModel = e.target.value;
                setConfig(prev => ({...prev, model: newModel}));
              }}
              disabled={loading}
              className="input-field"
            >
              <option value="">Select a model...</option>
              <option value="tinyllama_1.1b">TinyLlama 1.1B Chat</option>
              <option value="llama_3.3_70b">Llama 3.3 70B Instruct</option>
            </select>
          </div>

        {/* Left Sidebar - Prefill Phase Configuration */}
        <aside style={{
          width: '300px',
          minWidth: '300px',
          backgroundColor: '#1D2F61',
          padding: '1.5rem',
          overflowY: 'auto',
          borderLeft: '1px solid #29AF83',
          borderRight: '1px solid #29AF83'
        }}>
          <div style={{ height: '95px' }} />

          {/* Prefill Phase Configuration */}
        {layers.length > 0 && (
          <>
            <div className="flex flex-col gap-4 mb-6" style={{ marginTop: '1.5rem' }}>
              <h3 className="text-white font-bold flex items-center gap-2" style={{ color: '#3B82F6' }}>
                🔹 Prefill
              </h3>
              
              {/* Hardware */}
              <div>
                <label className="label">Hardware</label>
                <select
                  value={config.prefill.hardware}
                  onChange={(e) => setConfig(prev => ({...prev, prefill: {...prev.prefill, hardware: e.target.value}}))}
                  className="input-field"
                >
                  <option value="">Select hardware...</option>
                  <option value="nvidia_gb200_single">NVIDIA GB200 (Single)</option>
                  <option value="nvidia_nvl72_rack">NVIDIA NVL-72 (Full Rack)</option>
                  <option value="nvidia_h100_80gb">NVIDIA H100 80GB</option>
                </select>
              </div>

              {/* Runtime Parameters */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="label">Batch Size</label>
                  <input
                    type="number"
                    value={config.prefill.batchSize || ''}
                    onChange={(e) => setConfig(prev => ({...prev, prefill: {...prev.prefill, batchSize: parseInt(e.target.value) || null}}))}
                    placeholder="1"
                    className="input-field"
                    min="1"
                  />
                </div>
                <div>
                  <label className="label">Seq Length</label>
                  <input
                    type="number"
                    value={config.prefill.seqLen || ''}
                    onChange={(e) => setConfig(prev => ({...prev, prefill: {...prev.prefill, seqLen: parseInt(e.target.value) || null}}))}
                    placeholder="2048"
                    className="input-field"
                    min="1"
                  />
                </div>
              </div>
            </div>

            {/* Layer Configuration Cards */}
            <div className="flex flex-col gap-4" style={{ marginTop: '1.5rem' }}>
              <h3 className="text-accent font-semibold flex items-center gap-2">
                <Layers size={18} /> Layer Configuration
              </h3>
              {layers.map(layer => (
                <LayerConfigCard
                  key={layer.name}
                  layer={layer}
                  config={config.layerConfigs[layer.name]}
                  onConfigChange={handleLayerConfigChange}
                  phase="prefill"
                />
              ))}
            </div>
          </>
        )}

        {!layers.length && (
          <div className="text-dim text-sm italic text-center mt-8">
            Select a model to begin configuration
          </div>
        )}
      </aside>

      {/* Middle Sidebar - Decode Phase Configuration */}
      <aside style={{
        width: '300px',
        minWidth: '300px',
        backgroundColor: '#1D2F61',
        padding: '1.5rem',
        overflowY: 'auto',
        borderLeft: '1px solid #10B981',
        borderRight: '1px solid #10B981'
      }}>
        {/* Spacer to align with model dropdown in prefill sidebar */}
        <div style={{ height: '95px' }} />
        
        {/* Decode Phase Configuration */}
        {layers.length > 0 && (
          <>
              <div className="flex flex-col gap-4 mb-6" style={{ marginTop: '1.5rem' }}>
                <div className="flex items-center justify-between">
                  <h3 className="text-white font-bold flex items-center gap-2" style={{ color: '#10B981' }}>
                    🔸 Decode
                  </h3>
                  <button
                    onClick={() => {
                      setConfig(prev => ({
                        ...prev,
                        decode: {
                          hardware: prev.prefill.hardware,
                          batchSize: prev.prefill.batchSize,
                          seqLen: prev.prefill.seqLen
                        },
                        layerConfigs: Object.keys(prev.layerConfigs).reduce((acc, layerName) => {
                          acc[layerName] = {
                            ...prev.layerConfigs[layerName],
                            decode: { ...prev.layerConfigs[layerName].prefill }
                          };
                          return acc;
                        }, {})
                      }));
                    }}
                    className="text-xs"
                    style={{
                      background: 'none',
                      color: '#10B981',
                      border: 'none',
                      cursor: 'pointer',
                      fontSize: '0.75rem',
                      padding: '0'
                    }}
                    title="Copy all configuration from Prefill phase"
                  >
                    Copy from Prefill
                  </button>
                </div>
                
                {/* Hardware */}
                <div>
                  <label className="label">Hardware</label>
                  <select
                    value={config.decode.hardware}
                    onChange={(e) => setConfig(prev => ({...prev, decode: {...prev.decode, hardware: e.target.value}}))}
                    className="input-field"
                  >
                    <option value="">Select hardware...</option>
                    <option value="nvidia_gb200_single">NVIDIA GB200 (Single)</option>
                    <option value="nvidia_nvl72_rack">NVIDIA NVL-72 (Full Rack)</option>
                    <option value="nvidia_h100_80gb">NVIDIA H100 80GB</option>
                  </select>
                </div>

                {/* Runtime Parameters */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="label">Batch Size</label>
                    <input
                      type="number"
                      value={config.decode.batchSize || ''}
                      onChange={(e) => setConfig(prev => ({...prev, decode: {...prev.decode, batchSize: parseInt(e.target.value) || null}}))}
                      placeholder="32"
                      className="input-field"
                      min="1"
                    />
                  </div>
                  <div>
                    <label className="label">Seq Length</label>
                    <input
                      type="number"
                      value={config.decode.seqLen || ''}
                      onChange={(e) => setConfig(prev => ({...prev, decode: {...prev.decode, seqLen: parseInt(e.target.value) || null}}))}
                      placeholder="256"
                      className="input-field"
                      min="1"
                    />
                  </div>
                </div>
              </div>

            {/* Layer Configuration Cards */}
            <div className="flex flex-col gap-4" style={{ marginTop: '1.5rem' }}>
              <h3 className="text-accent font-semibold flex items-center gap-2">
                <Layers size={18} /> Layer Configuration
              </h3>
              {layers.map(layer => (
                <LayerConfigCard
                  key={layer.name}
                  layer={layer}
                  config={config.layerConfigs[layer.name]}
                  onConfigChange={handleLayerConfigChange}
                  phase="decode"
                />
              ))}
            </div>
          </>
        )}

        {!layers.length && (
          <div className="text-dim text-sm italic text-center mt-8">
            Select a model to begin
          </div>
        )}
      </aside>
      </div> {/* End Primary Sidebars Container */}

      {/* Main Content Area */}
      <main className="flex-1 p-8 overflow-y-auto" style={{ width: '100%', minWidth: 0 }}>
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

        {layers.length > 0 && (
          <>
            {/* Auto-computing status indicator */}
            {loading && (
              <div className="mb-4 flex items-center gap-2 text-sm text-gray-400">
                <Activity size={16} className="animate-spin" />
                <span>Computing metrics...</span>
              </div>
            )}

            {/* Metrics Display */}
            {metrics && (
              <MetricsDisplay 
                metrics={metrics} 
                comparisonMetrics={comparisonMode ? comparisonMetrics : null}
              />
            )}
          </>
        )}

        {!layers.length && !error && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-dim max-w-md">
              <h2 className="text-2xl font-bold text-white mb-4">Welcome to Mantile</h2>
              <p className="mb-4">
                Select a model from the top to begin configuring your system.
              </p>
            </div>
          </div>
        )}
        </main>

        {/* Comparison Sidebars - Only shown when comparison mode is active */}
        {comparisonMode && (
          <div style={{ position: 'relative', display: 'flex', marginLeft: 'auto' }}>
            {/* Model Selection - Spans Both Comparison Phases */}
            <div style={{ 
              position: 'absolute',
              left: '0',
              top: '0',
              width: '600px',
              backgroundColor: '#1D2F61',
              padding: '1.5rem',
              borderBottom: '1px solid #f96c56',
              zIndex: 100
            }}>
              <label className="label">Model</label>
              <select
                value={comparisonConfig.model}
                onChange={(e) => {
                  const newModel = e.target.value;
                  setComparisonConfig(prev => ({...prev, model: newModel}));
                }}
                disabled={loading}
                className="input-field"
              >
                <option value="">Select a model...</option>
                <option value="tinyllama_1.1b">TinyLlama 1.1B Chat</option>
                <option value="llama_3.3_70b">Llama 3.3 70B Instruct</option>
              </select>
            </div>

            {/* Comparison Prefill Sidebar */}
            <aside style={{
              width: '300px',
              minWidth: '300px',
              backgroundColor: '#1D2F61',
              padding: '1.5rem',
              overflowY: 'auto',
              borderLeft: '1px solid #f96c56',
              borderRight: '1px solid #f96c56'
            }}>
              <div style={{ height: '95px' }} />

              {/* Prefill Phase Configuration */}
              {layers.length > 0 && (
                <>
                  <div className="flex flex-col gap-4 mb-6" style={{ marginTop: '1.5rem' }}>
                    <h3 className="text-white font-bold flex items-center gap-2" style={{ color: '#f96c56' }}>
                      🔹 Prefill
                    </h3>
                    
                    {/* Hardware */}
                    <div>
                      <label className="label">Hardware</label>
                      <select
                        value={comparisonConfig.prefill.hardware}
                        onChange={(e) => setComparisonConfig(prev => ({...prev, prefill: {...prev.prefill, hardware: e.target.value}}))}
                        className="input-field"
                      >
                        <option value="">Select hardware...</option>
                        <option value="nvidia_gb200_single">NVIDIA GB200 (Single)</option>
                        <option value="nvidia_nvl72_rack">NVIDIA NVL-72 (Full Rack)</option>
                        <option value="nvidia_h100_80gb">NVIDIA H100 80GB</option>
                      </select>
                    </div>

                    {/* Runtime Parameters */}
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="label">Batch Size</label>
                        <input
                          type="number"
                          value={comparisonConfig.prefill.batchSize || ''}
                          onChange={(e) => setComparisonConfig(prev => ({...prev, prefill: {...prev.prefill, batchSize: parseInt(e.target.value) || null}}))}
                          placeholder="1"
                          className="input-field"
                          min="1"
                        />
                      </div>
                      <div>
                        <label className="label">Seq Length</label>
                        <input
                          type="number"
                          value={comparisonConfig.prefill.seqLen || ''}
                          onChange={(e) => setComparisonConfig(prev => ({...prev, prefill: {...prev.prefill, seqLen: parseInt(e.target.value) || null}}))}
                          placeholder="2048"
                          className="input-field"
                          min="1"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Layer Configuration Cards */}
                  <div className="flex flex-col gap-4" style={{ marginTop: '1.5rem' }}>
                    <h3 className="text-accent font-semibold flex items-center gap-2">
                      <Layers size={18} /> Layer Configuration
                    </h3>
                    {layers.map(layer => (
                      <LayerConfigCard
                        key={layer.name}
                        layer={layer}
                        config={comparisonConfig.layerConfigs[layer.name]}
                        onConfigChange={(layerName, phase, field, value) => {
                          setComparisonConfig(prev => ({
                            ...prev,
                            layerConfigs: {
                              ...prev.layerConfigs,
                              [layerName]: {
                                ...prev.layerConfigs[layerName],
                                ...(phase ? {
                                  [phase]: {
                                    ...prev.layerConfigs[layerName]?.[phase],
                                    [field]: value
                                  }
                                } : {
                                  [field]: value
                                })
                              }
                            }
                          }));
                        }}
                        phase="prefill"
                        accentColor="#f96c56"
                      />
                    ))}
                  </div>
                </>
              )}

              {!layers.length && (
                <div className="text-dim text-sm italic text-center mt-8">
                  Select a model to begin
                </div>
              )}
            </aside>

            {/* Comparison Decode Sidebar */}
            <aside style={{
              width: '300px',
              minWidth: '300px',
              backgroundColor: '#1D2F61',
              padding: '1.5rem',
              overflowY: 'auto',
              borderLeft: '1px solid #f96c56',
              borderRight: '1px solid #f96c56'
            }}>
              {/* Spacer to align with model dropdown in prefill sidebar */}
              <div style={{ height: '95px' }} />
              
              {/* Decode Phase Configuration */}
              {layers.length > 0 && (
                <>
                  <div className="flex flex-col gap-4 mb-6" style={{ marginTop: '1.5rem' }}>
                    <div className="flex items-center justify-between">
                      <h3 className="text-white font-bold flex items-center gap-2" style={{ color: '#f96c56' }}>
                        🔸 Decode
                      </h3>
                      <button
                        onClick={() => {
                          setComparisonConfig(prev => ({
                            ...prev,
                            decode: {
                              hardware: prev.prefill.hardware,
                              batchSize: prev.prefill.batchSize,
                              seqLen: prev.prefill.seqLen
                            }
                          }));
                          const copiedConfigs = {};
                          Object.keys(comparisonConfig.layerConfigs).forEach(layerName => {
                            copiedConfigs[layerName] = {
                              ...comparisonConfig.layerConfigs[layerName],
                              decode: { ...comparisonConfig.layerConfigs[layerName].prefill }
                            };
                          });
                          setComparisonConfig(prev => ({
                            ...prev,
                            layerConfigs: copiedConfigs
                          }));
                        }}
                        className="text-xs"
                        style={{
                          background: 'none',
                          color: '#f96c56',
                          border: 'none',
                          cursor: 'pointer',
                          fontSize: '0.75rem',
                          padding: '0'
                        }}
                      >
                        Copy from Prefill
                      </button>
                    </div>
                    
                    {/* Hardware */}
                    <div>
                      <label className="label">Hardware</label>
                      <select
                        value={comparisonConfig.decode.hardware}
                        onChange={(e) => setComparisonConfig(prev => ({...prev, decode: {...prev.decode, hardware: e.target.value}}))}
                        className="input-field"
                      >
                        <option value="">Select hardware...</option>
                        <option value="nvidia_gb200_single">NVIDIA GB200 (Single)</option>
                        <option value="nvidia_nvl72_rack">NVIDIA NVL-72 (Full Rack)</option>
                        <option value="nvidia_h100_80gb">NVIDIA H100 80GB</option>
                      </select>
                    </div>

                    {/* Runtime Parameters */}
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="label">Batch Size</label>
                        <input
                          type="number"
                          value={comparisonConfig.decode.batchSize || ''}
                          onChange={(e) => setComparisonConfig(prev => ({...prev, decode: {...prev.decode, batchSize: parseInt(e.target.value) || null}}))}
                          placeholder="1"
                          className="input-field"
                          min="1"
                        />
                      </div>
                      <div>
                        <label className="label">Seq Length</label>
                        <input
                          type="number"
                          value={comparisonConfig.decode.seqLen || ''}
                          onChange={(e) => setComparisonConfig(prev => ({...prev, decode: {...prev.decode, seqLen: parseInt(e.target.value) || null}}))}
                          placeholder="1"
                          className="input-field"
                          min="1"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Layer Configuration Cards */}
                  <div className="flex flex-col gap-4" style={{ marginTop: '1.5rem' }}>
                    <h3 className="text-accent font-semibold flex items-center gap-2">
                      <Layers size={18} /> Layer Configuration
                    </h3>
                    {layers.map(layer => (
                      <LayerConfigCard
                        key={layer.name}
                        layer={layer}
                        config={comparisonConfig.layerConfigs[layer.name]}
                        onConfigChange={(layerName, phase, field, value) => {
                          setComparisonConfig(prev => ({
                            ...prev,
                            layerConfigs: {
                              ...prev.layerConfigs,
                              [layerName]: {
                                ...prev.layerConfigs[layerName],
                                ...(phase ? {
                                  [phase]: {
                                    ...prev.layerConfigs[layerName]?.[phase],
                                    [field]: value
                                  }
                                } : {
                                  [field]: value
                                })
                              }
                            }
                          }));
                        }}
                        phase="decode"
                        accentColor="#f96c56"
                      />
                    ))}
                  </div>
                </>
              )}

              {!layers.length && (
                <div className="text-dim text-sm italic text-center mt-8">
                  Select a model to begin
                </div>
              )}
            </aside>
          </div>
        )}
      </div>
    </div>
  );
}
