import React from 'react';
import { Copy } from 'lucide-react';

/**
 * Two-column layer configuration component
 * Shows prefill and decode side-by-side with layer TYPE configurations
 * 
 * Hierarchy:
 * 1. System level (model) - in parent App.jsx
 * 2. Phase level (hardware, batch size, seq length) - top section
 * 3. Layer level (parallelism and dtype) - table below
 */
export default function TwoPhaseLayerConfig({
  layers,
  layerConfigs,
  prefillConfig,
  decodeConfig,
  onPrefillChange,
  onDecodeChange,
  onLayerConfigChange,
  onCopyToDecodePhase,
  onCopyToDecodeLayers
}) {
  // Check if configuration is complete
  const isPrefillComplete = prefillConfig.hardware && 
                            prefillConfig.batchSize !== null && 
                            prefillConfig.seqLen !== null;
  const isDecodeComplete = decodeConfig.hardware && 
                           decodeConfig.batchSize !== null && 
                           decodeConfig.seqLen !== null;
  const areLayersComplete = layers.length > 0 && 
                            layers.every(layer => {
                              const cfg = layerConfigs[layer.name];
                              return cfg && cfg.dtype;
                            });
  const isFullyComplete = isPrefillComplete && isDecodeComplete && areLayersComplete;
  
  return (
    <div className="two-phase-config">
      <div style={{ marginBottom: '2rem' }}>
        <h2 className="section-title">Configuration Hierarchy</h2>
        <div style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '1.5rem' }}>
          <p>1. <strong style={{ color: '#29AF83' }}>System Level:</strong> Model selected above</p>
          <p>2. <strong style={{ color: '#29AF83' }}>Phase Level:</strong> Hardware, batch size, and sequence length per phase</p>
          <p>3. <strong style={{ color: '#29AF83' }}>Layer Level:</strong> Parallelism (TP/CP/SP) and dtype per layer type</p>
        </div>
      </div>
      
      {/* Phase-level settings */}
      <h3 className="subsection-title">Phase Configuration</h3>
      <div className="phase-settings-grid">
        {/* Prefill Column */}
        <div className="phase-column prefill-column">
          <h3 className="phase-header">🔹 Prefill Phase</h3>
          
          <div className="phase-inputs">
            <div className="input-group">
              <label>Hardware</label>
              <select 
                value={prefillConfig.hardware}
                onChange={(e) => onPrefillChange('hardware', e.target.value)}
                className="input-field"
              >
                <option value="">Select hardware...</option>
                <option value="nvidia_gb200_single">NVIDIA GB200 (Single Package)</option>
                <option value="nvidia_nvl72_rack">NVIDIA NVL-72 (Full Rack)</option>
                <option value="nvidia_h100_80gb">NVIDIA H100 80GB</option>
              </select>
            </div>
            
            <div className="input-group">
              <label>Batch Size</label>
              <input
                type="number"
                value={prefillConfig.batchSize || ''}
                onChange={(e) => onPrefillChange('batchSize', e.target.value)}
                placeholder="e.g., 1"
                className="input-field"
              />
            </div>
            
            <div className="input-group">
              <label>Sequence Length</label>
              <input
                type="number"
                value={prefillConfig.seqLen || ''}
                onChange={(e) => onPrefillChange('seqLen', e.target.value)}
                placeholder="e.g., 2048"
                className="input-field"
              />
            </div>
          </div>
        </div>
        
        {/* Decode Column */}
        <div className="phase-column decode-column">
          <h3 className="phase-header">🔸 Decode Phase</h3>
          
          <div className="phase-inputs">
            <div className="input-group">
              <label>Hardware</label>
              <select 
                value={decodeConfig.hardware}
                onChange={(e) => onDecodeChange('hardware', e.target.value)}
                className="input-field"
              >
                <option value="">Select hardware...</option>
                <option value="nvidia_gb200_single">NVIDIA GB200 (Single Package)</option>
                <option value="nvidia_nvl72_rack">NVIDIA NVL-72 (Full Rack)</option>
                <option value="nvidia_h100_80gb">NVIDIA H100 80GB</option>
              </select>
            </div>
            
            <div className="input-group">
              <label>Batch Size</label>
              <input
                type="number"
                value={decodeConfig.batchSize || ''}
                onChange={(e) => onDecodeChange('batchSize', e.target.value)}
                placeholder="e.g., 32"
                className="input-field"
              />
            </div>
            
            <div className="input-group">
              <label>Sequence Length</label>
              <input
                type="number"
                value={decodeConfig.seqLen || ''}
                onChange={(e) => onDecodeChange('seqLen', e.target.value)}
                placeholder="e.g., 1"
                className="input-field"
              />
            </div>
          </div>
          
          <button 
            onClick={onCopyToDecodePhase}
            className="btn btn-secondary copy-btn"
            title="Copy phase settings (hardware, batch size, sequence length) from prefill to decode"
          >
            <Copy size={14} style={{ marginRight: '0.25rem' }} />
            Copy Phase Settings from Prefill
          </button>
        </div>
      </div>

      {/* Layer configuration table */}
      <div className="layers-table-container">
        <h3 className="subsection-title">Layer-Level Configuration</h3>
        <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '1rem' }}>
          Configure parallelism strategies (TP/CP/SP) and data type for each layer type. 
          Settings apply to all {layers.reduce((sum, l) => sum + l.count, 0)} layer instances.
        </p>
        
        {layers.length === 0 ? (
          <div className="empty-state">
            Load a model to configure layer parallelism
          </div>
        ) : (
          <div className="layers-table-wrapper">
            <table className="layers-table">
              <thead>
                <tr>
                  <th rowSpan="2" style={{ minWidth: '120px' }}>Layer Type</th>
                  <th rowSpan="2" style={{ textAlign: 'center', width: '80px' }}>Count</th>
                  <th rowSpan="2" style={{ textAlign: 'center', width: '100px' }}>Data Type</th>
                  <th colSpan="3" className="prefill-header" style={{ textAlign: 'center' }}>🔹 Prefill Parallelism</th>
                  <th colSpan="3" className="decode-header" style={{ textAlign: 'center' }}>🔸 Decode Parallelism</th>
                </tr>
                <tr>
                  <th className="sub-header" style={{ textAlign: 'center' }}>TP</th>
                  <th className="sub-header" style={{ textAlign: 'center' }}>CP</th>
                  <th className="sub-header" style={{ textAlign: 'center' }}>SP</th>
                  <th className="sub-header" style={{ textAlign: 'center' }}>TP</th>
                  <th className="sub-header" style={{ textAlign: 'center' }}>CP</th>
                  <th className="sub-header" style={{ textAlign: 'center' }}>SP</th>
                </tr>
              </thead>
              <tbody>
                {layers.map((layer) => {
                  const layerConfig = layerConfigs[layer.name] || {
                    prefill: { tp_degree: 1, cp_degree: 1, sp_degree: 1 },
                    decode: { tp_degree: 1, cp_degree: 1, sp_degree: 1 },
                    dtype: ''
                  };
                  
                  return (
                    <tr key={layer.name}>
                      <td className="layer-type-cell">{layer.name}</td>
                      <td className="count-cell">{layer.count}</td>
                      <td className="dtype-cell">
                        <select
                          value={layerConfig.dtype}
                          onChange={(e) => onLayerConfigChange(layer.name, null, 'dtype', e.target.value)}
                          className={`input-field-sm ${!layerConfig.dtype ? 'border-red-500' : ''}`}
                          style={{ width: '90px' }}
                        >
                          <option value="">Select...</option>
                          <option value="fp32">FP32</option>
                          <option value="fp16">FP16</option>
                          <option value="bf16">BF16</option>
                          <option value="int8">INT8</option>
                          <option value="fp8">FP8</option>
                        </select>
                      </td>
                      
                      {/* Prefill parallelism */}
                      <td className="parallelism-cell">
                        <input
                          type="number"
                          min="1"
                          value={layerConfig.prefill.tp_degree}
                          onChange={(e) => onLayerConfigChange(layer.name, 'prefill', 'tp_degree', parseInt(e.target.value))}
                          className="input-field-sm"
                        />
                      </td>
                      <td className="parallelism-cell">
                        <input
                          type="number"
                          min="1"
                          value={layerConfig.prefill.cp_degree}
                          onChange={(e) => onLayerConfigChange(layer.name, 'prefill', 'cp_degree', parseInt(e.target.value))}
                          className="input-field-sm"
                        />
                      </td>
                      <td className="parallelism-cell">
                        <input
                          type="number"
                          min="1"
                          value={layerConfig.prefill.sp_degree}
                          onChange={(e) => onLayerConfigChange(layer.name, 'prefill', 'sp_degree', parseInt(e.target.value))}
                          className="input-field-sm"
                        />
                      </td>
                      
                      {/* Decode parallelism */}
                      <td className="parallelism-cell">
                        <input
                          type="number"
                          min="1"
                          value={layerConfig.decode.tp_degree}
                          onChange={(e) => onLayerConfigChange(layer.name, 'decode', 'tp_degree', parseInt(e.target.value))}
                          className="input-field-sm"
                        />
                      </td>
                      <td className="parallelism-cell">
                        <input
                          type="number"
                          min="1"
                          value={layerConfig.decode.cp_degree}
                          onChange={(e) => onLayerConfigChange(layer.name, 'decode', 'cp_degree', parseInt(e.target.value))}
                          className="input-field-sm"
                        />
                      </td>
                      <td className="parallelism-cell">
                        <input
                          type="number"
                          min="1"
                          value={layerConfig.decode.sp_degree}
                          onChange={(e) => onLayerConfigChange(layer.name, 'decode', 'sp_degree', parseInt(e.target.value))}
                          className="input-field-sm"
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            
            <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center' }}>
              <button 
                onClick={onCopyToDecodeLayers}
                className="btn btn-secondary"
                title="Copy all layer parallelism configurations from prefill to decode"
                style={{ fontSize: '0.875rem' }}
              >
                <Copy size={14} style={{ marginRight: '0.25rem' }} />
                Copy All Layer Configs: Prefill → Decode
              </button>
            </div>
          </div>
        )}
      </div>
      
      {/* Configuration Status */}
      <div style={{ 
        marginTop: '1.5rem', 
        padding: '1rem', 
        backgroundColor: '#1D2F61', 
        borderRadius: '8px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: isPrefillComplete ? '#29AF83' : '#EF4444'
            }}></div>
            <span style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
              Prefill Phase {isPrefillComplete ? '✓' : '⚠'}
            </span>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: isDecodeComplete ? '#29AF83' : '#EF4444'
            }}></div>
            <span style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
              Decode Phase {isDecodeComplete ? '✓' : '⚠'}
            </span>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: areLayersComplete ? '#29AF83' : '#EF4444'
            }}></div>
            <span style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
              Layer Configs {areLayersComplete ? '✓' : '⚠'}
            </span>
          </div>
        </div>
        
        <div style={{ 
          fontSize: '0.875rem', 
          fontWeight: '600',
          color: isFullyComplete ? '#29AF83' : '#94a3b8'
        }}>
          {isFullyComplete ? '✓ Configuration Complete' : 'Please complete all required fields'}
        </div>
      </div>
    </div>
  );
}
