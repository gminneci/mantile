import React from 'react';
import { Server, Cpu, Copy } from 'lucide-react';

/**
 * PhaseConfigPanel - Reusable component for configuring prefill or decode phase
 * Each phase has its own hardware, batch size, seq length
 * Layer configs are shown in a compact table format (per-layer-index, not per-type)
 */
export default function PhaseConfigPanel({
  phase, // 'prefill' or 'decode'
  hardware,
  batchSize,
  seqLen,
  layerConfigs,
  layers,
  sharedDtype,
  modelId,
  onHardwareChange,
  onBatchSizeChange,
  onSeqLenChange,
  onLayerConfigChange,
  onCopyFromPrefill, // Only for decode phase
  hardwareOptions = [
    { value: 'nvidia_gb200_single', label: 'NVIDIA GB200 (Single Package)' },
    { value: 'nvidia_nvl72_rack', label: 'NVIDIA NVL-72 (Full Rack)' },
    { value: 'nvidia_h100_80gb', label: 'NVIDIA H100 80GB' }
  ]
}) {
  const isPrefill = phase === 'prefill';
  const phaseColor = isPrefill ? '#3B82F6' : '#10B981';
  const phaseIcon = isPrefill ? '🔹' : '🔸';
  const phaseLabel = isPrefill ? 'PREFILL PHASE' : 'DECODE PHASE';
  
  // Expand layers by count to get individual layer instances
  const expandedLayers = [];
  let layerIndex = 0;
  layers.forEach(layerType => {
    for (let i = 0; i < layerType.count; i++) {
      expandedLayers.push({
        index: layerIndex++,
        type: layerType.type,
        available_parallelism: layerType.available_parallelism || []
      });
    }
  });
  
  // Check if all required fields are filled
  const isComplete = hardware && batchSize !== null && seqLen !== null && 
                     Object.keys(layerConfigs || {}).length === expandedLayers.length;

  const getLayerConfig = (layerIdx) => {
    return layerConfigs?.[layerIdx] || {
      tp_degree: 1,
      cp_degree: 1,
      sp_degree: 1
    };
  };

  const updateLayerConfig = (layerIdx, field, value) => {
    onLayerConfigChange(phase, layerIdx, field, value);
  };

  return (
    <div style={{ marginBottom: '1.5rem' }}>
      {/* Phase Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '1rem',
        backgroundColor: '#1A2847',
        borderLeft: `4px solid ${phaseColor}`,
        borderRadius: '8px',
        marginBottom: '1rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ fontSize: '24px' }}>{phaseIcon}</span>
          <h3 className="text-white font-semibold text-lg">{phaseLabel}</h3>
        </div>
        {!isPrefill && onCopyFromPrefill && (
          <button
            onClick={onCopyFromPrefill}
            className="btn btn-secondary text-xs"
            style={{ padding: '0.5rem 1rem', fontSize: '0.75rem' }}
            title="Copy configuration from prefill phase"
          >
            <Copy size={14} />
            Copy from Prefill
          </button>
        )}
      </div>

      {/* Phase-Level Config (Hardware, Batch, Seq Len) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr 1fr',
        gap: '1rem',
        marginBottom: '1rem'
      }}>
        <div>
          <label className="label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Server size={14} />
            Hardware
            {!hardware && <span className="text-red-500 text-xs">*</span>}
          </label>
          <select
            className={`input-field ${!hardware ? 'border-red-500' : ''}`}
            value={hardware || ''}
            onChange={(e) => onHardwareChange(e.target.value)}
            style={{ fontSize: '0.75rem', padding: '0.5rem' }}
          >
            <option value="">Select...</option>
            {hardwareOptions.map(hw => (
              <option key={hw.value} value={hw.value}>{hw.label}</option>
            ))}
          </select>
        </div>
        
        <div>
          <label className="label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Cpu size={14} />
            Batch Size
            {batchSize === null && <span className="text-red-500 text-xs">*</span>}
          </label>
          <input
            type="number"
            className={`input-field ${batchSize === null ? 'border-red-500' : ''}`}
            placeholder={isPrefill ? '1' : '1'}
            value={batchSize === null ? '' : batchSize}
            onChange={(e) => onBatchSizeChange(parseInt(e.target.value) || null)}
            min="1"
            style={{ fontSize: '0.75rem', padding: '0.5rem' }}
          />
        </div>
        
        <div>
          <label className="label">
            Seq Length
            {seqLen === null && <span className="text-red-500 text-xs">*</span>}
          </label>
          <input
            type="number"
            className={`input-field ${seqLen === null ? 'border-red-500' : ''}`}
            placeholder={isPrefill ? '8192' : '256'}
            value={seqLen === null ? '' : seqLen}
            onChange={(e) => onSeqLenChange(parseInt(e.target.value) || null)}
            min="1"
            style={{ fontSize: '0.75rem', padding: '0.5rem' }}
          />
        </div>
      </div>

      {/* Layer Configuration Table */}
      {expandedLayers.length > 0 && (
        <div style={{
          backgroundColor: '#0F1729',
          borderRadius: '8px',
          padding: '1rem',
          maxHeight: '400px',
          overflowY: 'auto'
        }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '60px 120px 80px 80px 80px',
            gap: '0.5rem',
            padding: '0.5rem',
            borderBottom: '1px solid #526497',
            marginBottom: '0.5rem',
            position: 'sticky',
            top: 0,
            backgroundColor: '#0F1729',
            zIndex: 1
          }}>
            <div className="text-dim text-xs font-semibold">Layer</div>
            <div className="text-dim text-xs font-semibold">Type</div>
            <div className="text-dim text-xs font-semibold">TP</div>
            <div className="text-dim text-xs font-semibold">CP</div>
            <div className="text-dim text-xs font-semibold">SP</div>
          </div>
          
          {expandedLayers.map(layer => {
            const config = getLayerConfig(layer.index);
            const hasTP = layer.available_parallelism.includes('tensor_parallel');
            const hasCP = layer.available_parallelism.includes('context_parallel');
            const hasSP = layer.available_parallelism.includes('sequence_parallel');
            
            return (
              <div
                key={layer.index}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '60px 120px 80px 80px 80px',
                  gap: '0.5rem',
                  padding: '0.5rem',
                  alignItems: 'center',
                  borderBottom: '1px solid #1A2847'
                }}
              >
                <div className="text-white text-sm font-mono">{layer.index}</div>
                <div className="text-dim text-xs capitalize">{layer.type}</div>
                
                {/* TP Degree */}
                <div>
                  {hasTP ? (
                    <input
                      type="number"
                      min="1"
                      max="8"
                      value={config.tp_degree}
                      onChange={(e) => updateLayerConfig(layer.index, 'tp_degree', parseInt(e.target.value) || 1)}
                      className="input-field"
                      style={{ 
                        fontSize: '0.75rem', 
                        padding: '0.25rem 0.5rem',
                        textAlign: 'center'
                      }}
                    />
                  ) : (
                    <span className="text-dim text-xs">-</span>
                  )}
                </div>
                
                {/* CP Degree */}
                <div>
                  {hasCP ? (
                    <input
                      type="number"
                      min="1"
                      max="8"
                      value={config.cp_degree}
                      onChange={(e) => updateLayerConfig(layer.index, 'cp_degree', parseInt(e.target.value) || 1)}
                      className="input-field"
                      style={{ 
                        fontSize: '0.75rem', 
                        padding: '0.25rem 0.5rem',
                        textAlign: 'center'
                      }}
                    />
                  ) : (
                    <span className="text-dim text-xs">-</span>
                  )}
                </div>
                
                {/* SP Degree */}
                <div>
                  {hasSP ? (
                    <input
                      type="number"
                      min="1"
                      max="8"
                      value={config.sp_degree}
                      onChange={(e) => updateLayerConfig(layer.index, 'sp_degree', parseInt(e.target.value) || 1)}
                      className="input-field"
                      style={{ 
                        fontSize: '0.75rem', 
                        padding: '0.25rem 0.5rem',
                        textAlign: 'center'
                      }}
                    />
                  ) : (
                    <span className="text-dim text-xs">-</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Status indicator */}
      <div style={{ 
        marginTop: '0.75rem', 
        paddingTop: '0.75rem', 
        borderTop: '1px solid #1A2847',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem'
      }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: isComplete ? '#29AF83' : '#EF4444'
        }}></div>
        <span className="text-dim text-xs">
          {isComplete 
            ? `${expandedLayers.length} layers configured` 
            : 'Configure all required fields'}
        </span>
      </div>
    </div>
  );
}

