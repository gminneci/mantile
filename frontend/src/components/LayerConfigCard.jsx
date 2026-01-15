import React from 'react';
import { CheckCircle } from 'lucide-react';
import { motion } from 'framer-motion';

/**
 * LayerConfigCard - Individual layer configuration panel
 * Shows parallelism sliders and dtype selector for a single layer type
 */
export default function LayerConfigCard({ 
  layer, 
  config, 
  onConfigChange,
  phase = 'prefill', // 'prefill' or 'decode'
  accentColor = '#29AF83' // Default teal, can be overridden with orange
}) {
  const availableStrategies = layer.available_parallelism || [];
  const maxChips = 8; // Configurable max
  const phaseConfig = config?.[phase] || { tp_degree: 1, cp_degree: 1, sp_degree: 1 };
  const dtype = config?.dtype || '';

  const strategyMapping = {
    'tensor_parallel': { key: 'tp_degree', label: 'Tensor Parallel' },
    'context_parallel': { key: 'cp_degree', label: 'Context Parallel' },
    'sequence_parallel': { key: 'sp_degree', label: 'Sequence Parallel' }
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="card"
      style={{ backgroundColor: '#0F1729', border: '1px solid #526497', borderRadius: '8px', padding: '1rem', marginBottom: '0.75rem' }}
    >
      <div className="flex justify-between items-start mb-3">
        <div>
          <h4 className="text-white font-semibold capitalize flex items-baseline gap-2">
            {layer.name} 
            <span className="text-dim text-sm font-mono">({layer.count}x)</span>
          </h4>
        </div>
        {config && dtype && <CheckCircle size={18} style={{ color: accentColor }} />}
      </div>

      {availableStrategies.length > 0 ? (
        <div className="flex flex-col gap-3 mt-3">
          {availableStrategies.map(strategy => {
            const mapping = strategyMapping[strategy];
            if (!mapping) return null;
            
            return (
              <div key={strategy}>
                <label className="text-dim uppercase tracking-wider" style={{ fontSize: '0.7rem' }}>
                  {mapping.label}
                </label>
                <div className="flex items-center gap-3 mt-1">
                  <input
                    type="range"
                    min="1"
                    max={maxChips}
                    value={phaseConfig[mapping.key]}
                    onChange={(e) => onConfigChange(layer.name, phase, mapping.key, parseInt(e.target.value))}
                    className="flex-1 w-full h-2 rounded-lg appearance-none cursor-pointer"
                    style={{ 
                      accentColor: accentColor,
                      backgroundColor: '#1D2F61'
                    }}
                  />
                  <span className="font-mono text-sm w-8 text-right" style={{ color: accentColor }}>
                    {phaseConfig[mapping.key]}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-dim text-sm italic">Replicated (no parallelism)</div>
      )}

      {/* Data Type Selection */}
      <div className="flex items-center gap-3 mt-3 pt-3 border-t border-surface" style={{ borderColor: '#526497' }}>
        <label className="text-dim uppercase tracking-wider whitespace-nowrap" style={{ fontSize: '0.7rem' }}>Data Type</label>
        <select
          className="input-field text-xs"
          style={{ 
            padding: '0.35rem 0.5rem', 
            minWidth: '80px', 
            width: 'auto',
            fontSize: '0.75rem'
          }}
          value={dtype}
          onChange={(e) => onConfigChange(layer.name, null, 'dtype', e.target.value)}
        >
          <option value="">Select...</option>
          <option value="fp32">FP32</option>
          <option value="fp16">FP16</option>
          <option value="bf16">BF16</option>
          <option value="fp8">FP8</option>
          <option value="int8">INT8</option>
        </select>
      </div>
    </motion.div>
  );
}
