import React from 'react';
import { Box } from 'lucide-react';

/**
 * SharedConfigPanel - Displays shared configuration (model and dtype only)
 * These settings apply to both prefill and decode phases
 */
export default function SharedConfigPanel({ 
  model, 
  dtype, 
  availableDtypes,
  onModelChange, 
  onDtypeChange,
  models = [
    { value: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', label: 'TinyLlama 1.1B' },
    { value: 'meta-llama/Llama-3.3-70B-Instruct', label: 'Llama 3.3 70B' }
  ]
}) {
  return (
    <div className="card mb-6" style={{ backgroundColor: '#1A2847' }}>
      <div className="flex items-center gap-3 mb-4">
        <div className="w-8 h-8 rounded bg-accent flex items-center justify-center">
          <Box size={20} className="text-bg" />
        </div>
        <h2 className="text-xl font-bold text-white">Shared Configuration</h2>
      </div>
      
      <p className="text-dim text-sm mb-4">
        These settings apply to both prefill and decode phases
      </p>

      {/* Model Selection */}
      <div className="mb-4">
        <label className="label">Model Architecture</label>
        <select
          className="input-field"
          value={model}
          onChange={(e) => onModelChange(e.target.value)}
        >
          {models.map(m => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>
      </div>

      {/* Dtype Selection */}
      <div>
        <label className="label">Data Type (Precision)</label>
        <select
          className="input-field"
          value={dtype}
          onChange={(e) => onDtypeChange(e.target.value)}
        >
          {availableDtypes.map(dt => (
            <option key={dt} value={dt}>{dt.toUpperCase()}</option>
          ))}
        </select>
      </div>
    </div>
  );
}
