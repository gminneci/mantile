import React, { useState } from 'react';
import { Layers, Copy, Check } from 'lucide-react';
import { motion } from 'framer-motion';
import { formatNumber } from '../utils/formatters';
import { CHART_COLORS } from '../utils/constants';

// Helper function to format bytes to appropriate unit
const formatBytes = (bytes) => {
  if (bytes === null || bytes === undefined || bytes === 0) return { value: '0', unit: 'B' };
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const value = parseFloat((bytes / Math.pow(k, i)).toFixed(2));
  return { value: formatNumber(value, 2), unit: sizes[i] };
};

// Format phase as P (prefill) or D (decode)
const formatPhase = (phase) => {
  if (!phase) return '';
  const p = phase.toLowerCase();
  if (p === 'prefill' || p === 'p') return 'P';
  if (p === 'decode' || p === 'd') return 'D';
  return phase.charAt(0).toUpperCase();
};

// Get layer label: "layerName P" or "layerName D"
const getLayerLabel = (layer) => {
  return `${layer.layerName} ${formatPhase(layer.phase)}`;
};

// Get colors for layers based on their configSide
const getLayerColors = (layer1, layer2) => {
  if (!layer2) {
    // Single layer - use appropriate color based on configSide
    return {
      color1: layer1.configSide === 'comparison' ? CHART_COLORS.comparison : CHART_COLORS.primary,
      label1: getLayerLabel(layer1)
    };
  }

  // Two layers - check if they're from the same configSide
  if (layer1.configSide === layer2.configSide) {
    // Same side - use light and dark shades of same color
    if (layer1.configSide === 'primary') {
      return {
        color1: CHART_COLORS.primary,
        color2: CHART_COLORS.primaryDark,
        label1: getLayerLabel(layer1),
        label2: getLayerLabel(layer2)
      };
    } else {
      return {
        color1: CHART_COLORS.comparison,
        color2: CHART_COLORS.comparisonDark,
        label1: getLayerLabel(layer1),
        label2: getLayerLabel(layer2)
      };
    }
  } else {
    // Different sides - use green for primary, orange for comparison
    if (layer1.configSide === 'primary') {
      return {
        color1: CHART_COLORS.primary,
        color2: CHART_COLORS.comparison,
        label1: getLayerLabel(layer1),
        label2: getLayerLabel(layer2)
      };
    } else {
      return {
        color1: CHART_COLORS.comparison,
        color2: CHART_COLORS.primary,
        label1: getLayerLabel(layer1),
        label2: getLayerLabel(layer2)
      };
    }
  }
};

// Single metric card with horizontal bar chart
function LayerMetricCard({ label, value1, value2, format, unit, delay, colors, isSingleLayer, isText = false }) {
  const formattedValue1 = format(value1);
  const formattedValue2 = format(value2);
  
  // For bar chart calculations
  const numValue1 = typeof value1 === 'number' ? value1 : 0;
  const numValue2 = typeof value2 === 'number' ? value2 : 0;
  const maxValue = isSingleLayer ? numValue1 : Math.max(numValue1, numValue2);
  const percent1 = maxValue > 0 ? (numValue1 / maxValue) * 100 : 0;
  const percent2 = maxValue > 0 ? (numValue2 / maxValue) * 100 : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="card metrics-card"
      style={{ 
        backgroundColor: '#e7e3da',
        border: '1px solid #E5E7EB',
        padding: '1.5rem',
        width: '100%',
        minWidth: 0,
        borderRadius: '8px'
      }}
    >
      <div className="mb-3">
        <span className="text-sm uppercase tracking-wider" style={{ color: '#6B7280' }}>{label}</span>
      </div>
      
      {isText ? (
        // Text-based metric (like Bottleneck)
        <>
          <div className="mb-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold" style={{ color: colors.color1 }}>{colors.label1}</span>
              <span className="text-lg font-bold" style={{ color: '#1F2937' }}>
                {formattedValue1}
              </span>
            </div>
          </div>
          {!isSingleLayer && (
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold" style={{ color: colors.color2 }}>{colors.label2}</span>
                <span className="text-lg font-bold" style={{ color: '#1F2937' }}>
                  {formattedValue2}
                </span>
              </div>
            </div>
          )}
        </>
      ) : (
        // Numeric metric with bar chart
        <>
          {/* First Layer Bar */}
          <div className={isSingleLayer ? 'mb-3' : 'mb-3'}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold" style={{ color: colors.color1 }}>{colors.label1}</span>
              <span className="text-lg font-bold" style={{ color: '#1F2937' }}>
                {formattedValue1}{unit && <span className="text-xs" style={{ color: '#6B7280' }}> {unit}</span>}
              </span>
            </div>
            <div style={{ width: '100%', height: '24px', backgroundColor: '#d1d5db', borderRadius: '4px', overflow: 'hidden' }}>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${percent1}%` }}
                transition={{ delay: delay + 0.2, duration: 0.6 }}
                style={{ 
                  height: '100%', 
                  backgroundColor: colors.color1,
                  borderRadius: '4px'
                }}
              />
            </div>
          </div>

          {/* Second Layer Bar (if comparison) */}
          {!isSingleLayer && (
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold" style={{ color: colors.color2 }}>{colors.label2}</span>
                <span className="text-lg font-bold" style={{ color: '#1F2937' }}>
                  {formattedValue2}{unit && <span className="text-xs" style={{ color: '#6B7280' }}> {unit}</span>}
                </span>
              </div>
              <div style={{ width: '100%', height: '24px', backgroundColor: '#d1d5db', borderRadius: '4px', overflow: 'hidden' }}>
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${percent2}%` }}
                  transition={{ delay: delay + 0.2, duration: 0.6 }}
                  style={{ 
                    height: '100%', 
                    backgroundColor: colors.color2,
                    borderRadius: '4px'
                  }}
                />
              </div>
            </div>
          )}
        </>
      )}
    </motion.div>
  );
}

// Combined timing breakdown card with horizontal bars for compute, load, and comms
function TimingBreakdownCard({ metrics1, metrics2, delay, colors, isSingleLayer }) {
  const timings = [
    { label: 'Compute', key: 'compute_time_ms' },
    { label: 'Load', key: 'load_time_ms' },
    { label: 'Communication', key: 'communication_time_ms' }
  ];

  // Get all timing values for max calculation
  const allValues = timings.flatMap(t => [
    metrics1?.[t.key] || 0,
    ...(isSingleLayer ? [] : [metrics2?.[t.key] || 0])
  ]);
  const maxValue = Math.max(...allValues, 0.001); // Avoid division by zero

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="card metrics-card"
      style={{ 
        backgroundColor: '#e7e3da',
        border: '1px solid #E5E7EB',
        padding: '1.5rem',
        width: '100%',
        minWidth: 0,
        borderRadius: '8px'
      }}
    >
      <div className="mb-4">
        <span className="text-sm uppercase tracking-wider" style={{ color: '#6B7280' }}>Timing Breakdown</span>
      </div>

      {/* Legend */}
      <div className="flex gap-4 mb-4" style={{ flexWrap: 'wrap' }}>
        <div className="flex items-center gap-2">
          <div style={{ width: '12px', height: '12px', backgroundColor: colors.color1, borderRadius: '2px' }} />
          <span className="text-xs font-semibold" style={{ color: colors.color1 }}>{colors.label1}</span>
        </div>
        {!isSingleLayer && (
          <div className="flex items-center gap-2">
            <div style={{ width: '12px', height: '12px', backgroundColor: colors.color2, borderRadius: '2px' }} />
            <span className="text-xs font-semibold" style={{ color: colors.color2 }}>{colors.label2}</span>
          </div>
        )}
      </div>

      {/* Timing rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {timings.map((timing, idx) => {
          // For communication time, treat null as 0 (no parallelism = no comms)
          const rawValue1 = metrics1?.[timing.key];
          const rawValue2 = metrics2?.[timing.key];
          const value1 = timing.key === 'communication_time_ms' && rawValue1 == null ? 0 : rawValue1;
          const value2 = timing.key === 'communication_time_ms' && rawValue2 == null ? 0 : rawValue2;
          const numValue1 = typeof value1 === 'number' ? value1 : 0;
          const numValue2 = typeof value2 === 'number' ? value2 : 0;
          const percent1 = (numValue1 / maxValue) * 100;
          const percent2 = (numValue2 / maxValue) * 100;

          return (
            <div key={timing.key}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium" style={{ color: '#4B5563' }}>{timing.label}</span>
              </div>
              
              {/* Horizontal bars */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {/* Layer 1 bar */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ flex: 1, height: '20px', backgroundColor: '#d1d5db', borderRadius: '4px', overflow: 'hidden' }}>
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${percent1}%` }}
                      transition={{ delay: delay + 0.2 + idx * 0.1, duration: 0.6 }}
                      style={{ 
                        height: '100%', 
                        backgroundColor: colors.color1,
                        borderRadius: '4px'
                      }}
                    />
                  </div>
                  <span style={{ minWidth: '70px', textAlign: 'right', fontSize: '0.75rem', fontWeight: '600', color: '#1F2937' }}>
                    {value1 != null ? `${formatNumber(value1, 3)} ms` : 'N/A'}
                  </span>
                </div>
                
                {/* Layer 2 bar (if comparison) */}
                {!isSingleLayer && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{ flex: 1, height: '20px', backgroundColor: '#d1d5db', borderRadius: '4px', overflow: 'hidden' }}>
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${percent2}%` }}
                        transition={{ delay: delay + 0.2 + idx * 0.1, duration: 0.6 }}
                        style={{ 
                          height: '100%', 
                          backgroundColor: colors.color2,
                          borderRadius: '4px'
                        }}
                      />
                    </div>
                    <span style={{ minWidth: '70px', textAlign: 'right', fontSize: '0.75rem', fontWeight: '600', color: '#1F2937' }}>
                      {value2 != null ? `${formatNumber(value2, 3)} ms` : 'N/A'}
                    </span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}

/**
 * LayerMetricsDisplay - Shows metrics for 1 or 2 loaded layers in comparison format
 * Matches the system metrics styling with cream cards and horizontal bar charts
 */
export default function LayerMetricsDisplay({ loadedLayers }) {
  const [copiedIndex, setCopiedIndex] = useState(null);

  const handleCopyDebugDetails = (layerIndex) => {
    const layer = loadedLayers[layerIndex];
    const debugData = {
      layer_name: layer.layerName,
      phase: layer.phase,
      config_side: layer.configSide,
      debug_details: layer.debug_details
    };
    navigator.clipboard.writeText(JSON.stringify(debugData, null, 2));
    setCopiedIndex(layerIndex);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  if (loadedLayers.length === 0) {
    return (
      <div style={{ width: '100%', padding: '1rem' }}>
        <div 
          style={{ 
            backgroundColor: '#e7e3da',
            border: '1px solid #E5E7EB',
            borderRadius: '12px',
            padding: '3rem',
            width: '100%',
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '200px'
          }}
        >
          <Layers size={48} style={{ color: '#6B7280', marginBottom: '1rem' }} />
          <h3 style={{ color: '#1F2937', fontSize: '1.25rem', fontWeight: '600', marginBottom: '0.75rem' }}>
            Layer Metrics View
          </h3>
          <p style={{ color: '#6B7280', lineHeight: '1.6', maxWidth: '500px' }}>
            Select layer panels from the sidebars and drop them here to view detailed layer-level metrics.
            Compare up to 2 layers side-by-side.
          </p>
        </div>
      </div>
    );
  }

  // Extract layers and metrics
  const layer1 = loadedLayers[0];
  const layer2 = loadedLayers[1];
  const metrics1 = layer1?.metrics;
  const metrics2 = layer2?.metrics;
  const isSingleLayer = loadedLayers.length === 1;

  // Get colors based on layer sides
  const colors = getLayerColors(layer1, layer2);

  // Metric definitions
  const metricDefs = [
    {
      label: 'Weight Size',
      getValue: (m) => m?.weight_memory_per_package,
      format: (v) => {
        const { value, unit } = formatBytes(v);
        return `${value} ${unit}`;
      },
      unit: '',
      formatRaw: (v) => formatBytes(v)
    },
    {
      label: 'Activation Size',
      getValue: (m) => m?.activation_memory_per_package,
      format: (v) => {
        const { value, unit } = formatBytes(v);
        return `${value} ${unit}`;
      },
      unit: '',
      formatRaw: (v) => formatBytes(v)
    },
    {
      label: 'KV Cache Size',
      getValue: (m) => m?.kv_cache_per_package,
      format: (v) => {
        const { value, unit } = formatBytes(v);
        return `${value} ${unit}`;
      },
      unit: '',
      formatRaw: (v) => formatBytes(v)
    },
    {
      label: 'Wall Clock Time',
      getValue: (m) => m?.wall_clock_time_ms,
      format: (v) => v != null ? formatNumber(v, 3) : 'N/A',
      unit: 'ms'
    },
    {
      label: 'Bottleneck',
      getValue: (m) => m?.bottleneck,
      format: (v) => v || 'N/A',
      unit: '',
      isText: true
    }
  ];

  return (
    <div style={{ width: '100%' }}>
      {/* Metrics grid */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
        gap: '1rem' 
      }}>
        {metricDefs.map((metric, idx) => (
          <LayerMetricCard
            key={metric.label}
            label={metric.label}
            value1={metric.getValue(metrics1)}
            value2={metric.getValue(metrics2)}
            format={metric.format}
            unit={metric.unit}
            delay={0.1 + idx * 0.05}
            colors={colors}
            isSingleLayer={isSingleLayer}
            isText={metric.isText}
          />
        ))}
        
        {/* Combined Timing Breakdown Card */}
        <TimingBreakdownCard
          metrics1={metrics1}
          metrics2={metrics2}
          delay={0.1 + metricDefs.length * 0.05}
          colors={colors}
          isSingleLayer={isSingleLayer}
        />
      </div>

      {/* Debug details section */}
      {loadedLayers.some(l => l.debug_details) && (
        <details className="mt-4" style={{ paddingLeft: '1.4rem', paddingRight: '1.4rem' }}>
          <summary className="text-xs cursor-pointer" style={{ color: '#6B7280' }}>
            Debug details
          </summary>
          <div className="mt-2 p-3 rounded" style={{ backgroundColor: '#1F2937' }}>
            {loadedLayers.map((layer, idx) => {
              const debugDetails = layer.debug_details;
              if (!debugDetails) return null;

              // Extract prompt and formulas for display purposes only
              const { prompt, formulas, ...restDetails } = debugDetails;
              const isCopied = copiedIndex === idx;

              return (
                <div key={idx} className={idx > 0 ? 'mt-4 pt-4' : ''} style={{ 
                  borderTop: idx > 0 ? '1px solid #374151' : 'none',
                  position: 'relative',
                  paddingRight: '5rem'
                }}>
                  <button
                    onClick={() => handleCopyDebugDetails(idx)}
                    style={{
                      position: 'absolute',
                      top: idx > 0 ? '1.75rem' : '0.75rem',
                      right: '0',
                      padding: '0.5rem',
                      backgroundColor: isCopied ? '#10B981' : '#374151',
                      borderRadius: '0.375rem',
                      border: 'none',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => {
                      if (!isCopied) e.currentTarget.style.backgroundColor = '#4B5563';
                    }}
                    onMouseLeave={(e) => {
                      if (!isCopied) e.currentTarget.style.backgroundColor = '#374151';
                    }}
                  >
                    {isCopied ? (
                      <Check size={16} style={{ color: '#FFFFFF' }} />
                    ) : (
                      <Copy size={16} style={{ color: '#9CA3AF' }} />
                    )}
                    <span style={{ fontSize: '0.75rem', color: isCopied ? '#FFFFFF' : '#9CA3AF' }}>
                      {isCopied ? 'Copied!' : 'Copy'}
                    </span>
                  </button>

                  <span className="text-xs font-semibold" style={{ 
                    color: layer.selectionColor || '#10B981' 
                  }}>
                    {layer.layerName} ({layer.phase})
                  </span>

                  {/* System Prompt at the top in plain text */}
                  {prompt && (
                    <div className="mt-2 mb-3 p-2 rounded" style={{ 
                      backgroundColor: '#111827',
                      borderLeft: '3px solid ' + (layer.selectionColor || '#10B981')
                    }}>
                      <p className="text-xs" style={{ 
                        color: '#D1D5DB',
                        lineHeight: '1.5',
                        whiteSpace: 'pre-wrap'
                      }}>
                        {prompt}
                      </p>
                    </div>
                  )}

                  {/* Debug details JSON (excluding formulas for display) */}
                  <pre className="text-xs mt-1 overflow-auto max-h-64" style={{ color: '#D1D5DB' }}>
                    {JSON.stringify(restDetails, null, 2)}
                  </pre>
                </div>
              );
            })}
          </div>
        </details>
      )}
    </div>
  );
}
