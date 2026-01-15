import React from 'react';
import { Activity, Server, Zap, Database, Clock } from 'lucide-react';
import { motion } from 'framer-motion';

function StatCard({ label, value, unit, icon: Icon, delay, highlight = false, comparisonValue = null, comparisonUnit = null }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="card metrics-card flex flex-col gap-2"
      style={{ 
        backgroundColor: '#e7e3da',
        borderLeft: highlight ? '4px solid #29AF83' : 'none',
        border: '1px solid #E5E7EB',
        padding: '1.5rem',
        width: '100%',
        minWidth: 0
      }}
    >
      <div className="flex justify-between items-start">
        <span className="text-sm uppercase tracking-wider" style={{ color: '#6B7280' }}>{label}</span>
        {Icon && <Icon size={18} style={{ color: '#3B82F6', opacity: 0.7 }} />}
      </div>
      
      {comparisonValue !== null ? (
        // Comparison mode: show both values
        <div className="flex flex-col gap-2">
          <div className="flex items-baseline gap-2">
            <span className="text-xs" style={{ color: '#10B981', fontWeight: '600', minWidth: '70px' }}>Primary:</span>
            <span className="text-2xl font-bold" style={{ color: '#1F2937' }}>
              {value}
            </span>
            {unit && <span className="text-sm" style={{ color: '#6B7280', opacity: 0.85 }}>{unit}</span>}
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-xs" style={{ color: '#f96c56', fontWeight: '600', minWidth: '70px' }}>Comparison:</span>
            <span className="text-2xl font-bold" style={{ color: '#1F2937' }}>
              {comparisonValue}
            </span>
            {comparisonUnit && <span className="text-sm" style={{ color: '#6B7280', opacity: 0.85 }}>{comparisonUnit}</span>}
          </div>
        </div>
      ) : (
        // Normal mode: single value
        <div className="flex items-baseline gap-1">
          <span className={`${highlight ? 'text-4xl' : 'text-3xl'} font-bold`} style={{ color: '#1F2937' }}>
            {value}
          </span>
          {unit && <span className="text-sm" style={{ color: '#6B7280', opacity: 0.85 }}>{unit}</span>}
        </div>
      )}
    </motion.div>
  );
}

/**
 * MetricsDisplay - Shows system, prefill, and decode metrics
 * Displays three-level hierarchy: System > Prefill/Decode
 * Supports comparison mode with overlay of two datasets
 */
export default function MetricsDisplay({ metrics, comparisonMetrics = null }) {
  if (!metrics) return null;

  const { system, prefill, decode } = metrics;
  const comparisonSystem = comparisonMetrics?.system;
  const comparisonPrefill = comparisonMetrics?.prefill;
  const comparisonDecode = comparisonMetrics?.decode;

  return (
    <div style={{ paddingTop: '1.05rem', width: '100%' }}>
      {/* System-Level Metrics (Most Prominent) */}
      {system && (
        <div className="mb-8" style={{ width: '100%' }}>
          <div className="flex items-center justify-center gap-3" style={{ marginBottom: '1.75rem', paddingLeft: '1.4rem', paddingRight: '1.4rem' }}>
            <Zap size={24} className="text-accent" />
            <h2 className="text-2xl font-bold text-white">System Metrics</h2>
          </div>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '1.05rem',
            width: '100%',
            paddingLeft: '1.4rem',
            paddingRight: '1.4rem'
          }}>
            <StatCard
              label="Total Time"
              value={system.total_time_ms?.toFixed(2) || 'N/A'}
              unit="ms"
              icon={Clock}
              delay={0.1}
              highlight={true}
              comparisonValue={comparisonSystem ? (comparisonSystem.total_time_ms?.toFixed(2) || 'N/A') : null}
              comparisonUnit={comparisonSystem ? "ms" : null}
            />
            <StatCard
              label="Throughput"
              value={system.throughput_tokens_s?.toFixed(1) || 'N/A'}
              unit="tok/s"
              icon={Activity}
              delay={0.2}
              highlight={true}
              comparisonValue={comparisonSystem ? (comparisonSystem.throughput_tokens_s?.toFixed(1) || 'N/A') : null}
              comparisonUnit={comparisonSystem ? "tok/s" : null}
            />
            <StatCard
              label="Total Memory"
              value={system.total_memory_gb?.toFixed(1) || 'N/A'}
              unit="GB"
              icon={Database}
              delay={0.3}
              highlight={true}
              comparisonValue={comparisonSystem ? (comparisonSystem.total_memory_gb?.toFixed(1) || 'N/A') : null}
              comparisonUnit={comparisonSystem ? "GB" : null}
            />
            <StatCard
              label="Utilization"
              value={system.utilization_percent?.toFixed(1) || 'N/A'}
              unit="%"
              icon={Server}
              delay={0.4}
              highlight={true}
              comparisonValue={comparisonSystem ? (comparisonSystem.utilization_percent?.toFixed(1) || 'N/A') : null}
              comparisonUnit={comparisonSystem ? "%" : null}
            />
          </div>
        </div>
      )}

      {/* Phase-Specific Metrics */}
      <div className="grid grid-cols-2 gap-6">
        {/* Prefill Phase Metrics */}
        {prefill && (
          <div>
            <div className="flex items-center gap-2 mb-6">
              <span style={{ fontSize: '20px' }}>🔹</span>
              <h3 className="text-xl font-bold text-white">Prefill Phase</h3>
            </div>
            <div className="card" style={{ backgroundColor: '#e7e3da', borderLeft: '3px solid #3B82F6', border: '1px solid #E5E7EB' }}>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-xs uppercase tracking-wider mb-1" style={{ color: '#6B7280' }}>Latency</div>
                  <div className="text-2xl font-bold" style={{ color: '#1F2937' }}>
                    {prefill.latency_ms?.toFixed(2) || 'N/A'}
                    <span className="text-sm ml-1" style={{ color: '#6B7280' }}>ms</span>
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wider mb-1" style={{ color: '#6B7280' }}>Memory</div>
                  <div className="text-2xl font-bold" style={{ color: '#1F2937' }}>
                    {prefill.memory_gb?.toFixed(2) || 'N/A'}
                    <span className="text-sm ml-1" style={{ color: '#6B7280' }}>GB</span>
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wider mb-1" style={{ color: '#6B7280' }}>FLOPs</div>
                  <div className="text-2xl font-bold" style={{ color: '#1F2937' }}>
                    {prefill.flops?.toFixed(1) || 'N/A'}
                    <span className="text-sm ml-1" style={{ color: '#6B7280' }}>TF</span>
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wider mb-1" style={{ color: '#6B7280' }}>Comm Time</div>
                  <div className="text-2xl font-bold" style={{ color: '#1F2937' }}>
                    {prefill.communication_time_ms?.toFixed(2) || 'N/A'}
                    <span className="text-sm ml-1" style={{ color: '#6B7280' }}>ms</span>
                  </div>
                </div>
              </div>
              
              {/* Show full prefill data */}
              <details className="mt-4">
                <summary className="text-dim text-xs cursor-pointer hover:text-white">
                  View detailed metrics
                </summary>
                <pre className="text-xs text-white bg-surface p-3 rounded mt-2 overflow-auto max-h-64">
                  {JSON.stringify(prefill, null, 2)}
                </pre>
              </details>
            </div>
          </div>
        )}

        {/* Decode Phase Metrics */}
        {decode && (
          <div>
            <div className="flex items-center gap-2 mb-6">
              <span style={{ fontSize: '20px' }}>🔸</span>
              <h3 className="text-xl font-bold text-white">Decode Phase</h3>
            </div>
            <div className="card" style={{ backgroundColor: '#e7e3da', borderLeft: '3px solid #10B981', border: '1px solid #E5E7EB' }}>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-xs uppercase tracking-wider mb-1" style={{ color: '#6B7280' }}>Latency</div>
                  <div className="text-2xl font-bold" style={{ color: '#1F2937' }}>
                    {decode.latency_ms?.toFixed(2) || 'N/A'}
                    <span className="text-sm ml-1" style={{ color: '#6B7280' }}>ms</span>
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wider mb-1" style={{ color: '#6B7280' }}>Memory</div>
                  <div className="text-2xl font-bold" style={{ color: '#1F2937' }}>
                    {decode.memory_gb?.toFixed(2) || 'N/A'}
                    <span className="text-sm ml-1" style={{ color: '#6B7280' }}>GB</span>
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wider mb-1" style={{ color: '#6B7280' }}>FLOPs</div>
                  <div className="text-2xl font-bold" style={{ color: '#1F2937' }}>
                    {decode.flops?.toFixed(1) || 'N/A'}
                    <span className="text-sm ml-1" style={{ color: '#6B7280' }}>TF</span>
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wider mb-1" style={{ color: '#6B7280' }}>Comm Time</div>
                  <div className="text-2xl font-bold" style={{ color: '#1F2937' }}>
                    {decode.communication_time_ms?.toFixed(2) || 'N/A'}
                    <span className="text-sm ml-1" style={{ color: '#6B7280' }}>ms</span>
                  </div>
                </div>
              </div>
              
              {/* Show full decode data */}
              <details className="mt-4">
                <summary className="text-dim text-xs cursor-pointer hover:text-white">
                  View detailed metrics
                </summary>
                <pre className="text-xs text-white bg-surface p-3 rounded mt-2 overflow-auto max-h-64">
                  {JSON.stringify(decode, null, 2)}
                </pre>
              </details>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
