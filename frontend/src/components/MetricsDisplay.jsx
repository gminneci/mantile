import React from 'react';
import { Activity, Server, Zap, Database, Clock } from 'lucide-react';
import { motion } from 'framer-motion';

function StatCard({ label, value, unit, icon: Icon, delay, highlight = false }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="card metrics-card flex flex-col gap-2"
      style={{ 
        backgroundColor: highlight ? '#1A2847' : '#0F1729',
        borderLeft: highlight ? '4px solid #29AF83' : 'none'
      }}
    >
      <div className="flex justify-between items-start">
        <span className="text-dark text-sm uppercase tracking-wider">{label}</span>
        {Icon && <Icon size={18} className="text-highlight-blue" style={{ opacity: 0.7 }} />}
      </div>
      <div className="flex items-baseline gap-1">
        <span className={`${highlight ? 'text-4xl' : 'text-3xl'} font-bold text-white`}>
          {value}
        </span>
        {unit && <span className="text-dark text-sm" style={{ opacity: 0.85 }}>{unit}</span>}
      </div>
    </motion.div>
  );
}

/**
 * MetricsDisplay - Shows system, prefill, and decode metrics
 * Displays three-level hierarchy: System > Prefill/Decode
 */
export default function MetricsDisplay({ metrics }) {
  if (!metrics) return null;

  const { system, prefill, decode } = metrics;

  return (
    <div>
      {/* System-Level Metrics (Most Prominent) */}
      {system && (
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <Zap size={24} className="text-accent" />
            <h2 className="text-2xl font-bold text-white">System Metrics</h2>
          </div>
          <p className="text-dim text-sm mb-4">
            Aggregated performance across both prefill and decode phases
          </p>
          <div className="grid grid-cols-4 gap-4">
            <StatCard
              label="Total Time"
              value={system.total_time_ms?.toFixed(2) || 'N/A'}
              unit="ms"
              icon={Clock}
              delay={0.1}
              highlight={true}
            />
            <StatCard
              label="Throughput"
              value={system.throughput_tokens_s?.toFixed(1) || 'N/A'}
              unit="tok/s"
              icon={Activity}
              delay={0.2}
              highlight={true}
            />
            <StatCard
              label="Total Memory"
              value={system.total_memory_gb?.toFixed(1) || 'N/A'}
              unit="GB"
              icon={Database}
              delay={0.3}
              highlight={true}
            />
            <StatCard
              label="Utilization"
              value={system.utilization_percent?.toFixed(1) || 'N/A'}
              unit="%"
              icon={Server}
              delay={0.4}
              highlight={true}
            />
          </div>
        </div>
      )}

      {/* Phase-Specific Metrics */}
      <div className="grid grid-cols-2 gap-6">
        {/* Prefill Phase Metrics */}
        {prefill && (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <span style={{ fontSize: '20px' }}>🔹</span>
              <h3 className="text-xl font-bold text-white">Prefill Phase</h3>
            </div>
            <div className="card" style={{ backgroundColor: '#1A2847', borderLeft: '3px solid #3B82F6' }}>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-dim text-xs uppercase tracking-wider mb-1">Latency</div>
                  <div className="text-white text-2xl font-bold">
                    {prefill.latency_ms?.toFixed(2) || 'N/A'}
                    <span className="text-sm text-dim ml-1">ms</span>
                  </div>
                </div>
                <div>
                  <div className="text-dim text-xs uppercase tracking-wider mb-1">Memory</div>
                  <div className="text-white text-2xl font-bold">
                    {prefill.memory_gb?.toFixed(2) || 'N/A'}
                    <span className="text-sm text-dim ml-1">GB</span>
                  </div>
                </div>
                <div>
                  <div className="text-dim text-xs uppercase tracking-wider mb-1">FLOPs</div>
                  <div className="text-white text-2xl font-bold">
                    {prefill.flops?.toFixed(1) || 'N/A'}
                    <span className="text-sm text-dim ml-1">TF</span>
                  </div>
                </div>
                <div>
                  <div className="text-dim text-xs uppercase tracking-wider mb-1">Comm Time</div>
                  <div className="text-white text-2xl font-bold">
                    {prefill.communication_time_ms?.toFixed(2) || 'N/A'}
                    <span className="text-sm text-dim ml-1">ms</span>
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
            <div className="flex items-center gap-2 mb-4">
              <span style={{ fontSize: '20px' }}>🔸</span>
              <h3 className="text-xl font-bold text-white">Decode Phase</h3>
            </div>
            <div className="card" style={{ backgroundColor: '#1A2847', borderLeft: '3px solid #10B981' }}>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-dim text-xs uppercase tracking-wider mb-1">Latency</div>
                  <div className="text-white text-2xl font-bold">
                    {decode.latency_ms?.toFixed(2) || 'N/A'}
                    <span className="text-sm text-dim ml-1">ms</span>
                  </div>
                </div>
                <div>
                  <div className="text-dim text-xs uppercase tracking-wider mb-1">Memory</div>
                  <div className="text-white text-2xl font-bold">
                    {decode.memory_gb?.toFixed(2) || 'N/A'}
                    <span className="text-sm text-dim ml-1">GB</span>
                  </div>
                </div>
                <div>
                  <div className="text-dim text-xs uppercase tracking-wider mb-1">FLOPs</div>
                  <div className="text-white text-2xl font-bold">
                    {decode.flops?.toFixed(1) || 'N/A'}
                    <span className="text-sm text-dim ml-1">TF</span>
                  </div>
                </div>
                <div>
                  <div className="text-dim text-xs uppercase tracking-wider mb-1">Comm Time</div>
                  <div className="text-white text-2xl font-bold">
                    {decode.communication_time_ms?.toFixed(2) || 'N/A'}
                    <span className="text-sm text-dim ml-1">ms</span>
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
