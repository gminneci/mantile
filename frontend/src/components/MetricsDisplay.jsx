import React, { useState } from 'react';
import { Activity, Server, Zap, Database, Clock, Layers, Cpu, HardDrive, CheckCircle, AlertTriangle, DollarSign } from 'lucide-react';
import { motion } from 'framer-motion';

// Helper function to format numbers with comma separators and fixed decimals
const formatNumber = (num, decimals = 2) => {
  if (num === null || num === undefined) return 'N/A';
  const fixed = num.toFixed(decimals);
  const [integer, decimal] = fixed.split('.');
  const formattedInteger = integer.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return decimal !== undefined ? `${formattedInteger}.${decimal}` : formattedInteger;
};

function StackedBarChart({ label, primaryMemory, comparisonMemory = null, delay }) {
  // Memory breakdown: weights, activations, KV cache
  const primaryTotal = (primaryMemory.weight_memory_gb || 0) + 
                       (primaryMemory.activation_memory_gb || 0) + 
                       (primaryMemory.kv_cache_gb || 0);
  
  const hasComparison = comparisonMemory !== null;
  
  let comparisonTotal = 0;
  if (hasComparison) {
    comparisonTotal = (comparisonMemory.weight_memory_gb || 0) + 
                      (comparisonMemory.activation_memory_gb || 0) + 
                      (comparisonMemory.kv_cache_gb || 0);
  }
  
  // Calculate max total for proportional sizing
  const maxTotal = hasComparison ? Math.max(primaryTotal, comparisonTotal) : primaryTotal;
  
  // Calculate percentages relative to their own totals for internal stacking
  const primaryWeightPercent = primaryTotal > 0 ? ((primaryMemory.weight_memory_gb || 0) / primaryTotal) * 100 : 0;
  const primaryActivationPercent = primaryTotal > 0 ? ((primaryMemory.activation_memory_gb || 0) / primaryTotal) * 100 : 0;
  const primaryKvPercent = primaryTotal > 0 ? ((primaryMemory.kv_cache_gb || 0) / primaryTotal) * 100 : 0;

  // Calculate bar widths relative to max total (for proportional comparison)
  const primaryBarWidth = (primaryTotal / maxTotal) * 100;
  
  let comparisonWeightPercent = 0, comparisonActivationPercent = 0, comparisonKvPercent = 0, comparisonBarWidth = 0;
  if (hasComparison) {
    comparisonWeightPercent = comparisonTotal > 0 ? ((comparisonMemory.weight_memory_gb || 0) / comparisonTotal) * 100 : 0;
    comparisonActivationPercent = comparisonTotal > 0 ? ((comparisonMemory.activation_memory_gb || 0) / comparisonTotal) * 100 : 0;
    comparisonKvPercent = comparisonTotal > 0 ? ((comparisonMemory.kv_cache_gb || 0) / comparisonTotal) * 100 : 0;
    comparisonBarWidth = (comparisonTotal / maxTotal) * 100;
  }

  // Color shades - green palette
  const weightColor = '#059669';      // Darker green (emerald-600)
  const activationColor = '#10B981';  // Medium green (emerald-500)
  const kvColor = '#6EE7B7';          // Lighter green (emerald-300)
  
  // Comparison colors - orange palette
  const compWeightColor = '#ea580c';   // Darker orange
  const compActivationColor = '#f96c56'; // Medium orange
  const compKvColor = '#fb923c';       // Lighter orange

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
        minWidth: 0
      }}
    >
      <div className="mb-3">
        <span className="text-sm uppercase tracking-wider" style={{ color: '#6B7280' }}>{label}</span>
      </div>
      
      {/* Primary System Stacked Bar */}
      <div className={hasComparison ? 'mb-4' : 'mb-3'}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold" style={{ color: hasComparison ? '#10B981' : '#059669' }}>
            {hasComparison ? 'Primary' : 'Memory Breakdown'}
          </span>
          <span className="text-lg font-bold" style={{ color: '#1F2937' }}>
            {formatNumber(primaryTotal, 1)} <span className="text-xs" style={{ color: '#6B7280' }}>GB</span>
          </span>
        </div>
        <div style={{ 
          width: '100%', 
          height: hasComparison ? '32px' : '40px', 
          backgroundColor: '#d1d5db', 
          borderRadius: '4px', 
          overflow: 'hidden',
          display: 'flex'
        }}>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${hasComparison ? primaryBarWidth : 100}%` }}
            transition={{ delay: delay + 0.2, duration: 0.6 }}
            style={{ 
              height: '100%', 
              display: 'flex',
              borderRadius: '4px'
            }}
          >
            <div
              style={{ 
                width: `${primaryWeightPercent}%`,
                height: '100%', 
                backgroundColor: weightColor,
              }}
              title={`Weights: ${formatNumber(primaryMemory.weight_memory_gb, 1)} GB`}
            />
            <div
              style={{ 
                width: `${primaryActivationPercent}%`,
                height: '100%', 
                backgroundColor: activationColor,
              }}
              title={`Activations: ${formatNumber(primaryMemory.activation_memory_gb, 1)} GB`}
            />
            <div
              style={{ 
                width: `${primaryKvPercent}%`,
                height: '100%', 
                backgroundColor: kvColor,
              }}
              title={`KV Cache: ${formatNumber(primaryMemory.kv_cache_gb, 1)} GB`}
            />
          </motion.div>
        </div>
        {/* Legend for primary */}
        <div className="flex gap-4 mt-2 text-xs" style={{ color: '#6B7280' }}>
          <div className="flex items-center gap-1">
            <div style={{ width: '12px', height: '12px', backgroundColor: weightColor, borderRadius: '2px' }} />
            <span>Weights: {formatNumber(primaryMemory.weight_memory_gb, 1)} GB</span>
          </div>
          <div className="flex items-center gap-1">
            <div style={{ width: '12px', height: '12px', backgroundColor: activationColor, borderRadius: '2px' }} />
            <span>Activations: {formatNumber(primaryMemory.activation_memory_gb, 1)} GB</span>
          </div>
          <div className="flex items-center gap-1">
            <div style={{ width: '12px', height: '12px', backgroundColor: kvColor, borderRadius: '2px' }} />
            <span>KV Cache: {formatNumber(primaryMemory.kv_cache_gb, 1)} GB</span>
          </div>
        </div>
      </div>

      {/* Comparison System Stacked Bar */}
      {hasComparison && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold" style={{ color: '#f96c56' }}>Comparison</span>
            <span className="text-lg font-bold" style={{ color: '#1F2937' }}>
              {formatNumber(comparisonTotal, 1)} <span className="text-xs" style={{ color: '#6B7280' }}>GB</span>
            </span>
          </div>
          <div style={{ 
            width: '100%', 
            height: '32px', 
            backgroundColor: '#d1d5db', 
            borderRadius: '4px', 
            overflow: 'hidden',
            display: 'flex'
          }}>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${comparisonBarWidth}%` }}
              transition={{ delay: delay + 0.2, duration: 0.6 }}
              style={{ 
                height: '100%',
                display: 'flex',
                borderRadius: '4px'
              }}
            >
              <div
                style={{ 
                  width: `${comparisonWeightPercent}%`,
                  height: '100%', 
                  backgroundColor: compWeightColor,
                }}
                title={`Weights: ${formatNumber(comparisonMemory.weight_memory_gb, 1)} GB`}
              />
              <div
                style={{ 
                  width: `${comparisonActivationPercent}%`,
                  height: '100%', 
                  backgroundColor: compActivationColor,
                }}
                title={`Activations: ${formatNumber(comparisonMemory.activation_memory_gb, 1)} GB`}
              />
              <div
                style={{ 
                  width: `${comparisonKvPercent}%`,
                  height: '100%', 
                  backgroundColor: compKvColor,
                }}
                title={`KV Cache: ${formatNumber(comparisonMemory.kv_cache_gb, 1)} GB`}
              />
            </motion.div>
          </div>
          {/* Legend for comparison */}
          <div className="flex gap-4 mt-2 text-xs" style={{ color: '#6B7280' }}>
            <div className="flex items-center gap-1">
              <div style={{ width: '12px', height: '12px', backgroundColor: compWeightColor, borderRadius: '2px' }} />
              <span>Weights: {formatNumber(comparisonMemory.weight_memory_gb, 1)} GB</span>
            </div>
            <div className="flex items-center gap-1">
              <div style={{ width: '12px', height: '12px', backgroundColor: compActivationColor, borderRadius: '2px' }} />
              <span>Activations: {formatNumber(comparisonMemory.activation_memory_gb, 1)} GB</span>
            </div>
            <div className="flex items-center gap-1">
              <div style={{ width: '12px', height: '12px', backgroundColor: compKvColor, borderRadius: '2px' }} />
              <span>KV Cache: {formatNumber(comparisonMemory.kv_cache_gb, 1)} GB</span>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}

function HorizontalBarChart({ label, primaryValue, comparisonValue, unit, maxValue, delay }) {
  const primaryPercentage = (primaryValue / maxValue) * 100;
  const comparisonPercentage = (comparisonValue / maxValue) * 100;

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
        minWidth: 0
      }}
    >
      <div className="mb-3">
        <span className="text-sm uppercase tracking-wider" style={{ color: '#6B7280' }}>{label}</span>
      </div>
      
      {/* Primary System Bar */}
      <div className="mb-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-semibold" style={{ color: '#10B981' }}>Primary</span>
          <span className="text-lg font-bold" style={{ color: '#1F2937' }}>
            {formatNumber(primaryValue, primaryValue >= 100 ? 0 : 1)} <span className="text-xs" style={{ color: '#6B7280' }}>{unit}</span>
          </span>
        </div>
        <div style={{ width: '100%', height: '24px', backgroundColor: '#d1d5db', borderRadius: '4px', overflow: 'hidden' }}>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${primaryPercentage}%` }}
            transition={{ delay: delay + 0.2, duration: 0.6 }}
            style={{ 
              height: '100%', 
              backgroundColor: '#10B981',
              borderRadius: '4px'
            }}
          />
        </div>
      </div>

      {/* Comparison System Bar */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-semibold" style={{ color: '#f96c56' }}>Comparison</span>
          <span className="text-lg font-bold" style={{ color: '#1F2937' }}>
            {formatNumber(comparisonValue, comparisonValue >= 100 ? 0 : 1)} <span className="text-xs" style={{ color: '#6B7280' }}>{unit}</span>
          </span>
        </div>
        <div style={{ width: '100%', height: '24px', backgroundColor: '#d1d5db', borderRadius: '4px', overflow: 'hidden' }}>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${comparisonPercentage}%` }}
            transition={{ delay: delay + 0.2, duration: 0.6 }}
            style={{ 
              height: '100%', 
              backgroundColor: '#f96c56',
              borderRadius: '4px'
            }}
          />
        </div>
      </div>
    </motion.div>
  );
}

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
 * MetricsDisplay - Shows system metrics with TTFT, TPOT, TPS/User, and Throughput
 * Displays horizontal bar charts when comparison is active
 */
export default function MetricsDisplay({ metrics, comparisonMetrics = null }) {
  const [activeTab, setActiveTab] = useState('system');
  
  if (!metrics) return null;

  // Backend returns flat structure: { ttft_ms, tpot_ms, tps_user, throughput_tokens_s, ... }
  const hasComparison = comparisonMetrics !== null;

  return (
    <div style={{ paddingTop: '1.05rem', width: '100%' }}>
      {/* Tab Navigation */}
      <div style={{ 
        display: 'flex', 
        gap: '0.5rem',
        paddingLeft: '1.4rem',
        paddingRight: '1.4rem',
        marginBottom: '1.5rem'
      }}>
        <button
          onClick={() => setActiveTab('system')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.75rem 1.5rem',
            backgroundColor: activeTab === 'system' ? '#29AF83' : '#2c2c2c',
            color: activeTab === 'system' ? '#ffffff' : '#9CA3AF',
            border: 'none',
            borderRadius: '6px',
            fontSize: '0.875rem',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
          onMouseEnter={(e) => {
            if (activeTab !== 'system') {
              e.currentTarget.style.backgroundColor = '#3a3a3a';
            }
          }}
          onMouseLeave={(e) => {
            if (activeTab !== 'system') {
              e.currentTarget.style.backgroundColor = '#2c2c2c';
            }
          }}
        >
          <Zap size={16} />
          <span>System Metrics</span>
        </button>
        <button
          onClick={() => setActiveTab('layers')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.75rem 1.5rem',
            backgroundColor: activeTab === 'layers' ? '#29AF83' : '#2c2c2c',
            color: activeTab === 'layers' ? '#ffffff' : '#9CA3AF',
            border: 'none',
            borderRadius: '6px',
            fontSize: '0.875rem',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
          onMouseEnter={(e) => {
            if (activeTab !== 'layers') {
              e.currentTarget.style.backgroundColor = '#3a3a3a';
            }
          }}
          onMouseLeave={(e) => {
            if (activeTab !== 'layers') {
              e.currentTarget.style.backgroundColor = '#2c2c2c';
            }
          }}
        >
          <Layers size={16} />
          <span>Layer Metrics</span>
        </button>
      </div>

      {/* System Metrics Tab */}
      {activeTab === 'system' && (
        <div style={{ width: '100%' }}>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
            gap: '1.05rem',
            width: '100%',
            paddingLeft: '1.4rem',
            paddingRight: '1.4rem'
          }}>
          {hasComparison ? (
            // Bar chart mode when comparison is active
            <>
              <HorizontalBarChart
                label="TTFT (Time to First Token)"
                primaryValue={metrics.ttft_ms || 0}
                comparisonValue={comparisonMetrics.ttft_ms || 0}
                unit="ms"
                maxValue={Math.max(metrics.ttft_ms || 0, comparisonMetrics.ttft_ms || 0) * 1.1}
                delay={0.1}
              />
              <HorizontalBarChart
                label="TPOT (Time per Output Token)"
                primaryValue={metrics.tpot_ms || 0}
                comparisonValue={comparisonMetrics.tpot_ms || 0}
                unit="ms"
                maxValue={Math.max(metrics.tpot_ms || 0, comparisonMetrics.tpot_ms || 0) * 1.1}
                delay={0.2}
              />
              <HorizontalBarChart
                label="TPS/User (Tokens per Second per User)"
                primaryValue={metrics.tps_user || 0}
                comparisonValue={comparisonMetrics.tps_user || 0}
                unit="tok/s"
                maxValue={Math.max(metrics.tps_user || 0, comparisonMetrics.tps_user || 0) * 1.1}
                delay={0.3}
              />
              <HorizontalBarChart
                label="Throughput (System-wide)"
                primaryValue={metrics.throughput_tokens_s || 0}
                comparisonValue={comparisonMetrics.throughput_tokens_s || 0}
                unit="tok/s"
                maxValue={Math.max(metrics.throughput_tokens_s || 0, comparisonMetrics.throughput_tokens_s || 0) * 1.1}
                delay={0.4}
              />
              {/* Memory Breakdown in Comparison Mode */}
              {metrics.memory && (
                <StackedBarChart
                  label="Memory Usage"
                  primaryMemory={metrics.memory}
                  comparisonMemory={comparisonMetrics?.memory || null}
                  delay={0.5}
                />
              )}
              
              {/* System Info in Comparison Mode */}
              {metrics.system && comparisonMetrics?.system && (
                <>
                  <HorizontalBarChart
                    label="Packages Used"
                    primaryValue={metrics.system.num_packages || 0}
                    comparisonValue={comparisonMetrics.system.num_packages || 0}
                    unit="packages"
                    maxValue={Math.max(metrics.system.num_packages || 0, comparisonMetrics.system.num_packages || 0) * 1.1}
                    delay={0.6}
                  />
                  <HorizontalBarChart
                    label="Power Consumption"
                    primaryValue={metrics.system.power_kw || 0}
                    comparisonValue={comparisonMetrics.system.power_kw || 0}
                    unit="kW"
                    maxValue={Math.max(metrics.system.power_kw || 0, comparisonMetrics.system.power_kw || 0) * 1.1}
                    delay={0.7}
                  />
                  <HorizontalBarChart
                    label="TCO per Second"
                    primaryValue={metrics.system.tco_sec_usd || 0}
                    comparisonValue={comparisonMetrics.system.tco_sec_usd || 0}
                    unit="$/s"
                    maxValue={Math.max(metrics.system.tco_sec_usd || 0, comparisonMetrics.system.tco_sec_usd || 0) * 1.1}
                    delay={0.8}
                  />
                  <StatCard
                    label="Hardware Fit"
                    value={metrics.system.fits_on_hardware ? '✓ Yes' : '✗ No'}
                    comparisonValue={comparisonMetrics.system.fits_on_hardware ? '✓ Yes' : '✗ No'}
                    icon={CheckCircle}
                    delay={0.9}
                  />
                  <StatCard
                    label="Bottleneck"
                    value={metrics.system.bottleneck || 'N/A'}
                    comparisonValue={comparisonMetrics.system.bottleneck || 'N/A'}
                    icon={Server}
                    delay={1.0}
                  />
                </>
              )}
            </>
          ) : (
            // Simple stat cards when no comparison
            <>
              <StatCard
                label="TTFT (Time to First Token)"
                value={formatNumber(metrics.ttft_ms) || 'N/A'}
                unit="ms"
                icon={Clock}
                delay={0.1}
                highlight={true}
              />
              <StatCard
                label="TPOT (Time per Output Token)"
                value={formatNumber(metrics.tpot_ms) || 'N/A'}
                unit="ms"
                icon={Clock}
                delay={0.2}
                highlight={true}
              />
              <StatCard
                label="TPS/User"
                value={formatNumber(metrics.tps_user) || 'N/A'}
                unit="tok/s"
                icon={Activity}
                delay={0.3}
                highlight={true}
              />
              <StatCard
                label="Throughput"
                value={formatNumber(metrics.throughput_tokens_s) || 'N/A'}
                unit="tok/s"
                icon={Activity}
                delay={0.4}
                highlight={true}
              />
              {/* Memory Breakdown as a card */}
              {metrics.memory && (
                <StackedBarChart
                  label="Memory Usage"
                  primaryMemory={metrics.memory}
                  comparisonMemory={null}
                  delay={0.5}
                />
              )}
              
              {/* System Info Cards */}
              {metrics.system && (
                <>
                  <StatCard
                    label="Packages Used"
                    value={metrics.system.num_packages || 'N/A'}
                    unit="packages"
                    icon={Cpu}
                    delay={0.6}
                  />
                  <StatCard
                    label="Memory per Package"
                    value={formatNumber(metrics.memory?.memory_per_package_gb, 1) || 'N/A'}
                    unit="GB"
                    icon={HardDrive}
                    delay={0.7}
                  />
                  <StatCard
                    label="Hardware Fit"
                    value={metrics.system.fits_on_hardware ? 'Yes' : 'No'}
                    icon={metrics.system.fits_on_hardware ? CheckCircle : AlertTriangle}
                    delay={0.8}
                  />
                  <StatCard
                    label="Power Consumption"
                    value={formatNumber(metrics.system.power_kw) || 'N/A'}
                    unit="kW"
                    icon={Zap}
                    delay={0.9}
                  />
                  <StatCard
                    label="TCO per Second"
                    value={metrics.system.tco_sec_usd ? `$${formatNumber(metrics.system.tco_sec_usd, 4)}` : 'N/A'}
                    icon={DollarSign}
                    delay={1.0}
                  />
                  <StatCard
                    label="Model FLOPs Util."
                    value={metrics.system.mfu ? `${formatNumber(metrics.system.mfu * 100, 1)}%` : 'N/A'}
                    icon={Activity}
                    delay={1.1}
                  />
                  <StatCard
                    label="Bottleneck"
                    value={metrics.system.bottleneck || 'N/A'}
                    icon={Server}
                    delay={1.2}
                  />
                </>
              )}
            </>
          )}
        </div>
        </div>
      )}

      {/* Layer Metrics Tab */}
      {activeTab === 'layers' && (
        <div style={{ width: '100%', paddingLeft: '1.4rem', paddingRight: '1.4rem' }}>
          <div style={{ 
            backgroundColor: '#e7e3da',
            border: '1px solid #E5E7EB',
            padding: '2rem',
            borderRadius: '8px',
            textAlign: 'center',
            color: '#6B7280'
          }}>
            <Layers size={48} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
            <p style={{ fontSize: '1rem', fontWeight: '500' }}>Layer Metrics</p>
            <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>Coming soon...</p>
          </div>
        </div>
      )}

      {/* Debug: Show raw data */}
      <details className="mt-4" style={{ paddingLeft: '1.4rem', paddingRight: '1.4rem' }}>
        <summary className="text-xs cursor-pointer" style={{ color: '#6B7280' }}>
          View raw metrics data
        </summary>
        <div className="mt-2 p-3 rounded" style={{ backgroundColor: '#1F2937' }}>
          <div className="mb-2">
            <span className="text-xs font-semibold" style={{ color: '#10B981' }}>Primary:</span>
            <pre className="text-xs mt-1 overflow-auto max-h-64" style={{ color: '#D1D5DB' }}>
              {JSON.stringify(metrics, null, 2)}
            </pre>
          </div>
          {hasComparison && (
            <div>
              <span className="text-xs font-semibold" style={{ color: '#f96c56' }}>Comparison:</span>
              <pre className="text-xs mt-1 overflow-auto max-h-64" style={{ color: '#D1D5DB' }}>
                {JSON.stringify(comparisonMetrics, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </details>
    </div>
  );
}
