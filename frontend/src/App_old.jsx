import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Server, Cpu, Box, Play, BarChart2, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';
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

function ConfigSection({ title, children }) {
  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-accent font-semibold flex items-center gap-2">
        {title}
      </h3>
      <div className="flex flex-col gap-4">
        {children}
      </div>
    </div>
  );
}

// --- Main App ---

export default function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Config State
  const [hwPreset, setHwPreset] = useState('nvl72_single'); // Default to single for easier reasoning, or rack? User wanted NVL-72
  const [modelId, setModelId] = useState('TinyLlama/TinyLlama-1.1B-Chat-v1.0');
  const [tpSize, setTpSize] = useState(1);
  const [batchSize, setBatchSize] = useState(1);
  const [inputSeq, setInputSeq] = useState(128);
  const [outputSeq, setOutputSeq] = useState(128);

  const handleEstimate = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/estimate`, {
        model_id: modelId,
        hardware_preset: hwPreset,
        tp_size: parseInt(tpSize),
        batch_size: parseInt(batchSize),
        input_seq: parseInt(inputSeq),
        output_seq: parseInt(outputSeq)
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to fetch estimate. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    handleEstimate();
  }, []);

  // Charts Data
  const memoryData = result ? [
    { name: 'Weights', value: result.weights_mem_gb, color: '#45A29E' },
    { name: 'KV Cache', value: result.kv_cache_mem_gb, color: '#66FCF1' },
    { name: 'Activations', value: result.activation_mem_gb, color: '#C5C6C7' },
  ] : [];

  const bottleneckData = result ? [
    { name: 'Compute', value: result.compute_bound_percent, color: '#45A29E' },
    { name: 'Memory', value: result.memory_bound_percent, color: '#66FCF1' },
    { name: 'Comm', value: result.comm_bound_percent, color: '#ff4d4d' },
  ] : [];

  return (
    <div className="app-container">
      {/* Sidebar - Controls */}
      <aside className="sidebar">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-8 h-8 rounded bg-accent flex items-center justify-center">
            <Activity size={20} className="text-bg" />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-white">Mantile</h1>
        </div>

        <ConfigSection title={<><Server size={18} /> Hardware</>}>
          <div>
            <label className="label">Configuration</label>
            <select
              className="input-field"
              value={hwPreset}
              onChange={e => setHwPreset(e.target.value)}
            >
              <option value="nvl72_single">NVIDIA GB200 (Single Package)</option>
              <option value="nvl72_rack">NVIDIA NVL-72 (Full Rack)</option>
            </select>
          </div>
        </ConfigSection>

        <ConfigSection title={<><Box size={18} /> Model</>}>
          <div>
            <label className="label">Model</label>
            <select
              className="input-field"
              value={modelId}
              onChange={e => setModelId(e.target.value)}
            >
              <option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama 1.1B</option>
              <option value="meta-llama/Llama-3.3-70B-Instruct">Llama 3.3 70B</option>
            </select>
          </div>
        </ConfigSection>

        <ConfigSection title={<><Cpu size={18} /> Parallelism & Runtime</>}>
          <div>
            <label className="label">Tensor Parallelism (TP)</label>
            <input
              type="range" min="1" max={hwPreset === 'nvl72_rack' ? 72 : 1}
              className="w-full h-2 bg-surface rounded-lg appearance-none cursor-pointer accent-accent"
              value={tpSize}
              onChange={e => setTpSize(e.target.value)}
            />
            <div className="flex justify-between text-xs text-dim mt-1">
              <span>1</span>
              <span className="text-accent font-mono">{tpSize}</span>
              <span>{hwPreset === 'nvl72_rack' ? 72 : 1}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Batch Size</label>
              <input
                type="number" className="input-field"
                value={batchSize} onChange={e => setBatchSize(e.target.value)}
              />
            </div>
            <div>
              <label className="label">Seq Length (In)</label>
              <input
                type="number" className="input-field"
                value={inputSeq} onChange={e => setInputSeq(e.target.value)}
              />
            </div>
          </div>
          <div>
            <label className="label">Seq Length (Gen)</label>
            <input
              type="number" className="input-field"
              value={outputSeq} onChange={e => setOutputSeq(e.target.value)}
            />
          </div>
        </ConfigSection>

        <button
          onClick={handleEstimate}
          disabled={loading}
          className="btn-primary mt-auto flex items-center justify-center gap-2"
        >
          {loading ? 'Calculating...' : <><Play size={18} /> Update Estimate</>}
        </button>
      </aside>

      {/* Main Content - Results */}
      <main className="main-content">
        <header className="flex justify-between items-center pb-4 border-b border-white/5">
          <h2 className="text-2xl font-bold">Performance Estimate</h2>
          {result && (
            <div className="flex items-center gap-2 text-dim text-sm">
              <span className={`w-2 h-2 rounded-full ${result.max_mem_capacity_gb > result.total_mem_gb ? 'bg-accent' : 'bg-danger'}`}></span>
              {result.max_mem_capacity_gb > result.total_mem_gb ? 'Fit' : 'OOM'}
            </div>
          )}
        </header>

        {error && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-200 flex items-center gap-3"
          >
            <AlertCircle size={20} />
            {error}
          </motion.div>
        )}

        {result && (
          <div className="flex flex-col gap-6">

            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <StatCard label="TTFT (Prefill)" value={result.time_to_first_token_ms.toFixed(2)} unit="ms" icon={Box} delay={0.1} />
              <StatCard label="TPOT (Decode)" value={result.time_per_output_token_ms.toFixed(4)} unit="ms" icon={Activity} delay={0.2} />
              <StatCard label="Throughput" value={result.total_throughput_tokens_s.toFixed(0)} unit="tok/s" icon={BarChart2} delay={0.3} />
              <StatCard label="Memory Used" value={result.total_mem_gb.toFixed(1)} unit={`/ ${result.max_mem_capacity_gb.toFixed(0)} GB`} icon={Cpu} delay={0.4} />
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

              {/* Memory Config */}
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5 }}
                className="card"
              >
                <h3 className="text-lg font-semibold mb-4 text-dim">Memory Breakdown (GB)</h3>
                <div style={{ width: '100%', height: '300px' }}>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={memoryData} layout="vertical" margin={{ left: 20 }}>
                      <XAxis type="number" stroke="#8892b0" fontSize={12} />
                      <YAxis dataKey="name" type="category" stroke="#8892b0" fontSize={12} width={80} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1F2833', borderColor: '#45A29E', color: '#fff' }}
                        itemStyle={{ color: '#66FCF1' }}
                      />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={40}>
                        {memoryData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>

              {/* Bottleneck Analysis */}
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.6 }}
                className="card"
              >
                <h3 className="text-lg font-semibold mb-4 text-dim">Bottleneck Analysis (%)</h3>
                <div style={{ width: '100%', height: '300px' }}>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={bottleneckData} barCategoryGap="20%">
                      <XAxis dataKey="name" stroke="#8892b0" fontSize={12} />
                      <YAxis stroke="#8892b0" fontSize={12} domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1F2833', borderColor: '#45A29E', color: '#fff' }}
                      />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {bottleneckData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="text-xs text-dim mt-4 text-center">
                    Indicates which factor limits performance. &gt;50% means dominant constraint.
                  </p>
                </div>
              </motion.div>

            </div>
          </div>
        )}
      </main>
    </div>
  );
}
