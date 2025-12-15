import { useState, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, BarChart, Bar, Legend, LineChart, Line } from 'recharts';
import { MapPin, TrendingUp, Search, Layers } from 'lucide-react';

interface DeploymentPath {
  honeypotId: string;
  pathSteps: PathStep[];
  totalCost: number;
  deploymentTime: string;
  status: 'optimal' | 'suboptimal' | 'pending';
}

interface PathStep {
  node: string;
  gCost: number;
  hCost: number;
  fCost: number;
  x: number;
  y: number;
}

interface HeatmapCell {
  x: number;
  y: number;
  value: number;
  label: string;
}

export default function DeploymentAnalytics() {
  const [selectedDeployment, setSelectedDeployment] = useState<string>('honeypot-7');
  
  const deployments: DeploymentPath[] = [
    {
      honeypotId: 'honeypot-7',
      pathSteps: [
        { node: 'Start', gCost: 0, hCost: 4.5, fCost: 4.5, x: 1, y: 1 },
        { node: 'Router-1', gCost: 1.2, hCost: 3.3, fCost: 4.5, x: 2, y: 2 },
        { node: 'Switch-A', gCost: 2.4, hCost: 2.1, fCost: 4.5, x: 3, y: 3 },
        { node: 'Subnet-3', gCost: 3.6, hCost: 0.9, fCost: 4.5, x: 4, y: 4 },
        { node: 'Target', gCost: 4.5, hCost: 0, fCost: 4.5, x: 5, y: 5 }
      ],
      totalCost: 4.5,
      deploymentTime: '14:23:45',
      status: 'optimal'
    },
    {
      honeypotId: 'honeypot-12',
      pathSteps: [
        { node: 'Start', gCost: 0, hCost: 3.2, fCost: 3.2, x: 1, y: 2 },
        { node: 'Router-2', gCost: 1.0, hCost: 2.2, fCost: 3.2, x: 2, y: 3 },
        { node: 'Target', gCost: 3.2, hCost: 0, fCost: 3.2, x: 3, y: 4 }
      ],
      totalCost: 3.2,
      deploymentTime: '14:28:12',
      status: 'optimal'
    },
    {
      honeypotId: 'honeypot-5',
      pathSteps: [
        { node: 'Start', gCost: 0, hCost: 5.8, fCost: 5.8, x: 1, y: 1 },
        { node: 'Router-1', gCost: 1.5, hCost: 4.3, fCost: 5.8, x: 2, y: 1 },
        { node: 'Switch-B', gCost: 3.0, hCost: 2.8, fCost: 5.8, x: 3, y: 2 },
        { node: 'Router-3', gCost: 4.2, hCost: 1.6, fCost: 5.8, x: 4, y: 3 },
        { node: 'Target', gCost: 5.8, hCost: 0, fCost: 5.8, x: 5, y: 4 }
      ],
      totalCost: 5.8,
      deploymentTime: '14:31:58',
      status: 'optimal'
    },
    {
      honeypotId: 'honeypot-9',
      pathSteps: [
        { node: 'Start', gCost: 0, hCost: 6.2, fCost: 6.2, x: 1, y: 3 },
        { node: 'Router-2', gCost: 1.8, hCost: 4.4, fCost: 6.2, x: 2, y: 4 },
        { node: 'Switch-A', gCost: 3.4, hCost: 2.8, fCost: 6.2, x: 3, y: 5 },
        { node: 'Subnet-2', gCost: 5.0, hCost: 1.2, fCost: 6.2, x: 4, y: 5 },
        { node: 'Target', gCost: 6.2, hCost: 0, fCost: 6.2, x: 5, y: 5 }
      ],
      totalCost: 6.2,
      deploymentTime: '14:35:23',
      status: 'suboptimal'
    }
  ];

  const currentDeployment = deployments.find(d => d.honeypotId === selectedDeployment) || deployments[0];

  // Heatmap data for deployment density
  const [heatmapData, setHeatmapData] = useState<HeatmapCell[]>([
    { x: 1, y: 1, value: 2, label: 'Zone-A' },
    { x: 2, y: 1, value: 5, label: 'Zone-B' },
    { x: 3, y: 1, value: 3, label: 'Zone-C' },
    { x: 1, y: 2, value: 4, label: 'Zone-D' },
    { x: 2, y: 2, value: 8, label: 'Zone-E' },
    { x: 3, y: 2, value: 6, label: 'Zone-F' },
    { x: 1, y: 3, value: 1, label: 'Zone-G' },
    { x: 2, y: 3, value: 7, label: 'Zone-H' },
    { x: 3, y: 3, value: 4, label: 'Zone-I' }
  ]);

  // Cost comparison data
  const costComparison = deployments.map(d => ({
    name: d.honeypotId,
    cost: d.totalCost,
    steps: d.pathSteps.length
  }));

  // Algorithm performance over time
  const [performanceData, setPerformanceData] = useState([
    { iteration: 1, avgCost: 6.8, nodesExplored: 145 },
    { iteration: 5, avgCost: 5.9, nodesExplored: 112 },
    { iteration: 10, avgCost: 5.2, nodesExplored: 98 },
    { iteration: 15, avgCost: 4.8, nodesExplored: 87 },
    { iteration: 20, avgCost: 4.5, nodesExplored: 76 }
  ]);

  useEffect(() => {
    // Simulate performance updates
    const interval = setInterval(() => {
      setPerformanceData(prev => {
        const lastIteration = prev[prev.length - 1];
        const newIteration = {
          iteration: lastIteration.iteration + 5,
          avgCost: Math.max(3.5, lastIteration.avgCost - 0.1 - Math.random() * 0.2),
          nodesExplored: Math.max(60, lastIteration.nodesExplored - Math.floor(Math.random() * 8))
        };
        return [...prev.slice(-4), newIteration];
      });
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'optimal':
        return 'bg-green-900/30 text-green-300 border-green-500/50';
      case 'suboptimal':
        return 'bg-amber-900/30 text-amber-300 border-amber-500/50';
      default:
        return 'bg-cyan-900/30 text-cyan-300 border-cyan-500/50';
    }
  };

  const getHeatmapColor = (value: number) => {
    const intensity = value / 8; // Max value is 8
    const r = Math.floor(6 + intensity * 234); // From dark blue to cyan
    const g = Math.floor(182 + intensity * 73);
    const b = Math.floor(212 - intensity * 0);
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-[#0d1238] border border-cyan-900/30 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <MapPin className="w-6 h-6 text-cyan-400" />
            <div>
              <h2 className="text-cyan-400">A* Deployment Path Analytics</h2>
              <p className="text-sm text-gray-400">Optimal honeypot placement visualization</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Search className="w-5 h-5 text-cyan-400" />
            <span className="text-sm text-gray-300">A* Search Algorithm</span>
          </div>
        </div>

        {/* Deployment Selector */}
        <div className="grid grid-cols-4 gap-3">
          {deployments.map(deployment => (
            <button
              key={deployment.honeypotId}
              onClick={() => setSelectedDeployment(deployment.honeypotId)}
              className={`p-3 rounded-lg border transition-all ${
                selectedDeployment === deployment.honeypotId
                  ? 'bg-cyan-900/30 border-cyan-500/50'
                  : 'bg-[#0a0e27] border-cyan-900/30 hover:border-cyan-500/30'
              }`}
            >
              <div className="text-sm text-gray-300 mb-1">{deployment.honeypotId}</div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">Cost: {deployment.totalCost.toFixed(1)}</span>
                <span className={`text-xs px-2 py-0.5 rounded border ${getStatusColor(deployment.status)}`}>
                  {deployment.status}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Main Analytics Grid */}
      <div className="grid grid-cols-2 gap-6">
        {/* A* Path Visualization */}
        <div className="bg-[#0d1238] border border-cyan-900/30 rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="w-5 h-5 text-cyan-400" />
            <h3 className="text-cyan-400">Path Search Tree</h3>
          </div>
          
          {/* Path Steps Table */}
          <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg overflow-hidden mb-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-cyan-900/30">
                  <th className="text-left p-3 text-gray-400">Node</th>
                  <th className="text-right p-3 text-gray-400">g(n)</th>
                  <th className="text-right p-3 text-gray-400">h(n)</th>
                  <th className="text-right p-3 text-gray-400">f(n)</th>
                </tr>
              </thead>
              <tbody>
                {currentDeployment.pathSteps.map((step, index) => (
                  <tr 
                    key={index}
                    className={`border-b border-cyan-900/20 ${
                      index === currentDeployment.pathSteps.length - 1 
                        ? 'bg-green-900/20' 
                        : 'hover:bg-cyan-900/10'
                    }`}
                  >
                    <td className="p-3 text-gray-300">{step.node}</td>
                    <td className="p-3 text-right text-cyan-400">{step.gCost.toFixed(1)}</td>
                    <td className="p-3 text-right text-amber-400">{step.hCost.toFixed(1)}</td>
                    <td className="p-3 text-right text-green-400">{step.fCost.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Path Info */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">Total Path Cost</div>
              <div className="text-cyan-400">{currentDeployment.totalCost.toFixed(1)}</div>
            </div>
            <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">Path Length</div>
              <div className="text-cyan-400">{currentDeployment.pathSteps.length} nodes</div>
            </div>
          </div>
        </div>

        {/* Cost Comparison */}
        <div className="bg-[#0d1238] border border-cyan-900/30 rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
            <h3 className="text-cyan-400">Deployment Cost Comparison</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={costComparison}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
              <XAxis 
                dataKey="name" 
                stroke="#6b7280" 
                tick={{ fill: '#9ca3af', fontSize: 10 }}
                angle={-15}
                textAnchor="end"
                height={60}
              />
              <YAxis 
                stroke="#6b7280" 
                tick={{ fill: '#9ca3af', fontSize: 10 }}
                label={{ value: 'Cost', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#0d1238', 
                  border: '1px solid #0e7490',
                  borderRadius: '8px',
                  fontSize: '12px'
                }}
              />
              <Bar dataKey="cost" radius={[8, 8, 0, 0]}>
                {costComparison.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.name === selectedDeployment ? '#06b6d4' : '#10b981'} 
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Deployment Heatmap */}
        <div className="bg-[#0d1238] border border-cyan-900/30 rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <MapPin className="w-5 h-5 text-cyan-400" />
            <h3 className="text-cyan-400">Deployment Density Heatmap</h3>
          </div>
          
          <div className="grid grid-cols-3 gap-2">
            {heatmapData.map((cell, index) => (
              <div
                key={index}
                className="aspect-square rounded-lg border border-cyan-900/30 flex flex-col items-center justify-center cursor-pointer transition-transform hover:scale-105"
                style={{ backgroundColor: getHeatmapColor(cell.value) }}
              >
                <div className="text-sm text-white/90">{cell.label}</div>
                <div className="text-xs text-white/70 mt-1">{cell.value} nodes</div>
              </div>
            ))}
          </div>
          
          <div className="mt-4 flex items-center justify-between text-xs text-gray-400">
            <span>Low Density</span>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5, 6, 7, 8].map(val => (
                <div
                  key={val}
                  className="w-6 h-3 rounded"
                  style={{ backgroundColor: getHeatmapColor(val) }}
                />
              ))}
            </div>
            <span>High Density</span>
          </div>
        </div>

        {/* Algorithm Performance */}
        <div className="bg-[#0d1238] border border-cyan-900/30 rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
            <h3 className="text-cyan-400">A* Performance Over Time</h3>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
              <XAxis 
                dataKey="iteration" 
                stroke="#6b7280" 
                tick={{ fill: '#9ca3af', fontSize: 10 }}
                label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
              />
              <YAxis 
                yAxisId="left"
                stroke="#6b7280" 
                tick={{ fill: '#9ca3af', fontSize: 10 }}
                label={{ value: 'Avg Cost', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
              />
              <YAxis 
                yAxisId="right"
                orientation="right"
                stroke="#6b7280" 
                tick={{ fill: '#9ca3af', fontSize: 10 }}
                label={{ value: 'Nodes', angle: 90, position: 'insideRight', fill: '#9ca3af' }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#0d1238', 
                  border: '1px solid #0e7490',
                  borderRadius: '8px',
                  fontSize: '12px'
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="avgCost" 
                stroke="#06b6d4" 
                strokeWidth={2}
                dot={{ fill: '#06b6d4', r: 4 }}
                name="Avg Path Cost"
              />
              <Line 
                yAxisId="right"
                type="monotone" 
                dataKey="nodesExplored" 
                stroke="#10b981" 
                strokeWidth={2}
                dot={{ fill: '#10b981', r: 4 }}
                name="Nodes Explored"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Algorithm Details */}
      <div className="bg-[#0d1238] border border-cyan-900/30 rounded-lg p-6">
        <h3 className="text-cyan-400 mb-4">A* Algorithm Configuration</h3>
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-2">Heuristic Function</div>
            <div className="text-gray-300">Manhattan Distance</div>
          </div>
          <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-2">Data Structure</div>
            <div className="text-gray-300">Priority Queue (Min-Heap)</div>
          </div>
          <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-2">Admissibility</div>
            <div className="text-green-400">Guaranteed Optimal</div>
          </div>
          <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-2">Time Complexity</div>
            <div className="text-gray-300">O(b<sup>d</sup>)</div>
          </div>
        </div>
      </div>
    </div>
  );
}
