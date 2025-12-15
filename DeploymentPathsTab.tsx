import { useState, useEffect } from 'react';
import { MapPin, TrendingUp, Target, Zap, Route } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line, Legend } from 'recharts';

interface DeploymentPathsTabProps {
  honeypotCount: number;
  isRunning: boolean;
}

interface PathNode {
  node: string;
  gCost: number;
  hCost: number;
  fCost: number;
}

interface DeploymentPath {
  id: string;
  honeypotName: string;
  algorithm: 'A*' | 'Best-First';
  pathNodes: PathNode[];
  totalCost: number;
  status: 'optimal' | 'suboptimal' | 'computing';
}

export default function DeploymentPathsTab({ honeypotCount, isRunning }: DeploymentPathsTabProps) {
  const [selectedDeployment, setSelectedDeployment] = useState<string>('H-1');
  const [deploymentPaths, setDeploymentPaths] = useState<DeploymentPath[]>([]);
  const [performanceMetrics, setPerformanceMetrics] = useState([
    { iteration: 1, astarCost: 5.2, bestFirstCost: 6.1, nodesExplored: 124 },
    { iteration: 5, astarCost: 4.8, bestFirstCost: 5.7, nodesExplored: 98 },
    { iteration: 10, astarCost: 4.5, bestFirstCost: 5.2, nodesExplored: 82 },
    { iteration: 15, astarCost: 4.2, bestFirstCost: 4.9, nodesExplored: 76 },
    { iteration: 20, astarCost: 3.9, bestFirstCost: 4.6, nodesExplored: 68 }
  ]);

  // Generate deployment paths based on honeypotCount
  useEffect(() => {
    const paths: DeploymentPath[] = [];
    
    for (let i = 0; i < Math.min(honeypotCount, 8); i++) {
      const algorithm = i % 2 === 0 ? 'A*' : 'Best-First';
      const pathLength = 3 + Math.floor(Math.random() * 3);
      const pathNodes: PathNode[] = [];
      
      let accumulatedG = 0;
      for (let j = 0; j < pathLength; j++) {
        const gCost = accumulatedG + (j === 0 ? 0 : 0.8 + Math.random() * 1.2);
        const hCost = (pathLength - j - 1) * (1.2 + Math.random() * 0.8);
        accumulatedG = gCost;
        
        pathNodes.push({
          node: j === 0 ? 'Start' : (j === pathLength - 1 ? `H-${i + 1}` : `Node-${j}`),
          gCost: parseFloat(gCost.toFixed(2)),
          hCost: parseFloat(hCost.toFixed(2)),
          fCost: parseFloat((gCost + hCost).toFixed(2))
        });
      }
      
      const totalCost = pathNodes[pathNodes.length - 1].gCost;
      const status = totalCost < 4 ? 'optimal' : (totalCost < 5.5 ? 'suboptimal' : 'computing');
      
      paths.push({
        id: `H-${i + 1}`,
        honeypotName: `Honeypot-${i + 1}`,
        algorithm,
        pathNodes,
        totalCost,
        status
      });
    }
    
    setDeploymentPaths(paths);
    if (paths.length > 0) {
      setSelectedDeployment(paths[0].id);
    }
  }, [honeypotCount]);

  // Update performance metrics during simulation
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setPerformanceMetrics(prev => {
        const last = prev[prev.length - 1];
        const newMetric = {
          iteration: last.iteration + 5,
          astarCost: Math.max(3.2, last.astarCost - 0.05 - Math.random() * 0.1),
          bestFirstCost: Math.max(3.8, last.bestFirstCost - 0.05 - Math.random() * 0.1),
          nodesExplored: Math.max(50, last.nodesExplored - Math.floor(Math.random() * 5))
        };
        return [...prev.slice(-4), newMetric];
      });
    }, 8000);

    return () => clearInterval(interval);
  }, [isRunning]);

  const currentPath = deploymentPaths.find(p => p.id === selectedDeployment);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'optimal':
        return 'from-green-900/30 to-emerald-900/30 border-green-500/50 text-green-300';
      case 'suboptimal':
        return 'from-amber-900/30 to-orange-900/30 border-amber-500/50 text-amber-300';
      default:
        return 'from-blue-900/30 to-cyan-900/30 border-blue-500/50 text-blue-300';
    }
  };

  const costComparisonData = deploymentPaths.map(p => ({
    name: p.id,
    cost: p.totalCost,
    algorithm: p.algorithm
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-purple-500/30 rounded-2xl p-8 shadow-2xl shadow-purple-500/20 overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 border-t-2 border-r-2 border-purple-400/50 rounded-tr-2xl" />
        <div className="absolute bottom-0 left-0 w-32 h-32 border-b-2 border-l-2 border-purple-400/50 rounded-bl-2xl" />
        
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl border border-purple-400/50">
              <MapPin className="w-8 h-8 text-purple-400 drop-shadow-[0_0_10px_rgba(168,85,247,0.8)]" />
            </div>
            <div>
              <h2 className="text-2xl bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent font-bold">
                Optimal Deployment Paths
              </h2>
              <p className="text-sm text-gray-400">A* & Best-First search algorithms</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="px-4 py-2 bg-gradient-to-r from-purple-900/40 to-pink-900/40 rounded-xl border-2 border-purple-500/50">
              <div className="text-xs text-gray-400">Active Algorithm</div>
              <div className="text-lg font-bold text-purple-300">A* + Best-First</div>
            </div>
          </div>
        </div>

        {/* Deployment Selector */}
        <div className="grid grid-cols-4 gap-3">
          {deploymentPaths.map((deployment) => (
            <button
              key={deployment.id}
              onClick={() => setSelectedDeployment(deployment.id)}
              className={`group relative p-4 rounded-xl border-2 transition-all duration-300 overflow-hidden ${
                selectedDeployment === deployment.id
                  ? 'bg-gradient-to-br from-purple-900/40 to-pink-900/40 border-purple-500/70 scale-105 shadow-[0_0_30px_rgba(168,85,247,0.4)]'
                  : 'bg-gradient-to-br from-[#0a0e27] to-[#0d1238] border-purple-900/30 hover:border-purple-500/50'
              }`}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/0 via-purple-500/10 to-purple-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
              <div className="relative z-10">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-gray-300">{deployment.id}</span>
                  <Target className="w-4 h-4 text-purple-400" />
                </div>
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="text-gray-400">Cost</span>
                  <span className="font-bold text-purple-300">{deployment.totalCost.toFixed(2)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className={`text-xs px-2 py-1 rounded border ${getStatusColor(deployment.status)}`}>
                    {deployment.algorithm}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded border ${getStatusColor(deployment.status)}`}>
                    {deployment.status}
                  </span>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-2 gap-6">
        {/* Path Search Tree */}
        {currentPath && (
          <div className="relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-cyan-500/30 rounded-2xl p-6 shadow-2xl shadow-cyan-500/20 overflow-hidden">
            <div className="absolute top-0 left-0 w-20 h-20 border-t-2 border-l-2 border-cyan-400/50 rounded-tl-2xl" />
            
            <div className="flex items-center gap-3 mb-4">
              <Route className="w-6 h-6 text-cyan-400 drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
              <div>
                <h3 className="text-cyan-400 font-semibold">Search Path Tree</h3>
                <p className="text-xs text-gray-400">{currentPath.honeypotName}</p>
              </div>
            </div>

            {/* Path Table */}
            <div className="bg-gradient-to-br from-[#020817] to-[#0a0e27] border-2 border-cyan-900/30 rounded-xl overflow-hidden mb-4">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b-2 border-cyan-900/50 bg-cyan-900/20">
                    <th className="text-left p-3 text-cyan-400 font-semibold">Node</th>
                    <th className="text-right p-3 text-cyan-400 font-semibold">g(n)</th>
                    <th className="text-right p-3 text-cyan-400 font-semibold">h(n)</th>
                    <th className="text-right p-3 text-cyan-400 font-semibold">f(n)</th>
                  </tr>
                </thead>
                <tbody>
                  {currentPath.pathNodes.map((node, index) => (
                    <tr 
                      key={index}
                      className={`border-b border-cyan-900/20 transition-colors ${
                        index === currentPath.pathNodes.length - 1 
                          ? 'bg-gradient-to-r from-green-900/30 to-emerald-900/30' 
                          : 'hover:bg-cyan-900/10'
                      }`}
                      style={{
                        animation: `fadeInUp 0.5s ease-out ${index * 0.1}s both`
                      }}
                    >
                      <td className="p-3 text-gray-300 font-medium">{node.node}</td>
                      <td className="p-3 text-right text-cyan-400 font-mono font-bold">{node.gCost.toFixed(2)}</td>
                      <td className="p-3 text-right text-amber-400 font-mono font-bold">{node.hCost.toFixed(2)}</td>
                      <td className="p-3 text-right text-green-400 font-mono font-bold">{node.fCost.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Path Metrics */}
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-gradient-to-br from-cyan-900/30 to-blue-900/30 border border-cyan-500/50 rounded-xl p-3">
                <div className="text-xs text-gray-400 mb-1">Algorithm</div>
                <div className="text-cyan-400 font-bold">{currentPath.algorithm}</div>
              </div>
              <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 border border-purple-500/50 rounded-xl p-3">
                <div className="text-xs text-gray-400 mb-1">Path Cost</div>
                <div className="text-purple-400 font-bold">{currentPath.totalCost.toFixed(2)}</div>
              </div>
              <div className="bg-gradient-to-br from-green-900/30 to-emerald-900/30 border border-green-500/50 rounded-xl p-3">
                <div className="text-xs text-gray-400 mb-1">Steps</div>
                <div className="text-green-400 font-bold">{currentPath.pathNodes.length}</div>
              </div>
            </div>
          </div>
        )}

        {/* Cost Comparison Chart */}
        <div className="relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-amber-500/30 rounded-2xl p-6 shadow-2xl shadow-amber-500/20 overflow-hidden">
          <div className="absolute top-0 right-0 w-20 h-20 border-t-2 border-r-2 border-amber-400/50 rounded-tr-2xl" />
          
          <div className="flex items-center gap-3 mb-4">
            <TrendingUp className="w-6 h-6 text-amber-400 drop-shadow-[0_0_8px_rgba(251,191,36,0.8)]" />
            <h3 className="text-amber-400 font-semibold">Deployment Cost Analysis</h3>
          </div>

          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={costComparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" opacity={0.3} />
              <XAxis 
                dataKey="name" 
                stroke="#6b7280" 
                tick={{ fill: '#9ca3af', fontSize: 11 }}
                angle={-15}
                textAnchor="end"
                height={60}
              />
              <YAxis 
                stroke="#6b7280" 
                tick={{ fill: '#9ca3af', fontSize: 11 }}
                label={{ value: 'Path Cost', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 12 }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#0d1238', 
                  border: '2px solid rgba(251, 191, 36, 0.5)',
                  borderRadius: '12px',
                  fontSize: '12px'
                }}
                formatter={(value: number, name: string, props: any) => [
                  `${value.toFixed(2)} (${props.payload.algorithm})`,
                  'Cost'
                ]}
              />
              <Bar dataKey="cost" radius={[12, 12, 0, 0]}>
                {costComparisonData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.name === selectedDeployment ? '#a855f7' : (entry.algorithm === 'A*' ? '#06b6d4' : '#10b981')}
                    style={{ filter: `drop-shadow(0 0 8px ${entry.name === selectedDeployment ? '#a855f7' : '#06b6d4'})` }}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Algorithm Performance Over Time */}
        <div className="col-span-2 relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-green-500/30 rounded-2xl p-6 shadow-2xl shadow-green-500/20 overflow-hidden">
          <div className="absolute bottom-0 left-0 w-32 h-32 border-b-2 border-l-2 border-green-400/50 rounded-bl-2xl" />
          <div className="absolute top-0 right-0 w-32 h-32 border-t-2 border-r-2 border-green-400/50 rounded-tr-2xl" />
          
          <div className="flex items-center gap-3 mb-4">
            <Zap className="w-6 h-6 text-green-400 drop-shadow-[0_0_8px_rgba(34,197,94,0.8)]" />
            <h3 className="text-green-400 font-semibold">Algorithm Performance Metrics</h3>
          </div>

          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={performanceMetrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" opacity={0.3} />
              <XAxis 
                dataKey="iteration" 
                stroke="#6b7280" 
                tick={{ fill: '#9ca3af', fontSize: 11 }}
                label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 12 }}
              />
              <YAxis 
                yAxisId="left"
                stroke="#6b7280" 
                tick={{ fill: '#9ca3af', fontSize: 11 }}
                label={{ value: 'Avg Path Cost', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 12 }}
              />
              <YAxis 
                yAxisId="right"
                orientation="right"
                stroke="#6b7280" 
                tick={{ fill: '#9ca3af', fontSize: 11 }}
                label={{ value: 'Nodes Explored', angle: 90, position: 'insideRight', fill: '#9ca3af', fontSize: 12 }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#0d1238', 
                  border: '2px solid rgba(34, 197, 94, 0.5)',
                  borderRadius: '12px',
                  fontSize: '12px'
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="astarCost" 
                stroke="#06b6d4" 
                strokeWidth={3}
                dot={{ fill: '#06b6d4', r: 5, strokeWidth: 2, stroke: '#0a0e27' }}
                name="A* Avg Cost"
                style={{ filter: 'drop-shadow(0 0 6px rgba(6, 182, 212, 0.8))' }}
              />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="bestFirstCost" 
                stroke="#10b981" 
                strokeWidth={3}
                dot={{ fill: '#10b981', r: 5, strokeWidth: 2, stroke: '#0a0e27' }}
                name="Best-First Avg Cost"
                style={{ filter: 'drop-shadow(0 0 6px rgba(16, 185, 129, 0.8))' }}
              />
              <Line 
                yAxisId="right"
                type="monotone" 
                dataKey="nodesExplored" 
                stroke="#a855f7" 
                strokeWidth={2}
                dot={{ fill: '#a855f7', r: 4 }}
                name="Nodes Explored"
                strokeDasharray="5 5"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Algorithm Information */}
      <div className="grid grid-cols-2 gap-6">
        <div className="relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-cyan-500/30 rounded-2xl p-6 shadow-2xl shadow-cyan-500/20">
          <h3 className="text-cyan-400 font-semibold mb-4">A* Search Algorithm</h3>
          <div className="space-y-3 text-sm text-gray-300">
            <div className="flex justify-between">
              <span className="text-gray-400">Heuristic:</span>
              <span className="font-semibold">Manhattan Distance</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Admissibility:</span>
              <span className="text-green-400 font-semibold">Guaranteed Optimal</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Completeness:</span>
              <span className="text-green-400 font-semibold">Complete</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Time Complexity:</span>
              <span className="font-mono">O(b<sup>d</sup>)</span>
            </div>
          </div>
        </div>

        <div className="relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-green-500/30 rounded-2xl p-6 shadow-2xl shadow-green-500/20">
          <h3 className="text-green-400 font-semibold mb-4">Best-First Search</h3>
          <div className="space-y-3 text-sm text-gray-300">
            <div className="flex justify-between">
              <span className="text-gray-400">Heuristic:</span>
              <span className="font-semibold">Greedy h(n)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Admissibility:</span>
              <span className="text-amber-400 font-semibold">Not Guaranteed</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Completeness:</span>
              <span className="text-green-400 font-semibold">Complete</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Time Complexity:</span>
              <span className="font-mono">O(b<sup>m</sup>)</span>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
