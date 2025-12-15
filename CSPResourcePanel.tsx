import { useState, useEffect } from 'react';
import { Server, Activity, AlertCircle, CheckCircle } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

interface CSPResourcePanelProps {
  cpuUsage: number;
  memoryUsage: number;
  nodeCount: number;
  honeypotCount: number;
}

export default function CSPResourcePanel({ cpuUsage, memoryUsage, nodeCount, honeypotCount }: CSPResourcePanelProps) {
  const [constraints, setConstraints] = useState([
    { id: 1, name: 'CPU Threshold', condition: 'CPU ≤ 85%', satisfied: true, value: cpuUsage },
    { id: 2, name: 'Memory Limit', condition: 'Memory ≤ 80%', satisfied: true, value: memoryUsage },
    { id: 3, name: 'Min Honeypots', condition: 'Honeypots ≥ 4', satisfied: true, value: honeypotCount },
    { id: 4, name: 'Node Distribution', condition: 'Nodes ≤ 50', satisfied: true, value: nodeCount }
  ]);

  const [resourceData, setResourceData] = useState([
    { name: 'CPU', value: cpuUsage, color: '#06b6d4' },
    { name: 'Memory', value: memoryUsage, color: '#8b5cf6' },
    { name: 'Network', value: 42, color: '#10b981' },
    { name: 'Storage', value: 28, color: '#f59e0b' }
  ]);

  useEffect(() => {
    setConstraints(prev => prev.map(constraint => {
      if (constraint.name === 'CPU Threshold') {
        return { ...constraint, value: cpuUsage, satisfied: cpuUsage <= 85 };
      }
      if (constraint.name === 'Memory Limit') {
        return { ...constraint, value: memoryUsage, satisfied: memoryUsage <= 80 };
      }
      if (constraint.name === 'Min Honeypots') {
        return { ...constraint, value: honeypotCount, satisfied: honeypotCount >= 4 };
      }
      if (constraint.name === 'Node Distribution') {
        return { ...constraint, value: nodeCount, satisfied: nodeCount <= 50 };
      }
      return constraint;
    }));

    setResourceData(prev => prev.map(resource => {
      if (resource.name === 'CPU') return { ...resource, value: cpuUsage };
      if (resource.name === 'Memory') return { ...resource, value: memoryUsage };
      return resource;
    }));
  }, [cpuUsage, memoryUsage, nodeCount, honeypotCount]);

  const satisfiedCount = constraints.filter(c => c.satisfied).length;
  const constraintStatus = satisfiedCount === constraints.length ? 'Optimal' : 'Violation Detected';

  return (
    <div className="group relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-blue-500/30 rounded-2xl p-6 shadow-2xl shadow-blue-500/20 hover:shadow-blue-500/40 transition-all duration-500 overflow-hidden">
      {/* Animated corner accents */}
      <div className="absolute top-0 right-0 w-20 h-20 border-t-2 border-r-2 border-blue-400/50 rounded-tr-2xl" />
      <div className="absolute bottom-0 left-0 w-20 h-20 border-b-2 border-l-2 border-blue-400/50 rounded-bl-2xl" />
      
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-blue-500/20 to-indigo-500/20 rounded-lg border border-blue-400/50">
            <Server className="w-6 h-6 text-blue-400 drop-shadow-[0_0_8px_rgba(59,130,246,0.8)]" />
          </div>
          <div>
            <h3 className="text-blue-400 font-semibold">CSP Resource Manager</h3>
            <p className="text-xs text-gray-400">Constraint satisfaction solver</p>
          </div>
        </div>
        <div className={`px-3 py-1 rounded-lg border ${
          satisfiedCount === constraints.length
            ? 'bg-green-900/30 border-green-500/50 text-green-300'
            : 'bg-red-900/30 border-red-500/50 text-red-300'
        }`}>
          <span className="text-xs font-semibold">{constraintStatus}</span>
        </div>
      </div>

      {/* Resource Allocation */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gradient-to-br from-[#020817] to-[#0a0e27] border-2 border-blue-900/30 rounded-xl p-4">
          <div className="text-xs text-gray-400 mb-3">Resource Distribution</div>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie
                data={resourceData}
                cx="50%"
                cy="50%"
                innerRadius={35}
                outerRadius={60}
                paddingAngle={5}
                dataKey="value"
              >
                {resourceData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.color}
                    stroke={entry.color}
                    strokeWidth={2}
                    style={{ filter: `drop-shadow(0 0 8px ${entry.color})` }}
                  />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#0d1238', 
                  border: '2px solid rgba(59, 130, 246, 0.5)',
                  borderRadius: '12px',
                  fontSize: '12px'
                }}
                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Usage']}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="space-y-3">
          {resourceData.map((resource, index) => (
            <div 
              key={index}
              className="bg-gradient-to-r from-[#020817] to-[#0a0e27] border border-blue-900/30 rounded-lg p-3 hover:border-blue-500/50 transition-all duration-300"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-400">{resource.name}</span>
                <span className="text-sm font-bold" style={{ color: resource.color }}>
                  {resource.value.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-[#0d1238] rounded-full h-2 overflow-hidden border border-blue-900/30">
                <div 
                  className="h-full rounded-full transition-all duration-1000 relative overflow-hidden"
                  style={{ 
                    width: `${resource.value}%`,
                    backgroundColor: resource.color,
                    boxShadow: `0 0 10px ${resource.color}`
                  }}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent translate-x-[-100%] animate-shimmer" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Active Constraints */}
      <div className="bg-gradient-to-br from-[#020817] to-[#0a0e27] border-2 border-blue-900/30 rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <Activity className="w-4 h-4 text-blue-400" />
          <span className="text-sm font-semibold text-blue-400">Active Constraints</span>
          <span className="ml-auto text-xs text-gray-400">
            {satisfiedCount}/{constraints.length} Satisfied
          </span>
        </div>
        
        <div className="space-y-2">
          {constraints.map((constraint, index) => (
            <div 
              key={constraint.id}
              className={`bg-gradient-to-r border rounded-lg p-3 transition-all duration-300 ${
                constraint.satisfied
                  ? 'from-green-900/20 to-emerald-900/20 border-green-500/30 hover:border-green-500/50'
                  : 'from-red-900/20 to-rose-900/20 border-red-500/30 hover:border-red-500/50'
              }`}
              style={{
                animation: `popIn 0.4s ease-out ${index * 0.1}s both`
              }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-1">
                  {constraint.satisfied ? (
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 animate-pulse" />
                  )}
                  <div className="flex-1">
                    <div className="text-xs font-medium text-gray-300 mb-0.5">{constraint.name}</div>
                    <div className="text-xs text-gray-500 font-mono">{constraint.condition}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-gray-500">Current</div>
                  <div className={`text-sm font-bold ${constraint.satisfied ? 'text-green-400' : 'text-red-400'}`}>
                    {constraint.name.includes('CPU') || constraint.name.includes('Memory') 
                      ? `${constraint.value.toFixed(0)}%` 
                      : constraint.value}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* CSP Status */}
      <div className="mt-4 grid grid-cols-3 gap-3">
        <div className="bg-gradient-to-br from-cyan-900/30 to-blue-900/30 border border-cyan-500/40 rounded-xl p-3 text-center">
          <div className="text-xs text-gray-400 mb-1">Variables</div>
          <div className="text-lg font-bold text-cyan-400">{resourceData.length}</div>
        </div>
        <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 border border-purple-500/40 rounded-xl p-3 text-center">
          <div className="text-xs text-gray-400 mb-1">Constraints</div>
          <div className="text-lg font-bold text-purple-400">{constraints.length}</div>
        </div>
        <div className={`bg-gradient-to-br border rounded-xl p-3 text-center ${
          satisfiedCount === constraints.length
            ? 'from-green-900/30 to-emerald-900/30 border-green-500/40'
            : 'from-red-900/30 to-rose-900/30 border-red-500/40'
        }`}>
          <div className="text-xs text-gray-400 mb-1">Conflicts</div>
          <div className={`text-lg font-bold ${
            satisfiedCount === constraints.length ? 'text-green-400' : 'text-red-400'
          }`}>
            {constraints.length - satisfiedCount}
          </div>
        </div>
      </div>

      {/* Algorithm Info */}
      <div className="mt-4 p-3 bg-gradient-to-r from-blue-900/30 to-indigo-900/30 border border-blue-500/50 rounded-xl">
        <div className="text-xs text-gray-300">
          <span className="text-blue-400 font-semibold">Backtracking CSP Solver</span>
          {' '}with forward checking · Arc consistency · MRV heuristic
        </div>
      </div>

      <style>{`
        @keyframes popIn {
          from {
            opacity: 0;
            transform: scale(0.95);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
      `}</style>
    </div>
  );
}
