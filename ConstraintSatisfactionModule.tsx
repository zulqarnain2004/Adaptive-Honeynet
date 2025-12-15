import { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { Server, Zap, AlertCircle } from 'lucide-react';

interface Resource {
  name: string;
  allocated: number;
  capacity: number;
  unit: string;
}

interface Constraint {
  id: string;
  type: string;
  description: string;
  satisfied: boolean;
  value: string;
}

export default function ConstraintSatisfactionModule() {
  const [resources, setResources] = useState<Resource[]>([
    { name: 'CPU', allocated: 68, capacity: 100, unit: '%' },
    { name: 'Memory', allocated: 5.4, capacity: 8, unit: 'GB' },
    { name: 'Network', allocated: 420, capacity: 1000, unit: 'Mbps' },
    { name: 'Storage', allocated: 145, capacity: 500, unit: 'GB' }
  ]);

  const [constraints, setConstraints] = useState<Constraint[]>([
    {
      id: '1',
      type: 'Resource',
      description: 'CPU usage ≤ 85%',
      satisfied: true,
      value: '68%'
    },
    {
      id: '2',
      type: 'Network',
      description: 'Bandwidth per honeypot ≥ 20 Mbps',
      satisfied: true,
      value: '35 Mbps'
    },
    {
      id: '3',
      type: 'Deployment',
      description: 'Min honeypots per subnet = 2',
      satisfied: true,
      value: '3 nodes'
    },
    {
      id: '4',
      type: 'Security',
      description: 'Isolation level ≥ Medium',
      satisfied: true,
      value: 'High'
    }
  ]);

  const [cspStatus, setCspStatus] = useState({
    totalConstraints: 12,
    satisfiedConstraints: 11,
    conflictCount: 0,
    lastOptimization: new Date().toLocaleTimeString()
  });

  // Simulate resource updates
  useEffect(() => {
    const interval = setInterval(() => {
      setResources(prev => prev.map(resource => {
        const change = (Math.random() - 0.5) * (resource.capacity * 0.05);
        const newAllocated = Math.max(0, Math.min(resource.capacity, resource.allocated + change));
        return { ...resource, allocated: newAllocated };
      }));

      // Occasionally update constraint satisfaction
      if (Math.random() > 0.7) {
        setConstraints(prev => prev.map(constraint => {
          if (Math.random() > 0.85) {
            return { ...constraint, satisfied: Math.random() > 0.2 };
          }
          return constraint;
        }));

        setCspStatus(prev => ({
          ...prev,
          satisfiedConstraints: Math.max(8, prev.totalConstraints - Math.floor(Math.random() * 2)),
          conflictCount: Math.floor(Math.random() * 2),
          lastOptimization: new Date().toLocaleTimeString()
        }));
      }
    }, 4000);

    return () => clearInterval(interval);
  }, []);

  // Prepare data for pie chart
  const allocationData = resources.map(resource => ({
    name: resource.name,
    value: parseFloat(((resource.allocated / resource.capacity) * 100).toFixed(1)),
    allocated: resource.allocated,
    capacity: resource.capacity,
    unit: resource.unit
  }));

  const COLORS = ['#06b6d4', '#10b981', '#f59e0b', '#8b5cf6'];

  const getUtilizationColor = (percentage: number) => {
    if (percentage > 80) return 'text-red-400';
    if (percentage > 60) return 'text-amber-400';
    return 'text-green-400';
  };

  return (
    <div className="space-y-4">
      {/* CSP Status */}
      <div className="grid grid-cols-3 gap-2">
        <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-2">
          <div className="text-xs text-gray-400">Constraints</div>
          <div className="text-cyan-400 text-sm">
            {cspStatus.satisfiedConstraints}/{cspStatus.totalConstraints}
          </div>
        </div>
        <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-2">
          <div className="text-xs text-gray-400">Conflicts</div>
          <div className={`text-sm ${cspStatus.conflictCount === 0 ? 'text-green-400' : 'text-red-400'}`}>
            {cspStatus.conflictCount}
          </div>
        </div>
        <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-2">
          <div className="text-xs text-gray-400">Last Update</div>
          <div className="text-cyan-400 text-xs">{cspStatus.lastOptimization}</div>
        </div>
      </div>

      {/* Resource Allocation Visualization */}
      <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
        <div className="flex items-center gap-2 mb-3">
          <Server className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-cyan-400">Resource Allocation</span>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          {/* Pie Chart */}
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie
                data={allocationData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={60}
                paddingAngle={3}
                dataKey="value"
              >
                {allocationData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#0d1238', 
                  border: '1px solid #0e7490',
                  borderRadius: '8px',
                  fontSize: '11px'
                }}
                formatter={(value: number, name: string, props: any) => [
                  `${props.payload.allocated.toFixed(1)} / ${props.payload.capacity} ${props.payload.unit}`,
                  name
                ]}
              />
              <Legend 
                wrapperStyle={{ fontSize: '10px' }}
                iconType="circle"
              />
            </PieChart>
          </ResponsiveContainer>

          {/* Resource Details */}
          <div className="space-y-2">
            {resources.map((resource, index) => {
              const percentage = (resource.allocated / resource.capacity) * 100;
              return (
                <div key={resource.name} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-400">{resource.name}</span>
                    <span className={getUtilizationColor(percentage)}>
                      {percentage.toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-[#0d1238] rounded-full h-2 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${percentage}%`,
                        backgroundColor: COLORS[index % COLORS.length]
                      }}
                    />
                  </div>
                  <div className="text-xs text-gray-500">
                    {resource.allocated.toFixed(1)} / {resource.capacity} {resource.unit}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Active Constraints */}
      <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
        <div className="flex items-center gap-2 mb-3">
          <Zap className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-cyan-400">Active Constraints</span>
        </div>
        <div className="space-y-2 max-h-[160px] overflow-y-auto">
          {constraints.map(constraint => (
            <div 
              key={constraint.id}
              className="bg-[#0d1238] border border-cyan-900/20 rounded p-2"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs px-2 py-0.5 rounded bg-cyan-900/30 text-cyan-300">
                      {constraint.type}
                    </span>
                    {constraint.satisfied ? (
                      <span className="text-xs text-green-400">✓ Satisfied</span>
                    ) : (
                      <span className="text-xs text-red-400 flex items-center gap-1">
                        <AlertCircle className="w-3 h-3" />
                        Violated
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-400">{constraint.description}</div>
                  <div className="text-xs text-gray-500 mt-1">Current: {constraint.value}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* CSP Algorithm Info */}
      <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
        <div className="text-xs text-gray-400 mb-1">Solver: Backtracking with Forward Checking</div>
        <div className="text-xs text-gray-500">
          Arc consistency maintained · Variable ordering: MRV heuristic
        </div>
      </div>
    </div>
  );
}
