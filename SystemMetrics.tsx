import { Activity, Cpu, HardDrive, Wifi, Zap } from 'lucide-react';

interface SystemMetricsProps {
  simulationState: {
    isRunning: boolean;
    nodeCount: number;
    honeypotCount: number;
    attacksDetected: number;
    accuracy: number;
    rlReward: number;
    cpuUsage: number;
    memoryUsage: number;
  };
}

export default function SystemMetrics({ simulationState }: SystemMetricsProps) {
  const metrics = [
    {
      icon: Activity,
      label: 'Detection Accuracy',
      value: `${(simulationState.accuracy * 100).toFixed(1)}%`,
      color: 'cyan',
      gradient: 'from-cyan-500 to-blue-500'
    },
    {
      icon: Zap,
      label: 'RL Agent Reward',
      value: simulationState.rlReward.toFixed(3),
      color: 'green',
      gradient: 'from-green-500 to-emerald-500'
    },
    {
      icon: Cpu,
      label: 'CPU Usage',
      value: `${simulationState.cpuUsage.toFixed(0)}%`,
      color: 'amber',
      gradient: 'from-amber-500 to-orange-500'
    },
    {
      icon: HardDrive,
      label: 'Memory Usage',
      value: `${simulationState.memoryUsage.toFixed(0)}%`,
      color: 'purple',
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      icon: Wifi,
      label: 'Active Nodes',
      value: simulationState.nodeCount.toString(),
      color: 'blue',
      gradient: 'from-blue-500 to-indigo-500'
    },
    {
      icon: Activity,
      label: 'Honeypots',
      value: simulationState.honeypotCount.toString(),
      color: 'cyan',
      gradient: 'from-cyan-500 to-teal-500'
    }
  ];

  const getProgressColor = (label: string, value: number) => {
    if (label === 'CPU Usage' || label === 'Memory Usage') {
      if (value > 80) return 'from-red-500 to-rose-500';
      if (value > 60) return 'from-amber-500 to-orange-500';
      return 'from-green-500 to-emerald-500';
    }
    return 'from-cyan-500 to-blue-500';
  };

  return (
    <div className="group relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-cyan-500/30 rounded-2xl p-6 shadow-2xl shadow-cyan-500/20 hover:shadow-cyan-500/40 transition-all duration-500 overflow-hidden">
      {/* Animated corner accents */}
      <div className="absolute top-0 right-0 w-20 h-20 border-t-2 border-r-2 border-cyan-400/50 rounded-tr-2xl" />
      <div className="absolute bottom-0 left-0 w-20 h-20 border-b-2 border-l-2 border-cyan-400/50 rounded-bl-2xl" />
      
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-lg border border-cyan-400/50">
            <Activity className="w-6 h-6 text-cyan-400 drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
          </div>
          <div>
            <h3 className="text-cyan-400 font-semibold">System Metrics</h3>
            <p className="text-xs text-gray-400">Real-time performance</p>
          </div>
        </div>
        {simulationState.isRunning && (
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse shadow-[0_0_10px_rgba(74,222,128,0.8)]" />
        )}
      </div>

      <div className="space-y-4">
        {metrics.map((metric, index) => {
          const Icon = metric.icon;
          const numValue = parseFloat(metric.value);
          const isPercentage = metric.value.includes('%');
          const progress = isPercentage ? numValue : (numValue / 100) * 100;

          return (
            <div 
              key={index}
              className="relative bg-gradient-to-br from-[#020817] to-[#0a0e27] border border-cyan-900/30 rounded-xl p-4 hover:border-cyan-500/50 transition-all duration-300 group/metric overflow-hidden"
              style={{
                animation: `fadeIn 0.5s ease-out ${index * 0.1}s both`
              }}
            >
              {/* Hover effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/0 via-cyan-500/5 to-cyan-500/0 translate-x-[-100%] group-hover/metric:translate-x-[100%] transition-transform duration-1000" />
              
              <div className="relative z-10">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div className={`p-1.5 bg-gradient-to-br from-${metric.color}-500/20 to-${metric.color}-600/20 rounded-lg border border-${metric.color}-400/50`}>
                      <Icon className={`w-4 h-4 text-${metric.color}-400`} />
                    </div>
                    <span className="text-sm text-gray-300">{metric.label}</span>
                  </div>
                  <span className={`text-lg font-bold bg-gradient-to-r ${metric.gradient} bg-clip-text text-transparent`}>
                    {metric.value}
                  </span>
                </div>
                
                {/* Progress bar for percentage values or normalized values */}
                <div className="w-full bg-[#0d1238] rounded-full h-2 overflow-hidden border border-cyan-900/30">
                  <div 
                    className={`h-full bg-gradient-to-r ${isPercentage ? getProgressColor(metric.label, numValue) : metric.gradient} rounded-full transition-all duration-1000 relative overflow-hidden`}
                    style={{ 
                      width: `${Math.min(100, progress)}%`,
                      boxShadow: `0 0 10px rgba(6, 182, 212, 0.5)`
                    }}
                  >
                    {/* Animated shine effect */}
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent translate-x-[-100%] animate-shimmer" />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Overall Status */}
      <div className="mt-4 p-4 bg-gradient-to-r from-cyan-900/30 to-blue-900/30 border border-cyan-500/50 rounded-xl">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs text-gray-400 mb-1">System Status</div>
            <div className="text-sm font-semibold text-cyan-300">
              {simulationState.isRunning ? 'Operational - All Systems Go' : 'Ready - Awaiting Commands'}
            </div>
          </div>
          <div className="flex flex-col items-end gap-1">
            <div className="text-xs text-gray-400">Uptime</div>
            <div className="text-sm font-mono text-cyan-400">
              {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
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
