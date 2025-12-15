import { useState, useEffect } from 'react';
import { Shield, AlertTriangle, Activity, TrendingUp } from 'lucide-react';

interface Attack {
  id: number;
  type: string;
  source: string;
  target: string;
  timestamp: string;
  confidence: number;
  model: string;
}

interface AttackVisualizationProps {
  attacks: Attack[];
  isRunning: boolean;
}

export default function AttackVisualization({ attacks, isRunning }: AttackVisualizationProps) {
  const [recentAttacks, setRecentAttacks] = useState<Attack[]>([]);
  const [attackStats, setAttackStats] = useState({
    total: 0,
    blocked: 0,
    analyzed: 0,
    highSeverity: 0
  });

  useEffect(() => {
    setRecentAttacks(attacks.slice(0, 6));
    
    setAttackStats({
      total: attacks.length,
      blocked: Math.floor(attacks.length * 0.87),
      analyzed: attacks.length,
      highSeverity: attacks.filter(a => a.confidence > 0.85).length
    });
  }, [attacks]);

  const getAttackTypeColor = (type: string) => {
    const colors: { [key: string]: string } = {
      'Port Scan': 'from-yellow-500/20 to-orange-500/20 border-yellow-500/50 text-yellow-300',
      'DoS Attack': 'from-red-500/20 to-rose-500/20 border-red-500/50 text-red-300',
      'Brute Force': 'from-purple-500/20 to-pink-500/20 border-purple-500/50 text-purple-300',
      'Reconnaissance': 'from-blue-500/20 to-cyan-500/20 border-blue-500/50 text-blue-300',
      'Exploits': 'from-red-600/20 to-orange-600/20 border-red-600/50 text-red-400'
    };
    return colors[type] || 'from-gray-500/20 to-gray-600/20 border-gray-500/50 text-gray-300';
  };

  const getAttackIcon = (confidence: number) => {
    if (confidence > 0.9) return '🔴';
    if (confidence > 0.8) return '🟠';
    return '🟡';
  };

  return (
    <div className="group relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-red-500/30 rounded-2xl p-6 shadow-2xl shadow-red-500/20 hover:shadow-red-500/40 transition-all duration-500 overflow-hidden">
      {/* Animated corner accents */}
      <div className="absolute top-0 right-0 w-20 h-20 border-t-2 border-r-2 border-red-400/50 rounded-tr-2xl" />
      <div className="absolute bottom-0 left-0 w-20 h-20 border-b-2 border-l-2 border-red-400/50 rounded-bl-2xl" />
      
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="relative p-2 bg-gradient-to-br from-red-500/20 to-rose-500/20 rounded-lg border border-red-400/50">
            <AlertTriangle className="w-6 h-6 text-red-400 drop-shadow-[0_0_8px_rgba(239,68,68,0.8)]" />
            {isRunning && (
              <div className="absolute inset-0 bg-red-400/30 rounded-lg animate-ping" />
            )}
          </div>
          <div>
            <h3 className="text-red-400 font-semibold">Live Attack Feed</h3>
            <p className="text-xs text-gray-400">Real-time threat detection</p>
          </div>
        </div>
        {isRunning && (
          <div className="flex items-center gap-2 px-3 py-1 bg-red-900/30 rounded-lg border border-red-500/50 animate-pulse">
            <Activity className="w-4 h-4 text-red-400" />
            <span className="text-xs text-red-300">ACTIVE</span>
          </div>
        )}
      </div>

      {/* Attack Statistics */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        <div className="relative bg-gradient-to-br from-red-900/30 to-rose-900/30 border border-red-500/40 rounded-xl p-3 overflow-hidden group/stat hover:scale-105 transition-transform">
          <div className="absolute inset-0 bg-gradient-to-r from-red-500/0 via-red-500/20 to-red-500/0 translate-x-[-100%] group-hover/stat:translate-x-[100%] transition-transform duration-1000" />
          <div className="text-xs text-gray-400 mb-1">Total Attacks</div>
          <div className="text-2xl font-bold text-red-400">{attackStats.total}</div>
        </div>
        
        <div className="relative bg-gradient-to-br from-green-900/30 to-emerald-900/30 border border-green-500/40 rounded-xl p-3 overflow-hidden group/stat hover:scale-105 transition-transform">
          <div className="absolute inset-0 bg-gradient-to-r from-green-500/0 via-green-500/20 to-green-500/0 translate-x-[-100%] group-hover/stat:translate-x-[100%] transition-transform duration-1000" />
          <div className="text-xs text-gray-400 mb-1">Blocked</div>
          <div className="text-2xl font-bold text-green-400">{attackStats.blocked}</div>
        </div>
        
        <div className="relative bg-gradient-to-br from-blue-900/30 to-cyan-900/30 border border-blue-500/40 rounded-xl p-3 overflow-hidden group/stat hover:scale-105 transition-transform">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-blue-500/20 to-blue-500/0 translate-x-[-100%] group-hover/stat:translate-x-[100%] transition-transform duration-1000" />
          <div className="text-xs text-gray-400 mb-1">Analyzed</div>
          <div className="text-2xl font-bold text-blue-400">{attackStats.analyzed}</div>
        </div>
        
        <div className="relative bg-gradient-to-br from-orange-900/30 to-amber-900/30 border border-orange-500/40 rounded-xl p-3 overflow-hidden group/stat hover:scale-105 transition-transform">
          <div className="absolute inset-0 bg-gradient-to-r from-orange-500/0 via-orange-500/20 to-orange-500/0 translate-x-[-100%] group-hover/stat:translate-x-[100%] transition-transform duration-1000" />
          <div className="text-xs text-gray-400 mb-1">High Severity</div>
          <div className="text-2xl font-bold text-orange-400">{attackStats.highSeverity}</div>
        </div>
      </div>

      {/* Attack Feed */}
      <div className="bg-gradient-to-br from-[#020817] to-[#0a0e27] border-2 border-red-900/30 rounded-xl p-4 max-h-[320px] overflow-y-auto custom-scrollbar">
        {recentAttacks.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-[280px] text-gray-500">
            <Shield className="w-16 h-16 mb-3 opacity-30" />
            <p className="text-sm">No attacks detected yet</p>
            <p className="text-xs text-gray-600 mt-1">Start simulation to begin monitoring</p>
          </div>
        ) : (
          <div className="space-y-3">
            {recentAttacks.map((attack, index) => (
              <div
                key={attack.id}
                className={`relative bg-gradient-to-r ${getAttackTypeColor(attack.type)} backdrop-blur-sm border rounded-xl p-4 overflow-hidden transform hover:scale-[1.02] transition-all duration-300 cursor-pointer`}
                style={{
                  animation: `slideIn 0.5s ease-out ${index * 0.1}s both`
                }}
              >
                {/* Animated background pulse */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent translate-x-[-100%] animate-shimmer" />
                
                <div className="relative z-10">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{getAttackIcon(attack.confidence)}</span>
                      <div>
                        <div className="font-semibold text-sm">{attack.type}</div>
                        <div className="text-xs text-gray-400">
                          {new Date(attack.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-gray-400 mb-1">Confidence</div>
                      <div className="px-2 py-1 bg-black/30 rounded-lg border border-current">
                        <span className="font-bold text-sm">
                          {(attack.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3 text-xs mb-2">
                    <div>
                      <span className="text-gray-400">Source:</span>
                      <span className="ml-2 font-mono text-gray-300">{attack.source}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Target:</span>
                      <span className="ml-2 font-mono text-gray-300">{attack.target}</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between text-xs">
                    <div className="px-2 py-1 bg-black/30 rounded border border-cyan-500/30">
                      <span className="text-cyan-400">{attack.model}</span>
                    </div>
                    <div className="flex items-center gap-1 text-green-400">
                      <Shield className="w-3 h-3" />
                      <span>CAPTURED</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Real-time indicator */}
      {isRunning && recentAttacks.length > 0 && (
        <div className="mt-4 flex items-center justify-center gap-2 text-xs text-gray-400">
          <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse shadow-[0_0_10px_rgba(248,113,113,0.8)]" />
          <span>Real-time monitoring active</span>
          <TrendingUp className="w-3 h-3" />
        </div>
      )}

      <style>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(6, 182, 212, 0.1);
          border-radius: 4px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(239, 68, 68, 0.5);
          border-radius: 4px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(239, 68, 68, 0.7);
        }
      `}</style>
    </div>
  );
}
