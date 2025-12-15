import { useState, useEffect } from 'react';
import { Info, BarChart3, Sparkles } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface ExplainabilityPanelProps {
  attacks: any[];
  isRunning: boolean;
}

export default function ExplainabilityPanel({ attacks, isRunning }: ExplainabilityPanelProps) {
  const [method, setMethod] = useState<'SHAP' | 'LIME'>('SHAP');
  const [selectedAttack, setSelectedAttack] = useState<any>(null);

  const shapValues = [
    { feature: 'Packet Rate', value: 0.342, impact: 'Critical' },
    { feature: 'Port Diversity', value: 0.289, impact: 'High' },
    { feature: 'SSH Attempts', value: 0.198, impact: 'High' },
    { feature: 'Payload Size', value: 0.134, impact: 'Medium' },
    { feature: 'Protocol Type', value: -0.087, impact: 'Low' },
    { feature: 'Time Pattern', value: -0.052, impact: 'Low' }
  ];

  const limeValues = [
    { feature: 'Packet Rate', value: 0.367, impact: 'Critical' },
    { feature: 'SSH Attempts', value: 0.312, impact: 'Critical' },
    { feature: 'Port Diversity', value: 0.245, impact: 'High' },
    { feature: 'Payload Size', value: 0.156, impact: 'Medium' },
    { feature: 'Time Pattern', value: -0.043, impact: 'Low' },
    { feature: 'Protocol Type', value: -0.068, impact: 'Low' }
  ];

  const currentData = method === 'SHAP' ? shapValues : limeValues;

  useEffect(() => {
    if (attacks.length > 0) {
      setSelectedAttack(attacks[0]);
    }
  }, [attacks]);

  const getBarColor = (value: number, index: number) => {
    if (value > 0.3) return '#ef4444'; // Critical - Red
    if (value > 0.15) return '#f59e0b'; // High - Amber
    if (value > 0) return '#10b981'; // Medium - Green
    return '#6b7280'; // Low - Gray
  };

  const getImpactBadge = (impact: string) => {
    const styles: { [key: string]: string } = {
      'Critical': 'bg-red-900/30 text-red-300 border-red-500/50',
      'High': 'bg-amber-900/30 text-amber-300 border-amber-500/50',
      'Medium': 'bg-blue-900/30 text-blue-300 border-blue-500/50',
      'Low': 'bg-gray-800/30 text-gray-400 border-gray-600/50'
    };
    return styles[impact] || styles['Low'];
  };

  return (
    <div className="group relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-amber-500/30 rounded-2xl p-6 shadow-2xl shadow-amber-500/20 hover:shadow-amber-500/40 transition-all duration-500 overflow-hidden">
      {/* Animated corner accents */}
      <div className="absolute top-0 left-0 w-20 h-20 border-t-2 border-l-2 border-amber-400/50 rounded-tl-2xl" />
      <div className="absolute bottom-0 right-0 w-20 h-20 border-b-2 border-r-2 border-amber-400/50 rounded-br-2xl" />
      
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-amber-500/20 to-orange-500/20 rounded-lg border border-amber-400/50">
            <Sparkles className="w-6 h-6 text-amber-400 drop-shadow-[0_0_8px_rgba(251,191,36,0.8)]" />
          </div>
          <div>
            <h3 className="text-amber-400 font-semibold">XAI Explainability</h3>
            <p className="text-xs text-gray-400">Model interpretability</p>
          </div>
        </div>
        
        {/* Method Selector */}
        <div className="flex gap-2">
          <button
            onClick={() => setMethod('SHAP')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-300 ${
              method === 'SHAP'
                ? 'bg-gradient-to-r from-amber-500/30 to-orange-500/30 text-amber-300 border-2 border-amber-500/50 shadow-[0_0_15px_rgba(251,191,36,0.3)]'
                : 'bg-[#0a0e27] text-gray-400 border-2 border-transparent hover:border-amber-500/30'
            }`}
          >
            SHAP
          </button>
          <button
            onClick={() => setMethod('LIME')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-300 ${
              method === 'LIME'
                ? 'bg-gradient-to-r from-amber-500/30 to-orange-500/30 text-amber-300 border-2 border-amber-500/50 shadow-[0_0_15px_rgba(251,191,36,0.3)]'
                : 'bg-[#0a0e27] text-gray-400 border-2 border-transparent hover:border-amber-500/30'
            }`}
          >
            LIME
          </button>
        </div>
      </div>

      {/* Feature Importance Chart */}
      <div className="bg-gradient-to-br from-[#020817] to-[#0a0e27] border-2 border-amber-900/30 rounded-xl p-4 mb-4">
        <div className="flex items-center gap-2 mb-3">
          <BarChart3 className="w-4 h-4 text-amber-400" />
          <span className="text-sm font-semibold text-amber-400">
            {method} Feature Importance
          </span>
        </div>

        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={currentData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" opacity={0.3} />
            <XAxis 
              type="number" 
              stroke="#6b7280" 
              tick={{ fill: '#9ca3af', fontSize: 10 }}
              domain={[-0.1, 0.4]}
            />
            <YAxis 
              type="category" 
              dataKey="feature" 
              stroke="#6b7280" 
              tick={{ fill: '#9ca3af', fontSize: 10 }}
              width={100}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#0d1238', 
                border: '2px solid rgba(251, 191, 36, 0.5)',
                borderRadius: '12px',
                fontSize: '12px',
                padding: '8px 12px'
              }}
              formatter={(value: number) => [value.toFixed(3), 'Importance']}
            />
            <Bar dataKey="value" radius={[0, 8, 8, 0]}>
              {currentData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={getBarColor(entry.value, index)}
                  style={{ filter: `drop-shadow(0 0 6px ${getBarColor(entry.value, index)})` }}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Feature Details */}
      <div className="space-y-2">
        <div className="flex items-center gap-2 mb-2">
          <Info className="w-4 h-4 text-amber-400" />
          <span className="text-xs font-semibold text-amber-400">Feature Analysis</span>
        </div>
        
        <div className="space-y-2 max-h-[140px] overflow-y-auto custom-scrollbar">
          {currentData.slice(0, 4).map((item, index) => (
            <div 
              key={index}
              className="bg-gradient-to-r from-[#0a0e27] to-[#0d1238] border border-amber-900/30 rounded-lg p-3 hover:border-amber-500/50 transition-all duration-300"
              style={{
                animation: `slideUp 0.4s ease-out ${index * 0.1}s both`
              }}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-gray-300">{item.feature}</span>
                <span className={`text-xs px-2 py-0.5 rounded border ${getImpactBadge(item.impact)}`}>
                  {item.impact}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-[#020817] rounded-full h-1.5 overflow-hidden border border-amber-900/30">
                  <div 
                    className="h-full rounded-full transition-all duration-1000"
                    style={{ 
                      width: `${Math.abs(item.value) * 250}%`,
                      background: `linear-gradient(to right, ${getBarColor(item.value, index)}, ${getBarColor(item.value, index)}dd)`,
                      boxShadow: `0 0 8px ${getBarColor(item.value, index)}`
                    }}
                  />
                </div>
                <span className="text-xs font-mono text-amber-400 w-12 text-right">
                  {item.value > 0 ? '+' : ''}{item.value.toFixed(3)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Method Info */}
      <div className="mt-4 p-3 bg-gradient-to-r from-amber-900/30 to-orange-900/30 border border-amber-500/50 rounded-xl">
        <div className="text-xs text-gray-300">
          <span className="text-amber-400 font-semibold">{method}</span>
          {' '}
          {method === 'SHAP' 
            ? 'provides global feature importance using Shapley values from game theory'
            : 'explains individual predictions with local linear approximations'
          }
        </div>
      </div>

      {isRunning && (
        <div className="mt-3 flex items-center justify-center gap-2 text-xs text-gray-400">
          <div className="w-2 h-2 bg-amber-400 rounded-full animate-pulse shadow-[0_0_10px_rgba(251,191,36,0.8)]" />
          <span>Real-time analysis active</span>
        </div>
      )}

      <style>{`
        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(251, 191, 36, 0.1);
          border-radius: 3px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(251, 191, 36, 0.4);
          border-radius: 3px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(251, 191, 36, 0.6);
        }
      `}</style>
    </div>
  );
}
