import { useState, useEffect } from 'react';
import { Brain, Cpu, TrendingUp } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

interface MLDetectionPanelProps {
  attacks: any[];
  accuracy: number;
  isRunning: boolean;
}

export default function MLDetectionPanel({ attacks, accuracy, isRunning }: MLDetectionPanelProps) {
  const [modelMetrics, setModelMetrics] = useState({
    randomForest: { accuracy: 0.96, precision: 0.94, recall: 0.97, f1: 0.95 },
    logisticRegression: { accuracy: 0.93, precision: 0.91, recall: 0.94, f1: 0.92 }
  });

  const [clusterData, setClusterData] = useState([
    { name: 'Normal Traffic', value: 45, color: '#10b981' },
    { name: 'Port Scans', value: 25, color: '#f59e0b' },
    { name: 'DoS Attacks', value: 18, color: '#ef4444' },
    { name: 'Exploits', value: 12, color: '#8b5cf6' }
  ]);

  const [detectionRate, setDetectionRate] = useState(0);

  useEffect(() => {
    if (isRunning && attacks.length > 0) {
      const rfCount = attacks.filter(a => a.model === 'Random Forest').length;
      const lrCount = attacks.filter(a => a.model === 'Logistic Regression').length;
      const total = attacks.length;
      
      setDetectionRate((total / (total + 10)) * 100); // Simulated detection rate

      // Update metrics slightly
      setModelMetrics(prev => ({
        randomForest: {
          ...prev.randomForest,
          accuracy: Math.min(0.98, prev.randomForest.accuracy + 0.001)
        },
        logisticRegression: {
          ...prev.logisticRegression,
          accuracy: Math.min(0.96, prev.logisticRegression.accuracy + 0.001)
        }
      }));
    }
  }, [attacks, isRunning]);

  return (
    <div className="group relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-purple-500/30 rounded-2xl p-6 shadow-2xl shadow-purple-500/20 hover:shadow-purple-500/40 transition-all duration-500 overflow-hidden">
      {/* Animated corner accents */}
      <div className="absolute top-0 left-0 w-20 h-20 border-t-2 border-l-2 border-purple-400/50 rounded-tl-2xl" />
      <div className="absolute bottom-0 right-0 w-20 h-20 border-b-2 border-r-2 border-purple-400/50 rounded-br-2xl" />
      
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-lg border border-purple-400/50">
            <Brain className="w-6 h-6 text-purple-400 drop-shadow-[0_0_8px_rgba(168,85,247,0.8)]" />
          </div>
          <div>
            <h3 className="text-purple-400 font-semibold">ML Detection Models</h3>
            <p className="text-xs text-gray-400">UNSW-NB15 trained</p>
          </div>
        </div>
      </div>

      {/* Model Performance */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gradient-to-br from-cyan-900/30 to-blue-900/30 border border-cyan-500/40 rounded-xl p-4 hover:scale-105 transition-transform">
          <div className="flex items-center gap-2 mb-3">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-semibold text-cyan-400">Random Forest</span>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-gray-400">Accuracy</span>
              <span className="text-cyan-300 font-bold">{(modelMetrics.randomForest.accuracy * 100).toFixed(2)}%</span>
            </div>
            <div className="w-full bg-[#020817] rounded-full h-2 overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-1000"
                style={{ width: `${modelMetrics.randomForest.accuracy * 100}%` }}
              />
            </div>
            <div className="grid grid-cols-3 gap-2 pt-2 text-xs">
              <div>
                <span className="text-gray-500">Prec</span>
                <div className="text-cyan-400 font-semibold">{modelMetrics.randomForest.precision.toFixed(2)}</div>
              </div>
              <div>
                <span className="text-gray-500">Recall</span>
                <div className="text-cyan-400 font-semibold">{modelMetrics.randomForest.recall.toFixed(2)}</div>
              </div>
              <div>
                <span className="text-gray-500">F1</span>
                <div className="text-cyan-400 font-semibold">{modelMetrics.randomForest.f1.toFixed(2)}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-900/30 to-emerald-900/30 border border-green-500/40 rounded-xl p-4 hover:scale-105 transition-transform">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-green-400" />
            <span className="text-sm font-semibold text-green-400">Logistic Regression</span>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-gray-400">Accuracy</span>
              <span className="text-green-300 font-bold">{(modelMetrics.logisticRegression.accuracy * 100).toFixed(2)}%</span>
            </div>
            <div className="w-full bg-[#020817] rounded-full h-2 overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-green-500 to-emerald-500 rounded-full transition-all duration-1000"
                style={{ width: `${modelMetrics.logisticRegression.accuracy * 100}%` }}
              />
            </div>
            <div className="grid grid-cols-3 gap-2 pt-2 text-xs">
              <div>
                <span className="text-gray-500">Prec</span>
                <div className="text-green-400 font-semibold">{modelMetrics.logisticRegression.precision.toFixed(2)}</div>
              </div>
              <div>
                <span className="text-gray-500">Recall</span>
                <div className="text-green-400 font-semibold">{modelMetrics.logisticRegression.recall.toFixed(2)}</div>
              </div>
              <div>
                <span className="text-gray-500">F1</span>
                <div className="text-green-400 font-semibold">{modelMetrics.logisticRegression.f1.toFixed(2)}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* K-Means Clustering Visualization */}
      <div className="bg-gradient-to-br from-[#020817] to-[#0a0e27] border-2 border-purple-900/30 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-semibold text-purple-400">K-Means Clustering</span>
          <span className="text-xs px-2 py-1 bg-purple-900/30 rounded border border-purple-500/50 text-purple-300">
            4 Clusters
          </span>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie
                data={clusterData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={65}
                paddingAngle={4}
                dataKey="value"
              >
                {clusterData.map((entry, index) => (
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
                  border: '1px solid rgba(168, 85, 247, 0.5)',
                  borderRadius: '8px',
                  fontSize: '12px'
                }}
              />
            </PieChart>
          </ResponsiveContainer>

          <div className="space-y-2">
            {clusterData.map((cluster, index) => (
              <div key={index} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ 
                      backgroundColor: cluster.color,
                      boxShadow: `0 0 8px ${cluster.color}`
                    }}
                  />
                  <span className="text-gray-300">{cluster.name}</span>
                </div>
                <span className="font-bold text-gray-400">{cluster.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Detection Rate Indicator */}
      {isRunning && (
        <div className="mt-4 p-3 bg-gradient-to-r from-purple-900/30 to-pink-900/30 border border-purple-500/50 rounded-xl">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-400">Real-time Detection Rate</span>
            <span className="text-sm font-bold text-purple-300">{detectionRate.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-[#020817] rounded-full h-2 overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-purple-500 via-pink-500 to-purple-500 rounded-full animate-pulse"
              style={{ 
                width: `${detectionRate}%`,
                boxShadow: '0 0 10px rgba(168, 85, 247, 0.8)'
              }}
            />
          </div>
        </div>
      )}

      <div className="mt-4 text-xs text-center text-gray-500">
        Ensemble learning with UNSW-NB15 dataset
      </div>
    </div>
  );
}
