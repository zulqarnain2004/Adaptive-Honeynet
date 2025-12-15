import { useState, useEffect } from 'react';
import { Brain, Zap, TrendingUp, Target } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface RLAgentPanelProps {
  rlReward: number;
  isRunning: boolean;
}

export default function RLAgentPanel({ rlReward, isRunning }: RLAgentPanelProps) {
  const [rewardHistory, setRewardHistory] = useState<any[]>([
    { episode: 0, reward: 0.12, qValue: 1.23 },
    { episode: 50, reward: 0.34, qValue: 2.45 },
    { episode: 100, reward: 0.52, qValue: 3.67 },
    { episode: 150, reward: 0.68, qValue: 4.89 },
    { episode: 200, reward: 0.79, qValue: 5.91 },
    { episode: 250, reward: 0.85, qValue: 6.54 }
  ]);

  const [agentParams, setAgentParams] = useState({
    epsilon: 0.08,
    alpha: 0.001,
    gamma: 0.99,
    episode: 250
  });

  const [recentActions, setRecentActions] = useState([
    { action: 'Deploy Honeypot Node-12', reward: 0.89, qValue: 6.72, time: '14:23:45' },
    { action: 'Migrate Honeypot Node-7', reward: 0.76, qValue: 6.45, time: '14:22:18' },
    { action: 'Reconfigure Service Profile', reward: 0.82, qValue: 6.58, time: '14:21:03' }
  ]);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      // Update reward history
      setRewardHistory(prev => {
        const lastEpisode = prev[prev.length - 1];
        const newEpisode = {
          episode: lastEpisode.episode + 50,
          reward: Math.min(0.95, lastEpisode.reward + 0.01 + Math.random() * 0.02),
          qValue: Math.min(8, lastEpisode.qValue + 0.1 + Math.random() * 0.3)
        };
        return [...prev.slice(-5), newEpisode];
      });

      // Update parameters
      setAgentParams(prev => ({
        ...prev,
        epsilon: Math.max(0.01, prev.epsilon * 0.995),
        episode: prev.episode + 50
      }));

      // Add new action occasionally
      if (Math.random() > 0.7) {
        const actions = [
          'Deploy Honeypot',
          'Migrate Honeypot',
          'Reconfigure Service',
          'Adjust Deception Profile',
          'Update Network Topology'
        ];
        const newAction = {
          action: `${actions[Math.floor(Math.random() * actions.length)]} Node-${Math.floor(Math.random() * 20)}`,
          reward: 0.6 + Math.random() * 0.35,
          qValue: 5 + Math.random() * 2,
          time: new Date().toLocaleTimeString()
        };
        setRecentActions(prev => [newAction, ...prev].slice(0, 3));
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [isRunning]);

  return (
    <div className="group relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-green-500/30 rounded-2xl p-6 shadow-2xl shadow-green-500/20 hover:shadow-green-500/40 transition-all duration-500 overflow-hidden">
      {/* Animated corner accents */}
      <div className="absolute top-0 left-0 w-20 h-20 border-t-2 border-l-2 border-green-400/50 rounded-tl-2xl" />
      <div className="absolute bottom-0 right-0 w-20 h-20 border-b-2 border-r-2 border-green-400/50 rounded-br-2xl" />
      
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="relative p-2 bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded-lg border border-green-400/50">
            <Brain className="w-6 h-6 text-green-400 drop-shadow-[0_0_8px_rgba(34,197,94,0.8)]" />
            {isRunning && (
              <div className="absolute inset-0 bg-green-400/30 rounded-lg animate-pulse" />
            )}
          </div>
          <div>
            <h3 className="text-green-400 font-semibold">RL Agent (Q-Learning)</h3>
            <p className="text-xs text-gray-400">Adaptive deployment strategy</p>
          </div>
        </div>
        <div className="px-3 py-1 bg-green-900/30 rounded-lg border border-green-500/50">
          <span className="text-xs text-green-300">Episode {agentParams.episode}</span>
        </div>
      </div>

      {/* Agent Parameters */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        {[
          { label: 'ε (Epsilon)', value: agentParams.epsilon.toFixed(3), color: 'cyan', desc: 'Exploration' },
          { label: 'α (Alpha)', value: agentParams.alpha.toFixed(3), color: 'blue', desc: 'Learning Rate' },
          { label: 'γ (Gamma)', value: agentParams.gamma.toFixed(2), color: 'purple', desc: 'Discount' },
          { label: 'Reward', value: rlReward.toFixed(3), color: 'green', desc: 'Current' }
        ].map((param, index) => (
          <div 
            key={index}
            className="bg-gradient-to-br from-[#020817] to-[#0a0e27] border border-green-900/30 rounded-xl p-3 hover:border-green-500/50 transition-all duration-300 hover:scale-105"
          >
            <div className="text-xs text-gray-400 mb-1">{param.label}</div>
            <div className={`text-lg font-bold text-${param.color}-400`}>{param.value}</div>
            <div className="text-xs text-gray-600 mt-0.5">{param.desc}</div>
          </div>
        ))}
      </div>

      {/* Reward Progression Chart */}
      <div className="bg-gradient-to-br from-[#020817] to-[#0a0e27] border-2 border-green-900/30 rounded-xl p-4 mb-4">
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-4 h-4 text-green-400" />
          <span className="text-sm font-semibold text-green-400">Reward Progression</span>
        </div>
        
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={rewardHistory}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" opacity={0.3} />
            <XAxis 
              dataKey="episode" 
              stroke="#6b7280" 
              tick={{ fill: '#9ca3af', fontSize: 10 }}
              label={{ value: 'Episode', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 11 }}
            />
            <YAxis 
              stroke="#6b7280" 
              tick={{ fill: '#9ca3af', fontSize: 10 }}
              domain={[0, 1]}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#0d1238', 
                border: '2px solid rgba(34, 197, 94, 0.5)',
                borderRadius: '12px',
                fontSize: '12px'
              }}
            />
            <Legend wrapperStyle={{ fontSize: '11px' }} />
            <Line 
              type="monotone" 
              dataKey="reward" 
              stroke="#22c55e" 
              strokeWidth={3}
              dot={{ fill: '#22c55e', r: 4, strokeWidth: 2, stroke: '#0a0e27' }}
              name="Episode Reward"
              style={{ filter: 'drop-shadow(0 0 6px rgba(34, 197, 94, 0.6))' }}
            />
            <Line 
              type="monotone" 
              dataKey="qValue" 
              stroke="#06b6d4" 
              strokeWidth={2}
              dot={{ fill: '#06b6d4', r: 3 }}
              name="Q-Value"
              yAxisId={1}
              opacity={0.7}
            />
            <YAxis 
              yAxisId={1}
              orientation="right"
              stroke="#6b7280" 
              tick={{ fill: '#9ca3af', fontSize: 10 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Recent Actions */}
      <div className="bg-gradient-to-br from-[#020817] to-[#0a0e27] border-2 border-green-900/30 rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <Zap className="w-4 h-4 text-green-400" />
          <span className="text-sm font-semibold text-green-400">Recent Agent Decisions</span>
        </div>
        
        <div className="space-y-2">
          {recentActions.map((action, index) => (
            <div 
              key={index}
              className="bg-gradient-to-r from-green-900/20 to-emerald-900/20 border border-green-500/30 rounded-lg p-3 hover:border-green-500/50 transition-all duration-300"
              style={{
                animation: `fadeInLeft 0.5s ease-out ${index * 0.1}s both`
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Target className="w-3 h-3 text-green-400" />
                  <span className="text-xs text-gray-300 font-medium">{action.action}</span>
                </div>
                <span className="text-xs text-gray-500 font-mono">{action.time}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <div className="flex gap-4">
                  <div>
                    <span className="text-gray-500">Reward:</span>
                    <span className="ml-1 text-green-400 font-bold">{action.reward.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Q-Value:</span>
                    <span className="ml-1 text-cyan-400 font-bold">{action.qValue.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Algorithm Info */}
      <div className="mt-4 p-3 bg-gradient-to-r from-green-900/30 to-emerald-900/30 border border-green-500/50 rounded-xl">
        <div className="text-xs text-gray-300">
          <span className="text-green-400 font-semibold">Q-Learning</span> with ε-greedy exploration
          · Bellman optimality equation · Experience replay buffer
        </div>
      </div>

      {isRunning && (
        <div className="mt-3 flex items-center justify-center gap-2 text-xs text-gray-400">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse shadow-[0_0_10px_rgba(34,197,94,0.8)]" />
          <span>Agent learning in progress</span>
        </div>
      )}

      <style>{`
        @keyframes fadeInLeft {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
      `}</style>
    </div>
  );
}
