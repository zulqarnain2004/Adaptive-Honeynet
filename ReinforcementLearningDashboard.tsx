import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingUp, Cpu } from 'lucide-react';

interface RewardData {
  episode: number;
  reward: number;
  cumulativeReward: number;
}

interface Action {
  id: string;
  timestamp: string;
  action: string;
  state: string;
  reward: number;
  qValue: number;
}

export default function ReinforcementLearningDashboard() {
  const [rewardData, setRewardData] = useState<RewardData[]>([
    { episode: 0, reward: 0.12, cumulativeReward: 0.12 },
    { episode: 10, reward: 0.23, cumulativeReward: 1.45 },
    { episode: 20, reward: 0.45, cumulativeReward: 4.23 },
    { episode: 30, reward: 0.62, cumulativeReward: 8.67 },
    { episode: 40, reward: 0.78, cumulativeReward: 14.89 },
    { episode: 50, reward: 0.85, cumulativeReward: 22.34 },
    { episode: 60, reward: 0.91, cumulativeReward: 31.12 },
    { episode: 70, reward: 0.94, cumulativeReward: 40.45 },
    { episode: 80, reward: 0.96, cumulativeReward: 50.23 },
  ]);

  const [recentActions, setRecentActions] = useState<Action[]>([
    {
      id: '1',
      timestamp: new Date().toLocaleTimeString(),
      action: 'Deploy Honeypot at Node-7',
      state: 'High threat detected',
      reward: 0.87,
      qValue: 2.34
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 5000).toLocaleTimeString(),
      action: 'Reconfigure Honey-3',
      state: 'Medium activity',
      reward: 0.62,
      qValue: 1.89
    }
  ]);

  const [agentMetrics, setAgentMetrics] = useState({
    epsilon: 0.05,
    learningRate: 0.001,
    discountFactor: 0.99,
    totalSteps: 8240
  });

  // Simulate RL agent updates
  useEffect(() => {
    const interval = setInterval(() => {
      const newEpisode = rewardData[rewardData.length - 1].episode + 10;
      const newReward = Math.min(0.98, rewardData[rewardData.length - 1].reward + (Math.random() * 0.04 - 0.01));
      const newCumulative = rewardData[rewardData.length - 1].cumulativeReward + newReward * 10;

      setRewardData(prev => [...prev, {
        episode: newEpisode,
        reward: newReward,
        cumulativeReward: newCumulative
      }].slice(-9));

      setAgentMetrics(prev => ({
        ...prev,
        epsilon: Math.max(0.01, prev.epsilon * 0.995),
        totalSteps: prev.totalSteps + Math.floor(Math.random() * 50 + 10)
      }));

      // Add new action
      const actions = [
        'Deploy Honeypot',
        'Reconfigure Service',
        'Adjust Traffic Pattern',
        'Update Decoy Profile',
        'Migrate Honeypot'
      ];
      const states = [
        'High threat detected',
        'Scan activity',
        'Normal traffic',
        'Suspicious behavior'
      ];

      const newAction: Action = {
        id: Date.now().toString(),
        timestamp: new Date().toLocaleTimeString(),
        action: actions[Math.floor(Math.random() * actions.length)],
        state: states[Math.floor(Math.random() * states.length)],
        reward: 0.5 + Math.random() * 0.5,
        qValue: 1 + Math.random() * 2
      };

      setRecentActions(prev => [newAction, ...prev].slice(0, 4));
    }, 6000);

    return () => clearInterval(interval);
  }, [rewardData]);

  return (
    <div className="space-y-4">
      {/* Agent Metrics */}
      <div className="grid grid-cols-4 gap-2">
        <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-2">
          <div className="text-xs text-gray-400">Epsilon</div>
          <div className="text-cyan-400 text-sm">{agentMetrics.epsilon.toFixed(3)}</div>
        </div>
        <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-2">
          <div className="text-xs text-gray-400">Alpha</div>
          <div className="text-cyan-400 text-sm">{agentMetrics.learningRate.toFixed(3)}</div>
        </div>
        <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-2">
          <div className="text-xs text-gray-400">Gamma</div>
          <div className="text-cyan-400 text-sm">{agentMetrics.discountFactor.toFixed(2)}</div>
        </div>
        <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-2">
          <div className="text-xs text-gray-400">Steps</div>
          <div className="text-cyan-400 text-sm">{agentMetrics.totalSteps}</div>
        </div>
      </div>

      {/* Reward Progression Chart */}
      <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-cyan-400">Reward Progression (Q-Learning)</span>
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={rewardData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
            <XAxis 
              dataKey="episode" 
              stroke="#6b7280" 
              tick={{ fill: '#9ca3af', fontSize: 10 }}
              label={{ value: 'Episode', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 11 }}
            />
            <YAxis 
              stroke="#6b7280" 
              tick={{ fill: '#9ca3af', fontSize: 10 }}
              label={{ value: 'Reward', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 11 }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#0d1238', 
                border: '1px solid #0e7490',
                borderRadius: '8px',
                fontSize: '12px'
              }}
            />
            <Legend 
              wrapperStyle={{ fontSize: '11px' }}
              iconType="line"
            />
            <Line 
              type="monotone" 
              dataKey="reward" 
              stroke="#06b6d4" 
              strokeWidth={2}
              dot={{ fill: '#06b6d4', r: 3 }}
              name="Episode Reward"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Recent Agent Actions */}
      <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-cyan-400">Recent Agent Decisions</span>
        </div>
        <div className="space-y-2">
          {recentActions.map(action => (
            <div 
              key={action.id}
              className="bg-[#0d1238] border border-cyan-900/20 rounded p-2"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-300">{action.action}</span>
                <span className="text-xs text-gray-500">{action.timestamp}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-400">State: {action.state}</span>
                <div className="flex gap-3">
                  <span className="text-cyan-400">R: {action.reward.toFixed(2)}</span>
                  <span className="text-green-400">Q: {action.qValue.toFixed(2)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* RL Algorithm Info */}
      <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
        <div className="text-xs text-gray-400 mb-1">Algorithm: Q-Learning with Experience Replay</div>
        <div className="text-xs text-gray-500">
          Policy: ε-greedy exploration · Bellman equation optimization
        </div>
      </div>
    </div>
  );
}
