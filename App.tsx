import { useState, useEffect, useRef } from 'react';
import { Shield, Play, Square, RotateCcw, Activity, Network, Brain, Zap, FileText, TrendingUp, AlertTriangle } from 'lucide-react';
import NetworkTopology from './components/NetworkTopology';
import AttackVisualization from './components/AttackVisualization';
import MLDetectionPanel from './components/MLDetectionPanel';
import RLAgentPanel from './components/RLAgentPanel';
import CSPResourcePanel from './components/CSPResourcePanel';
import ExplainabilityPanel from './components/ExplainabilityPanel';
import SimulationLogsTab from './components/SimulationLogsTab';
import DeploymentPathsTab from './components/DeploymentPathsTab';
import SystemMetrics from './components/SystemMetrics';

type TabType = 'dashboard' | 'logs' | 'deployment';

interface SimulationState {
  isRunning: boolean;
  nodeCount: number;
  honeypotCount: number;
  attacksDetected: number;
  accuracy: number;
  rlReward: number;
  cpuUsage: number;
  memoryUsage: number;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [simulationState, setSimulationState] = useState<SimulationState>({
    isRunning: false,
    nodeCount: 20,
    honeypotCount: 8,
    attacksDetected: 0,
    accuracy: 0,
    rlReward: 0,
    cpuUsage: 0,
    memoryUsage: 0
  });

  const [sliderValue, setSliderValue] = useState(20);
  const simulationInterval = useRef<NodeJS.Timeout | null>(null);
  const [logs, setLogs] = useState<any[]>([]);
  const [attacks, setAttacks] = useState<any[]>([]);

  // Start Simulation
  const startSimulation = () => {
    if (simulationState.isRunning) return;

    setSimulationState(prev => ({ ...prev, isRunning: true }));
    
    // Add startup log
    addLog('success', 'Simulation', 'Simulation started - Loading UNSW-NB15 dataset');
    addLog('info', 'ML Models', 'Initializing Random Forest and Logistic Regression classifiers');
    addLog('info', 'RL Agent', 'Q-Learning agent initialized with epsilon=0.1');
    addLog('info', 'CSP Solver', 'Constraint satisfaction problem solver active');
    addLog('success', 'A* Search', 'A* pathfinding algorithm ready for honeypot deployment');

    // Simulation loop
    simulationInterval.current = setInterval(() => {
      // Simulate attack detection
      if (Math.random() > 0.7) {
        const attackTypes = ['Port Scan', 'DoS Attack', 'Brute Force', 'Reconnaissance', 'Exploits'];
        const attack = {
          id: Date.now(),
          type: attackTypes[Math.floor(Math.random() * attackTypes.length)],
          source: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
          target: `Honeypot-${Math.floor(Math.random() * simulationState.honeypotCount) + 1}`,
          timestamp: new Date().toISOString(),
          confidence: 0.7 + Math.random() * 0.3,
          model: Math.random() > 0.5 ? 'Random Forest' : 'Logistic Regression'
        };
        
        setAttacks(prev => [attack, ...prev].slice(0, 100));
        addLog('critical', 'Attack Detection', `${attack.model} detected ${attack.type} from ${attack.source}`);
        
        setSimulationState(prev => ({
          ...prev,
          attacksDetected: prev.attacksDetected + 1
        }));
      }

      // Update metrics
      setSimulationState(prev => ({
        ...prev,
        accuracy: Math.min(0.98, prev.accuracy + 0.001 + Math.random() * 0.002),
        rlReward: Math.min(0.95, prev.rlReward + 0.002 + Math.random() * 0.003),
        cpuUsage: Math.max(30, Math.min(85, prev.cpuUsage + (Math.random() - 0.5) * 10)),
        memoryUsage: Math.max(40, Math.min(80, prev.memoryUsage + (Math.random() - 0.5) * 8))
      }));

      // Simulate RL decisions
      if (Math.random() > 0.85) {
        addLog('success', 'RL Agent', 'Honeypot repositioned based on Q-Learning policy');
      }

      // Simulate A* deployment
      if (Math.random() > 0.9) {
        const cost = (2 + Math.random() * 4).toFixed(1);
        addLog('info', 'A* Search', `Optimal path computed for new honeypot (cost: ${cost})`);
      }

    }, 2000);
  };

  // Stop Simulation
  const stopSimulation = () => {
    if (!simulationState.isRunning) return;

    if (simulationInterval.current) {
      clearInterval(simulationInterval.current);
      simulationInterval.current = null;
    }

    setSimulationState(prev => ({ ...prev, isRunning: false }));
    addLog('warning', 'Simulation', 'Simulation stopped by user');
  };

  // Reset Logs
  const resetLogs = () => {
    setLogs([]);
    setAttacks([]);
    setSimulationState(prev => ({
      ...prev,
      attacksDetected: 0,
      accuracy: 0,
      rlReward: 0
    }));
    addLog('info', 'System', 'Logs and attack records cleared');
  };

  // Handle slider change
  const handleSliderChange = (value: number) => {
    setSliderValue(value);
    const honeypots = Math.floor(value * 0.4); // 40% of nodes are honeypots
    setSimulationState(prev => ({
      ...prev,
      nodeCount: value,
      honeypotCount: Math.max(4, honeypots)
    }));
    addLog('info', 'Network', `Network scaled to ${value} nodes with ${Math.max(4, honeypots)} honeypots`);
  };

  // Add log helper
  const addLog = (level: string, category: string, message: string) => {
    const newLog = {
      id: Date.now() + Math.random(),
      timestamp: new Date().toISOString(),
      level,
      category,
      message
    };
    setLogs(prev => [newLog, ...prev].slice(0, 500));
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (simulationInterval.current) {
        clearInterval(simulationInterval.current);
      }
    };
  }, []);

  // Initialize with starting values
  useEffect(() => {
    setSimulationState(prev => ({
      ...prev,
      accuracy: 0.82,
      rlReward: 0.45,
      cpuUsage: 45,
      memoryUsage: 52
    }));
  }, []);

  const tabs = [
    { id: 'dashboard' as TabType, label: 'Operational Dashboard', icon: Activity },
    { id: 'logs' as TabType, label: 'Simulation Logs', icon: FileText },
    { id: 'deployment' as TabType, label: 'Deployment Paths', icon: TrendingUp }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#020817] via-[#0a0e27] to-[#0d1238] text-gray-100 overflow-hidden">
      {/* Animated background */}
      <div className="fixed inset-0 opacity-30 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      </div>

      {/* Header */}
      <header className="relative bg-gradient-to-r from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-b border-cyan-500/30 shadow-2xl shadow-cyan-500/20">
        <div className="px-8 py-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="relative">
                <Shield className="w-12 h-12 text-cyan-400 drop-shadow-[0_0_10px_rgba(6,182,212,0.8)]" />
                <div className="absolute inset-0 bg-cyan-400/20 rounded-full blur-xl animate-pulse" />
              </div>
              <div>
                <h1 className="text-2xl bg-gradient-to-r from-cyan-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent animate-gradient">
                  Adaptive Deception-Mesh Platform
                </h1>
                <p className="text-sm text-gray-400 flex items-center gap-2 mt-1">
                  <span>AI-Powered Honeynet Simulation</span>
                  <span className="text-cyan-400">·</span>
                  <span>UNSW-NB15 Dataset</span>
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-3 px-4 py-2 bg-gradient-to-r from-cyan-900/40 to-blue-900/40 rounded-xl border border-cyan-500/50 backdrop-blur-sm">
                <div className={`w-3 h-3 rounded-full ${simulationState.isRunning ? 'bg-green-400 animate-pulse shadow-[0_0_10px_rgba(74,222,128,0.8)]' : 'bg-gray-500'}`} />
                <span className="text-sm text-gray-300">
                  {simulationState.isRunning ? 'Simulation Active' : 'System Ready'}
                </span>
              </div>
              <div className="text-sm text-gray-400">
                {new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>

          {/* Control Panel */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            {/* Start Button */}
            <button
              onClick={startSimulation}
              disabled={simulationState.isRunning}
              className={`group relative flex items-center justify-center gap-3 px-6 py-4 rounded-xl border-2 transition-all duration-300 overflow-hidden ${
                simulationState.isRunning
                  ? 'bg-gray-800/50 border-gray-600 text-gray-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-green-900/40 to-emerald-900/40 border-green-500/50 text-green-300 hover:border-green-400 hover:shadow-[0_0_30px_rgba(34,197,94,0.4)] hover:scale-105'
              }`}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-green-500/0 via-green-500/20 to-green-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
              <Play className="w-5 h-5" />
              <span className="font-semibold">Start Simulation</span>
            </button>

            {/* Stop Button */}
            <button
              onClick={stopSimulation}
              disabled={!simulationState.isRunning}
              className={`group relative flex items-center justify-center gap-3 px-6 py-4 rounded-xl border-2 transition-all duration-300 overflow-hidden ${
                !simulationState.isRunning
                  ? 'bg-gray-800/50 border-gray-600 text-gray-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-red-900/40 to-rose-900/40 border-red-500/50 text-red-300 hover:border-red-400 hover:shadow-[0_0_30px_rgba(239,68,68,0.4)] hover:scale-105'
              }`}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-red-500/0 via-red-500/20 to-red-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
              <Square className="w-5 h-5" />
              <span className="font-semibold">Stop Simulation</span>
            </button>

            {/* Reset Logs Button */}
            <button
              onClick={resetLogs}
              className="group relative flex items-center justify-center gap-3 px-6 py-4 rounded-xl border-2 bg-gradient-to-r from-amber-900/40 to-orange-900/40 border-amber-500/50 text-amber-300 hover:border-amber-400 hover:shadow-[0_0_30px_rgba(251,191,36,0.4)] hover:scale-105 transition-all duration-300 overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-amber-500/0 via-amber-500/20 to-amber-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
              <RotateCcw className="w-5 h-5" />
              <span className="font-semibold">Reset Logs</span>
            </button>

            {/* Stats Display */}
            <div className="flex items-center justify-center gap-2 px-6 py-4 rounded-xl border-2 border-cyan-500/50 bg-gradient-to-r from-cyan-900/40 to-blue-900/40 backdrop-blur-sm">
              <AlertTriangle className="w-5 h-5 text-cyan-400" />
              <div>
                <div className="text-xs text-gray-400">Attacks Detected</div>
                <div className="text-xl text-cyan-400 font-bold">{simulationState.attacksDetected}</div>
              </div>
            </div>
          </div>

          {/* Network Slider */}
          <div className="bg-gradient-to-r from-[#0a0e27]/80 to-[#0d1238]/80 backdrop-blur-sm border border-cyan-500/30 rounded-xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Network className="w-5 h-5 text-cyan-400" />
                <span className="text-sm text-gray-300">Network Scale</span>
              </div>
              <div className="flex items-center gap-6">
                <div className="text-sm">
                  <span className="text-gray-400">Nodes:</span>
                  <span className="ml-2 text-cyan-400 font-bold">{simulationState.nodeCount}</span>
                </div>
                <div className="text-sm">
                  <span className="text-gray-400">Honeypots:</span>
                  <span className="ml-2 text-cyan-400 font-bold">{simulationState.honeypotCount}</span>
                </div>
              </div>
            </div>
            <div className="relative">
              <input
                type="range"
                min="10"
                max="50"
                value={sliderValue}
                onChange={(e) => handleSliderChange(parseInt(e.target.value))}
                className="w-full h-3 bg-gradient-to-r from-cyan-900/30 to-blue-900/30 rounded-full appearance-none cursor-pointer slider-thumb"
                style={{
                  background: `linear-gradient(to right, rgb(6 182 212 / 0.5) 0%, rgb(6 182 212 / 0.5) ${((sliderValue - 10) / 40) * 100}%, rgb(30 58 95 / 0.3) ${((sliderValue - 10) / 40) * 100}%, rgb(30 58 95 / 0.3) 100%)`
                }}
              />
              <div className="flex justify-between text-xs text-gray-500 mt-2">
                <span>Min (10)</span>
                <span>Max (50)</span>
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-3 mt-6">
            {tabs.map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`group relative flex items-center gap-3 px-6 py-3 rounded-xl transition-all duration-300 overflow-hidden ${
                    activeTab === tab.id
                      ? 'bg-gradient-to-r from-cyan-900/50 to-blue-900/50 border-2 border-cyan-500/70 text-cyan-300 shadow-[0_0_20px_rgba(6,182,212,0.3)]'
                      : 'bg-[#0a0e27]/50 border-2 border-transparent text-gray-400 hover:border-cyan-500/30 hover:text-cyan-300'
                  }`}
                >
                  {activeTab === tab.id && (
                    <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/0 via-cyan-500/20 to-cyan-500/0 animate-shimmer" />
                  )}
                  <Icon className="w-5 h-5 relative z-10" />
                  <span className="font-medium relative z-10">{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative p-8">
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* Top Row - Network & Attack Visualization */}
            <div className="grid grid-cols-2 gap-6">
              <NetworkTopology 
                nodeCount={simulationState.nodeCount} 
                honeypotCount={simulationState.honeypotCount}
                isRunning={simulationState.isRunning}
              />
              <AttackVisualization 
                attacks={attacks}
                isRunning={simulationState.isRunning}
              />
            </div>

            {/* Middle Row - ML Detection & System Metrics */}
            <div className="grid grid-cols-3 gap-6">
              <MLDetectionPanel 
                attacks={attacks}
                accuracy={simulationState.accuracy}
                isRunning={simulationState.isRunning}
              />
              <SystemMetrics simulationState={simulationState} />
              <ExplainabilityPanel 
                attacks={attacks}
                isRunning={simulationState.isRunning}
              />
            </div>

            {/* Bottom Row - RL Agent & CSP Resources */}
            <div className="grid grid-cols-2 gap-6">
              <RLAgentPanel 
                rlReward={simulationState.rlReward}
                isRunning={simulationState.isRunning}
              />
              <CSPResourcePanel 
                cpuUsage={simulationState.cpuUsage}
                memoryUsage={simulationState.memoryUsage}
                nodeCount={simulationState.nodeCount}
                honeypotCount={simulationState.honeypotCount}
              />
            </div>
          </div>
        )}

        {activeTab === 'logs' && <SimulationLogsTab logs={logs} />}
        {activeTab === 'deployment' && (
          <DeploymentPathsTab 
            honeypotCount={simulationState.honeypotCount}
            isRunning={simulationState.isRunning}
          />
        )}
      </main>

      <style>{`
        @keyframes gradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        
        .animate-gradient {
          background-size: 200% auto;
          animation: gradient 3s linear infinite;
        }
        
        .animate-shimmer {
          animation: shimmer 2s linear infinite;
        }
        
        .slider-thumb::-webkit-slider-thumb {
          appearance: none;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: linear-gradient(135deg, #06b6d4, #3b82f6);
          cursor: pointer;
          box-shadow: 0 0 20px rgba(6, 182, 212, 0.8);
          border: 2px solid rgba(6, 182, 212, 0.5);
          transition: all 0.2s;
        }
        
        .slider-thumb::-webkit-slider-thumb:hover {
          transform: scale(1.2);
          box-shadow: 0 0 30px rgba(6, 182, 212, 1);
        }
        
        .slider-thumb::-moz-range-thumb {
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: linear-gradient(135deg, #06b6d4, #3b82f6);
          cursor: pointer;
          box-shadow: 0 0 20px rgba(6, 182, 212, 0.8);
          border: 2px solid rgba(6, 182, 212, 0.5);
          transition: all 0.2s;
        }
        
        .slider-thumb::-moz-range-thumb:hover {
          transform: scale(1.2);
          box-shadow: 0 0 30px rgba(6, 182, 212, 1);
        }
      `}</style>
    </div>
  );
}
