import { useState, useEffect } from 'react';
import { Search, Download, Filter, AlertTriangle, Info, CheckCircle, Shield } from 'lucide-react';

interface LogEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'critical' | 'success';
  category: string;
  message: string;
  details?: string;
}

export default function SimulationLogs() {
  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: '1',
      timestamp: new Date().toISOString(),
      level: 'critical',
      category: 'Attack Detection',
      message: 'Random Forest detected malicious traffic from 192.168.1.45',
      details: 'Classification confidence: 94.3% | Features: High packet rate, port scanning'
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 5000).toISOString(),
      level: 'success',
      category: 'RL Agent',
      message: 'Honeypot deployed at optimal location (Node-12)',
      details: 'A* pathfinding cost: 3.2 | Reward: +0.87 | Q-value: 2.34'
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 12000).toISOString(),
      level: 'warning',
      category: 'CSP',
      message: 'CPU utilization approaching constraint threshold',
      details: 'Current: 82% | Threshold: 85% | Constraint status: Satisfied'
    },
    {
      id: '4',
      timestamp: new Date(Date.now() - 18000).toISOString(),
      level: 'info',
      category: 'Network',
      message: 'Network topology updated with 3 new nodes',
      details: 'Total nodes: 24 | Active honeypots: 12 | Routers: 4'
    },
    {
      id: '5',
      timestamp: new Date(Date.now() - 25000).toISOString(),
      level: 'success',
      category: 'ML Model',
      message: 'SHAP explainability analysis completed',
      details: 'Top features: Packet rate (0.32), Port diversity (0.28), SSH attempts (0.19)'
    },
    {
      id: '6',
      timestamp: new Date(Date.now() - 32000).toISOString(),
      level: 'info',
      category: 'RL Agent',
      message: 'Q-Learning episode 90 completed',
      details: 'Episode reward: 0.96 | Cumulative reward: 50.23 | Epsilon: 0.048'
    },
    {
      id: '7',
      timestamp: new Date(Date.now() - 38000).toISOString(),
      level: 'critical',
      category: 'Attack Detection',
      message: 'Logistic Regression flagged SSH brute force attempt',
      details: 'Source: 10.0.5.128 | Target: Honeypot-7 | Confidence: 89.2%'
    },
    {
      id: '8',
      timestamp: new Date(Date.now() - 45000).toISOString(),
      level: 'success',
      category: 'Deployment',
      message: 'A* algorithm computed optimal honeypot placement',
      details: 'Path length: 4 hops | Total cost: 5.8 | Heuristic: Manhattan distance'
    }
  ]);

  const [searchTerm, setSearchTerm] = useState('');
  const [filterLevel, setFilterLevel] = useState<string>('all');
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [selectedLog, setSelectedLog] = useState<LogEntry | null>(null);

  // Simulate real-time log generation
  useEffect(() => {
    const interval = setInterval(() => {
      const levels: Array<'info' | 'warning' | 'critical' | 'success'> = ['info', 'warning', 'critical', 'success'];
      const categories = ['Attack Detection', 'RL Agent', 'CSP', 'Network', 'ML Model', 'Deployment'];
      const messages = [
        'Threat actor identified by ensemble model',
        'Reinforcement learning agent adjusted deployment strategy',
        'Constraint satisfaction problem resolved',
        'Network topology reconfigured',
        'LIME analysis provided prediction explanation',
        'A* search optimized honeypot location',
        'Resource allocation updated via CSP solver',
        'Random Forest achieved 96.2% accuracy on test set'
      ];

      const newLog: LogEntry = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        level: levels[Math.floor(Math.random() * levels.length)],
        category: categories[Math.floor(Math.random() * categories.length)],
        message: messages[Math.floor(Math.random() * messages.length)],
        details: `Event ID: ${Math.random().toString(36).substr(2, 9)} | Generated at ${new Date().toLocaleTimeString()}`
      };

      setLogs(prev => [newLog, ...prev].slice(0, 100));
    }, 8000);

    return () => clearInterval(interval);
  }, []);

  const categories = ['all', ...Array.from(new Set(logs.map(log => log.category)))];
  const levels = ['all', 'info', 'success', 'warning', 'critical'];

  const filteredLogs = logs.filter(log => {
    const matchesSearch = searchTerm === '' || 
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.category.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesLevel = filterLevel === 'all' || log.level === filterLevel;
    const matchesCategory = filterCategory === 'all' || log.category === filterCategory;
    return matchesSearch && matchesLevel && matchesCategory;
  });

  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'critical':
        return <AlertTriangle className="w-4 h-4 text-red-400" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-amber-400" />;
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      default:
        return <Info className="w-4 h-4 text-cyan-400" />;
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'critical':
        return 'bg-red-900/30 border-red-500/50 text-red-300';
      case 'warning':
        return 'bg-amber-900/30 border-amber-500/50 text-amber-300';
      case 'success':
        return 'bg-green-900/30 border-green-500/50 text-green-300';
      default:
        return 'bg-cyan-900/30 border-cyan-500/50 text-cyan-300';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  return (
    <div className="bg-[#0d1238] border border-cyan-900/30 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Shield className="w-6 h-6 text-cyan-400" />
          <div>
            <h2 className="text-cyan-400">Simulation Event Logs</h2>
            <p className="text-sm text-gray-400">Chronological system activity tracking</p>
          </div>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-cyan-900/30 text-cyan-300 rounded-lg border border-cyan-500/50 hover:bg-cyan-900/50 transition-colors">
          <Download className="w-4 h-4" />
          <span className="text-sm">Export Logs</span>
        </button>
      </div>

      {/* Filters */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search logs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-[#0a0e27] border border-cyan-900/30 rounded-lg text-gray-300 placeholder-gray-500 focus:outline-none focus:border-cyan-500/50 text-sm"
          />
        </div>

        {/* Level Filter */}
        <div className="relative">
          <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <select
            value={filterLevel}
            onChange={(e) => setFilterLevel(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-[#0a0e27] border border-cyan-900/30 rounded-lg text-gray-300 focus:outline-none focus:border-cyan-500/50 text-sm appearance-none"
          >
            {levels.map(level => (
              <option key={level} value={level}>
                {level === 'all' ? 'All Levels' : level.charAt(0).toUpperCase() + level.slice(1)}
              </option>
            ))}
          </select>
        </div>

        {/* Category Filter */}
        <div className="relative">
          <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <select
            value={filterCategory}
            onChange={(e) => setFilterCategory(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-[#0a0e27] border border-cyan-900/30 rounded-lg text-gray-300 focus:outline-none focus:border-cyan-500/50 text-sm appearance-none"
          >
            {categories.map(category => (
              <option key={category} value={category}>
                {category === 'all' ? 'All Categories' : category}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Total Events</div>
          <div className="text-cyan-400">{filteredLogs.length}</div>
        </div>
        <div className="bg-[#0a0e27] border border-red-900/30 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Critical</div>
          <div className="text-red-400">{filteredLogs.filter(l => l.level === 'critical').length}</div>
        </div>
        <div className="bg-[#0a0e27] border border-amber-900/30 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Warnings</div>
          <div className="text-amber-400">{filteredLogs.filter(l => l.level === 'warning').length}</div>
        </div>
        <div className="bg-[#0a0e27] border border-green-900/30 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Success</div>
          <div className="text-green-400">{filteredLogs.filter(l => l.level === 'success').length}</div>
        </div>
      </div>

      {/* Log Entries */}
      <div className="space-y-2 max-h-[600px] overflow-y-auto">
        {filteredLogs.map(log => (
          <div
            key={log.id}
            onClick={() => setSelectedLog(selectedLog?.id === log.id ? null : log)}
            className={`bg-[#0a0e27] border rounded-lg p-4 cursor-pointer transition-all ${
              selectedLog?.id === log.id
                ? 'border-cyan-500/50 bg-cyan-900/10'
                : 'border-cyan-900/30 hover:border-cyan-500/30'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-3 flex-1">
                <div className="mt-1">
                  {getLevelIcon(log.level)}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-xs px-2 py-1 rounded border ${getLevelColor(log.level)}`}>
                      {log.level.toUpperCase()}
                    </span>
                    <span className="text-xs px-2 py-1 rounded bg-cyan-900/30 text-cyan-300 border border-cyan-500/30">
                      {log.category}
                    </span>
                    <span className="text-xs text-gray-500">{formatTimestamp(log.timestamp)}</span>
                  </div>
                  <div className="text-sm text-gray-300 mb-2">{log.message}</div>
                  {selectedLog?.id === log.id && log.details && (
                    <div className="text-xs text-gray-400 bg-[#0d1238] border border-cyan-900/20 rounded p-3 mt-2">
                      {log.details}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
