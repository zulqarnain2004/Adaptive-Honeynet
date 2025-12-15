import { useState } from 'react';
import { FileText, Search, Filter, Download, AlertTriangle, Info, CheckCircle, XCircle } from 'lucide-react';

interface Log {
  id: number;
  timestamp: string;
  level: string;
  category: string;
  message: string;
}

interface SimulationLogsTabProps {
  logs: Log[];
}

export default function SimulationLogsTab({ logs }: SimulationLogsTabProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterLevel, setFilterLevel] = useState('all');
  const [filterCategory, setFilterCategory] = useState('all');

  const filteredLogs = logs.filter(log => {
    const matchesSearch = searchTerm === '' || 
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.category.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesLevel = filterLevel === 'all' || log.level === filterLevel;
    const matchesCategory = filterCategory === 'all' || log.category === filterCategory;
    return matchesSearch && matchesLevel && matchesCategory;
  });

  const categories = ['all', ...Array.from(new Set(logs.map(l => l.category)))];
  const levels = ['all', 'info', 'success', 'warning', 'critical'];

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

  const getLevelStyle = (level: string) => {
    switch (level) {
      case 'critical':
        return 'from-red-900/30 to-rose-900/30 border-red-500/50 text-red-300';
      case 'warning':
        return 'from-amber-900/30 to-orange-900/30 border-amber-500/50 text-amber-300';
      case 'success':
        return 'from-green-900/30 to-emerald-900/30 border-green-500/50 text-green-300';
      default:
        return 'from-cyan-900/30 to-blue-900/30 border-cyan-500/50 text-cyan-300';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const stats = {
    total: filteredLogs.length,
    critical: filteredLogs.filter(l => l.level === 'critical').length,
    warning: filteredLogs.filter(l => l.level === 'warning').length,
    success: filteredLogs.filter(l => l.level === 'success').length,
    info: filteredLogs.filter(l => l.level === 'info').length
  };

  return (
    <div className="relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-cyan-500/30 rounded-2xl p-8 shadow-2xl shadow-cyan-500/20 overflow-hidden">
      {/* Animated corner accents */}
      <div className="absolute top-0 left-0 w-32 h-32 border-t-2 border-l-2 border-cyan-400/50 rounded-tl-2xl" />
      <div className="absolute bottom-0 right-0 w-32 h-32 border-b-2 border-r-2 border-cyan-400/50 rounded-br-2xl" />
      
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-xl border border-cyan-400/50">
            <FileText className="w-8 h-8 text-cyan-400 drop-shadow-[0_0_10px_rgba(6,182,212,0.8)]" />
          </div>
          <div>
            <h2 className="text-2xl bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent font-bold">
              Simulation Event Logs
            </h2>
            <p className="text-sm text-gray-400">Comprehensive chronological system activity</p>
          </div>
        </div>
        <button className="group flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-cyan-900/40 to-blue-900/40 text-cyan-300 rounded-xl border-2 border-cyan-500/50 hover:border-cyan-400 hover:shadow-[0_0_20px_rgba(6,182,212,0.4)] transition-all duration-300">
          <Download className="w-4 h-4 group-hover:animate-bounce" />
          <span className="text-sm font-semibold">Export Logs</span>
        </button>
      </div>

      {/* Filters */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search logs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-12 pr-4 py-3 bg-gradient-to-r from-[#0a0e27] to-[#0d1238] border-2 border-cyan-900/30 rounded-xl text-gray-300 placeholder-gray-500 focus:outline-none focus:border-cyan-500/50 focus:shadow-[0_0_20px_rgba(6,182,212,0.2)] transition-all"
          />
        </div>

        {/* Level Filter */}
        <div className="relative">
          <Filter className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <select
            value={filterLevel}
            onChange={(e) => setFilterLevel(e.target.value)}
            className="w-full pl-12 pr-4 py-3 bg-gradient-to-r from-[#0a0e27] to-[#0d1238] border-2 border-cyan-900/30 rounded-xl text-gray-300 focus:outline-none focus:border-cyan-500/50 transition-all appearance-none cursor-pointer"
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
          <Filter className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <select
            value={filterCategory}
            onChange={(e) => setFilterCategory(e.target.value)}
            className="w-full pl-12 pr-4 py-3 bg-gradient-to-r from-[#0a0e27] to-[#0d1238] border-2 border-cyan-900/30 rounded-xl text-gray-300 focus:outline-none focus:border-cyan-500/50 transition-all appearance-none cursor-pointer"
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
      <div className="grid grid-cols-5 gap-4 mb-6">
        {[
          { label: 'Total', value: stats.total, color: 'cyan', icon: FileText },
          { label: 'Critical', value: stats.critical, color: 'red', icon: AlertTriangle },
          { label: 'Warning', value: stats.warning, color: 'amber', icon: AlertTriangle },
          { label: 'Success', value: stats.success, color: 'green', icon: CheckCircle },
          { label: 'Info', value: stats.info, color: 'blue', icon: Info }
        ].map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div 
              key={index}
              className={`relative bg-gradient-to-br from-${stat.color}-900/30 to-${stat.color}-800/20 border-2 border-${stat.color}-500/40 rounded-xl p-4 hover:scale-105 transition-all duration-300 overflow-hidden group`}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
              <div className="relative z-10">
                <div className="flex items-center justify-between mb-2">
                  <Icon className={`w-5 h-5 text-${stat.color}-400`} />
                  <span className="text-xs text-gray-400">{stat.label}</span>
                </div>
                <div className={`text-3xl font-bold text-${stat.color}-400`}>{stat.value}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Log Entries */}
      <div className="space-y-3 max-h-[600px] overflow-y-auto custom-scrollbar pr-2">
        {filteredLogs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-gray-500">
            <XCircle className="w-16 h-16 mb-4 opacity-30" />
            <p className="text-lg">No logs found</p>
            <p className="text-sm text-gray-600 mt-2">Try adjusting your filters or start the simulation</p>
          </div>
        ) : (
          filteredLogs.map((log, index) => (
            <div
              key={log.id}
              className={`relative bg-gradient-to-r ${getLevelStyle(log.level)} backdrop-blur-sm border-2 rounded-xl p-5 hover:scale-[1.01] transition-all duration-300 cursor-pointer overflow-hidden group`}
              style={{
                animation: `slideInRight 0.5s ease-out ${index * 0.05}s both`
              }}
            >
              {/* Animated background */}
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
              
              <div className="relative z-10">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3 flex-1">
                    {getLevelIcon(log.level)}
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span className={`text-xs px-3 py-1 rounded-lg border-2 font-bold uppercase ${getLevelStyle(log.level)}`}>
                          {log.level}
                        </span>
                        <span className="text-xs px-3 py-1 rounded-lg bg-cyan-900/30 text-cyan-300 border-2 border-cyan-500/50 font-semibold">
                          {log.category}
                        </span>
                      </div>
                      <p className="text-sm text-gray-200 leading-relaxed">{log.message}</p>
                    </div>
                  </div>
                  <span className="text-xs text-gray-500 font-mono whitespace-nowrap ml-4">
                    {formatTimestamp(log.timestamp)}
                  </span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      <style>{`
        @keyframes slideInRight {
          from {
            opacity: 0;
            transform: translateX(30px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        .custom-scrollbar::-webkit-scrollbar {
          width: 10px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(6, 182, 212, 0.1);
          border-radius: 5px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: linear-gradient(180deg, #06b6d4, #3b82f6);
          border-radius: 5px;
          box-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(180deg, #0891b2, #2563eb);
        }
      `}</style>
    </div>
  );
}
