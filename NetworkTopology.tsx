import { useState, useEffect } from 'react';
import { Network, Target, Shield, Server, Wifi } from 'lucide-react';

interface Node {
  id: string;
  x: number;
  y: number;
  type: 'honeypot' | 'server' | 'router' | 'attacker';
  status: 'active' | 'compromised' | 'scanning';
  label: string;
}

interface NetworkTopologyProps {
  nodeCount: number;
  honeypotCount: number;
  isRunning: boolean;
}

export default function NetworkTopology({ nodeCount, honeypotCount, isRunning }: NetworkTopologyProps) {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [astarPaths, setAstarPaths] = useState<any[]>([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  // Generate network topology based on nodeCount and honeypotCount
  useEffect(() => {
    const width = 600;
    const height = 400;
    const newNodes: Node[] = [];

    // Add attacker
    newNodes.push({
      id: 'attacker-1',
      x: width / 2,
      y: 30,
      type: 'attacker',
      status: 'active',
      label: 'Threat Actor'
    });

    // Add routers
    const routerCount = Math.max(2, Math.floor(nodeCount * 0.15));
    for (let i = 0; i < routerCount; i++) {
      newNodes.push({
        id: `router-${i}`,
        x: (width / (routerCount + 1)) * (i + 1),
        y: 100,
        type: 'router',
        status: 'active',
        label: `R${i + 1}`
      });
    }

    // Add servers
    const serverCount = Math.max(3, Math.floor(nodeCount * 0.25));
    for (let i = 0; i < serverCount; i++) {
      newNodes.push({
        id: `server-${i}`,
        x: (width / (serverCount + 1)) * (i + 1),
        y: 200,
        type: 'server',
        status: 'active',
        label: `S${i + 1}`
      });
    }

    // Add honeypots
    for (let i = 0; i < honeypotCount; i++) {
      newNodes.push({
        id: `honeypot-${i}`,
        x: (width / (honeypotCount + 1)) * (i + 1),
        y: 320,
        type: 'honeypot',
        status: 'active',
        label: `H${i + 1}`
      });
    }

    setNodes(newNodes);

    // Generate A* paths for honeypots
    const paths = [];
    for (let i = 0; i < Math.min(3, honeypotCount); i++) {
      paths.push({
        from: 'attacker-1',
        to: `honeypot-${i}`,
        cost: 2 + Math.random() * 3,
        heuristic: 1 + Math.random() * 2
      });
    }
    setAstarPaths(paths);

  }, [nodeCount, honeypotCount]);

  // Simulate node status changes during simulation
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setNodes(prev => prev.map(node => {
        if (node.type === 'honeypot' && Math.random() > 0.95) {
          return { ...node, status: node.status === 'compromised' ? 'active' : 'compromised' };
        }
        if (node.type === 'router' && Math.random() > 0.98) {
          return { ...node, status: 'scanning' };
        }
        if (node.status === 'scanning' && Math.random() > 0.7) {
          return { ...node, status: 'active' };
        }
        return node;
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, [isRunning]);

  const getNodeColor = (node: Node) => {
    if (node.type === 'attacker') return '#ef4444';
    if (node.status === 'compromised') return '#f59e0b';
    if (node.status === 'scanning') return '#8b5cf6';
    if (node.type === 'honeypot') return '#06b6d4';
    if (node.type === 'router') return '#10b981';
    return '#3b82f6';
  };

  const getNodeIcon = (node: Node) => {
    if (node.type === 'attacker') return Target;
    if (node.type === 'honeypot') return Shield;
    if (node.type === 'router') return Wifi;
    return Server;
  };

  // Generate connections
  const connections: Array<{ from: string; to: string }> = [];
  nodes.forEach(node => {
    if (node.type === 'router') {
      connections.push({ from: 'attacker-1', to: node.id });
      nodes.filter(n => n.type === 'server').forEach(server => {
        connections.push({ from: node.id, to: server.id });
      });
    }
    if (node.type === 'server') {
      nodes.filter(n => n.type === 'honeypot').forEach(honeypot => {
        if (Math.random() > 0.5) {
          connections.push({ from: node.id, to: honeypot.id });
        }
      });
    }
  });

  const getNodeById = (id: string) => nodes.find(n => n.id === id);

  return (
    <div className="group relative bg-gradient-to-br from-[#0d1238]/90 to-[#0a0e27]/90 backdrop-blur-xl border-2 border-cyan-500/30 rounded-2xl p-6 shadow-2xl shadow-cyan-500/20 hover:shadow-cyan-500/40 transition-all duration-500 overflow-hidden">
      {/* Animated corner accents */}
      <div className="absolute top-0 left-0 w-20 h-20 border-t-2 border-l-2 border-cyan-400/50 rounded-tl-2xl" />
      <div className="absolute bottom-0 right-0 w-20 h-20 border-b-2 border-r-2 border-cyan-400/50 rounded-br-2xl" />
      
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-lg border border-cyan-400/50">
            <Network className="w-6 h-6 text-cyan-400 drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
          </div>
          <div>
            <h3 className="text-cyan-400 font-semibold">Network Topology</h3>
            <p className="text-xs text-gray-400">Real-time mesh visualization</p>
          </div>
        </div>
        <div className="px-3 py-1 bg-cyan-900/30 rounded-lg border border-cyan-500/50">
          <span className="text-xs text-cyan-300">{nodes.length} Nodes</span>
        </div>
      </div>

      {/* Network Canvas */}
      <div className="relative bg-gradient-to-br from-[#020817] to-[#0a0e27] border-2 border-cyan-900/30 rounded-xl overflow-hidden" style={{ height: '420px' }}>
        {/* Grid background */}
        <svg className="absolute inset-0 opacity-10" width="100%" height="100%">
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="1" className="text-cyan-500" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>

        <svg width="100%" height="420" className="relative z-10">
          {/* Draw connections */}
          {connections.map((conn, i) => {
            const fromNode = getNodeById(conn.from);
            const toNode = getNodeById(conn.to);
            if (!fromNode || !toNode) return null;
            
            return (
              <line
                key={`conn-${i}`}
                x1={fromNode.x}
                y1={fromNode.y}
                x2={toNode.x}
                y2={toNode.y}
                stroke="rgba(6, 182, 212, 0.2)"
                strokeWidth="2"
                strokeDasharray="4,4"
                className="transition-all duration-500"
              />
            );
          })}

          {/* Draw A* optimal paths with animation */}
          {isRunning && astarPaths.map((path, i) => {
            const fromNode = getNodeById(path.from);
            const toNode = getNodeById(path.to);
            if (!fromNode || !toNode) return null;

            return (
              <g key={`astar-${i}`}>
                <line
                  x1={fromNode.x}
                  y1={fromNode.y}
                  x2={toNode.x}
                  y2={toNode.y}
                  stroke="url(#pathGradient)"
                  strokeWidth="4"
                  strokeDasharray="8,4"
                  className="animate-pulse"
                  style={{ filter: 'drop-shadow(0 0 6px rgba(6, 182, 212, 0.8))' }}
                />
                <text
                  x={(fromNode.x + toNode.x) / 2}
                  y={(fromNode.y + toNode.y) / 2 - 15}
                  fill="#06b6d4"
                  fontSize="11"
                  fontWeight="bold"
                  className="pointer-events-none"
                  style={{ filter: 'drop-shadow(0 0 4px rgba(6, 182, 212, 1))' }}
                >
                  A* f={path.cost.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Gradient definitions */}
          <defs>
            <linearGradient id="pathGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" style={{ stopColor: '#06b6d4', stopOpacity: 1 }} />
              <stop offset="50%" style={{ stopColor: '#3b82f6', stopOpacity: 1 }} />
              <stop offset="100%" style={{ stopColor: '#06b6d4', stopOpacity: 1 }} />
            </linearGradient>
          </defs>

          {/* Draw nodes */}
          {nodes.map(node => {
            const Icon = getNodeIcon(node);
            const color = getNodeColor(node);
            
            return (
              <g 
                key={node.id} 
                className="cursor-pointer transition-transform hover:scale-110" 
                onClick={() => setSelectedNode(node)}
              >
                {/* Glow effect for active nodes */}
                {isRunning && (
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={node.type === 'attacker' ? 22 : 18}
                    fill={color}
                    opacity="0.3"
                    className="animate-ping"
                  />
                )}
                
                {/* Node circle */}
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={node.type === 'attacker' ? 18 : 14}
                  fill={color}
                  stroke={selectedNode?.id === node.id ? '#fff' : color}
                  strokeWidth={selectedNode?.id === node.id ? '3' : '2'}
                  style={{ 
                    filter: `drop-shadow(0 0 8px ${color})`,
                    opacity: 0.9
                  }}
                />
                
                {/* Status indicator ring for compromised nodes */}
                {node.status === 'compromised' && (
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={20}
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth="3"
                    className="animate-ping"
                  />
                )}

                {/* Node label */}
                <text
                  x={node.x}
                  y={node.y + 30}
                  textAnchor="middle"
                  fill="#9ca3af"
                  fontSize="11"
                  fontWeight="600"
                  className="pointer-events-none"
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Legend */}
        <div className="absolute top-4 right-4 bg-[#0d1238]/90 backdrop-blur-sm border border-cyan-500/30 rounded-lg p-3 space-y-2">
          <div className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
            <span className="text-gray-300">Honeypot</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]" />
            <span className="text-gray-300">Attacker</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.8)]" />
            <span className="text-gray-300">Compromised</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-8 h-1 bg-gradient-to-r from-cyan-400 to-blue-500 rounded shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
            <span className="text-gray-300">A* Path</span>
          </div>
        </div>
      </div>

      {/* Node Details */}
      {selectedNode && (
        <div className="mt-4 p-4 bg-gradient-to-r from-cyan-900/30 to-blue-900/30 backdrop-blur-sm border border-cyan-500/50 rounded-xl">
          <div className="flex items-center justify-between mb-2">
            <span className="text-cyan-400 font-semibold">{selectedNode.label}</span>
            <span className="px-3 py-1 rounded-lg bg-cyan-900/50 text-cyan-300 text-xs border border-cyan-500/50">
              {selectedNode.type.toUpperCase()}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <span className="text-gray-400">Status:</span>
              <span className="ml-2 text-gray-300 font-medium">{selectedNode.status}</span>
            </div>
            <div>
              <span className="text-gray-400">Position:</span>
              <span className="ml-2 text-gray-300 font-medium">
                ({selectedNode.x.toFixed(0)}, {selectedNode.y.toFixed(0)})
              </span>
            </div>
            <div>
              <span className="text-gray-400">ID:</span>
              <span className="ml-2 text-gray-300 font-medium">{selectedNode.id}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
