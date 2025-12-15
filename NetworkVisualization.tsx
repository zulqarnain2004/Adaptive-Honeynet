import { useState, useEffect } from 'react';
import { Target, Zap } from 'lucide-react';

interface Node {
  id: string;
  x: number;
  y: number;
  type: 'honeypot' | 'server' | 'router' | 'attacker';
  status: 'active' | 'inactive' | 'compromised';
  label: string;
}

interface AStarPath {
  from: string;
  to: string;
  cost: number;
  heuristic: number;
}

export default function NetworkVisualization() {
  const [nodes, setNodes] = useState<Node[]>([
    { id: 'r1', x: 300, y: 100, type: 'router', status: 'active', label: 'Router-1' },
    { id: 's1', x: 150, y: 200, type: 'server', status: 'active', label: 'Server-1' },
    { id: 's2', x: 450, y: 200, type: 'server', status: 'active', label: 'Server-2' },
    { id: 'h1', x: 100, y: 300, type: 'honeypot', status: 'active', label: 'Honey-1' },
    { id: 'h2', x: 250, y: 300, type: 'honeypot', status: 'active', label: 'Honey-2' },
    { id: 'h3', x: 400, y: 300, type: 'honeypot', status: 'active', label: 'Honey-3' },
    { id: 'h4', x: 550, y: 300, type: 'honeypot', status: 'active', label: 'Honey-4' },
    { id: 'a1', x: 300, y: 50, type: 'attacker', status: 'active', label: 'Threat' }
  ]);

  const [astarPath, setAstarPath] = useState<AStarPath[]>([
    { from: 'a1', to: 'r1', cost: 1.2, heuristic: 2.4 },
    { from: 'r1', to: 'h2', cost: 1.8, heuristic: 1.5 }
  ]);

  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  // Simulate dynamic updates
  useEffect(() => {
    const interval = setInterval(() => {
      setNodes(prevNodes => 
        prevNodes.map(node => {
          if (node.type === 'honeypot' && Math.random() > 0.9) {
            return { ...node, status: node.status === 'active' ? 'compromised' : 'active' };
          }
          return node;
        })
      );
    }, 4000);

    return () => clearInterval(interval);
  }, []);

  const connections = [
    { from: 'a1', to: 'r1' },
    { from: 'r1', to: 's1' },
    { from: 'r1', to: 's2' },
    { from: 's1', to: 'h1' },
    { from: 's1', to: 'h2' },
    { from: 's2', to: 'h3' },
    { from: 's2', to: 'h4' }
  ];

  const getNodeColor = (node: Node) => {
    if (node.type === 'attacker') return '#ef4444';
    if (node.status === 'compromised') return '#f59e0b';
    if (node.type === 'honeypot') return '#06b6d4';
    return '#10b981';
  };

  const getNodeFromId = (id: string) => nodes.find(n => n.id === id);

  return (
    <div className="space-y-4">
      {/* Network Canvas */}
      <div className="relative bg-[#0a0e27] border border-cyan-900/20 rounded-lg" style={{ height: '380px' }}>
        <svg width="100%" height="380" className="absolute inset-0">
          {/* Draw connections */}
          {connections.map((conn, i) => {
            const fromNode = getNodeFromId(conn.from);
            const toNode = getNodeFromId(conn.to);
            if (!fromNode || !toNode) return null;
            
            return (
              <line
                key={i}
                x1={fromNode.x}
                y1={fromNode.y}
                x2={toNode.x}
                y2={toNode.y}
                stroke="#1e3a5f"
                strokeWidth="2"
                strokeDasharray="4,4"
              />
            );
          })}

          {/* Draw A* optimal path */}
          {astarPath.map((path, i) => {
            const fromNode = getNodeFromId(path.from);
            const toNode = getNodeFromId(path.to);
            if (!fromNode || !toNode) return null;

            return (
              <g key={`astar-${i}`}>
                <line
                  x1={fromNode.x}
                  y1={fromNode.y}
                  x2={toNode.x}
                  y2={toNode.y}
                  stroke="#06b6d4"
                  strokeWidth="3"
                  className="animate-pulse"
                />
                <text
                  x={(fromNode.x + toNode.x) / 2}
                  y={(fromNode.y + toNode.y) / 2 - 10}
                  fill="#06b6d4"
                  fontSize="11"
                  className="pointer-events-none"
                >
                  f={'{'}g:{path.cost.toFixed(1)}, h:{path.heuristic.toFixed(1)}{'}'}
                </text>
              </g>
            );
          })}

          {/* Draw nodes */}
          {nodes.map(node => (
            <g key={node.id} className="cursor-pointer" onClick={() => setSelectedNode(node)}>
              <circle
                cx={node.x}
                cy={node.y}
                r={node.type === 'attacker' ? 16 : 12}
                fill={getNodeColor(node)}
                stroke={selectedNode?.id === node.id ? '#fff' : 'transparent'}
                strokeWidth="2"
                className="transition-all"
              />
              {node.status === 'compromised' && (
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={16}
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth="2"
                  className="animate-ping"
                />
              )}
              <text
                x={node.x}
                y={node.y + 25}
                textAnchor="middle"
                fill="#9ca3af"
                fontSize="10"
                className="pointer-events-none"
              >
                {node.label}
              </text>
            </g>
          ))}
        </svg>

        {/* Legend */}
        <div className="absolute top-3 right-3 bg-[#0d1238] border border-cyan-900/30 rounded px-3 py-2 space-y-1">
          <div className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full bg-cyan-400" />
            <span className="text-gray-400">Honeypot</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span className="text-gray-400">Attacker</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full bg-amber-500" />
            <span className="text-gray-400">Compromised</span>
          </div>
        </div>
      </div>

      {/* Node Details */}
      {selectedNode && (
        <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-cyan-400">{selectedNode.label}</span>
            <span className="text-xs px-2 py-1 rounded bg-cyan-900/30 text-cyan-300">
              {selectedNode.type}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-gray-500">Status:</span>
              <span className="ml-2 text-gray-300">{selectedNode.status}</span>
            </div>
            <div>
              <span className="text-gray-500">Position:</span>
              <span className="ml-2 text-gray-300">
                ({selectedNode.x}, {selectedNode.y})
              </span>
            </div>
          </div>
          {selectedNode.type === 'honeypot' && (
            <div className="text-xs text-gray-400 pt-2 border-t border-cyan-900/20">
              <Zap className="w-3 h-3 inline mr-1" />
              A* deployed via optimal path (cost: 3.6, steps: 2)
            </div>
          )}
        </div>
      )}

      {/* A* Algorithm Status */}
      <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
        <div className="flex items-center gap-2 mb-2">
          <Target className="w-4 h-4 text-cyan-400" />
          <span className="text-sm text-cyan-400">A* Pathfinding Active</span>
        </div>
        <div className="text-xs text-gray-400">
          Computing optimal deployment paths using Manhattan distance heuristic
        </div>
      </div>
    </div>
  );
}
