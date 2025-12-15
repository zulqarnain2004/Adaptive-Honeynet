import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { AlertTriangle, CheckCircle, Info } from 'lucide-react';

interface Detection {
  id: string;
  timestamp: string;
  model: 'Random Forest' | 'Logistic Regression';
  prediction: 'Malicious' | 'Benign';
  confidence: number;
  features: string[];
}

export default function AttackDetectionPanel() {
  const [detections, setDetections] = useState<Detection[]>([
    {
      id: '1',
      timestamp: new Date().toLocaleTimeString(),
      model: 'Random Forest',
      prediction: 'Malicious',
      confidence: 0.94,
      features: ['High packet rate', 'Port scanning', 'SSH brute force']
    }
  ]);

  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(detections[0]);
  const [explainabilityMethod, setExplainabilityMethod] = useState<'SHAP' | 'LIME'>('SHAP');

  // Mock SHAP values for feature importance
  const shapValues = [
    { feature: 'Packet Rate', value: 0.32, impact: 'High' },
    { feature: 'Port Diversity', value: 0.28, impact: 'High' },
    { feature: 'SSH Attempts', value: 0.19, impact: 'Medium' },
    { feature: 'Payload Size', value: 0.12, impact: 'Medium' },
    { feature: 'Time Pattern', value: -0.08, impact: 'Low' },
    { feature: 'Protocol Type', value: 0.05, impact: 'Low' }
  ];

  // Mock LIME values for local interpretation
  const limeValues = [
    { feature: 'Packet Rate', value: 0.35, impact: 'High' },
    { feature: 'SSH Attempts', value: 0.31, impact: 'High' },
    { feature: 'Port Diversity', value: 0.22, impact: 'Medium' },
    { feature: 'Payload Size', value: 0.15, impact: 'Medium' },
    { feature: 'Protocol Type', value: -0.06, impact: 'Low' },
    { feature: 'Time Pattern', value: -0.04, impact: 'Low' }
  ];

  const currentExplainability = explainabilityMethod === 'SHAP' ? shapValues : limeValues;

  // Simulate new detections
  useEffect(() => {
    const interval = setInterval(() => {
      const models: Array<'Random Forest' | 'Logistic Regression'> = ['Random Forest', 'Logistic Regression'];
      const predictions: Array<'Malicious' | 'Benign'> = ['Malicious', 'Benign'];
      
      const newDetection: Detection = {
        id: Date.now().toString(),
        timestamp: new Date().toLocaleTimeString(),
        model: models[Math.floor(Math.random() * models.length)],
        prediction: predictions[Math.floor(Math.random() * predictions.length)],
        confidence: 0.75 + Math.random() * 0.24,
        features: ['High packet rate', 'Unusual timing', 'Multiple ports']
      };

      setDetections(prev => [newDetection, ...prev].slice(0, 5));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Model performance metrics
  const modelMetrics = {
    accuracy: 0.96,
    precision: 0.94,
    recall: 0.97,
    f1Score: 0.95
  };

  const getBarColor = (value: number) => {
    if (value > 0.2) return '#06b6d4';
    if (value > 0) return '#10b981';
    return '#f59e0b';
  };

  return (
    <div className="space-y-4">
      {/* Model Selection & Metrics */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-2">Active Models</div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-400" />
              <span className="text-xs text-gray-300">Random Forest</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-400" />
              <span className="text-xs text-gray-300">Logistic Regression</span>
            </div>
          </div>
        </div>

        <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-2">Ensemble Accuracy</div>
          <div className="text-cyan-400">{(modelMetrics.accuracy * 100).toFixed(1)}%</div>
          <div className="text-xs text-gray-500 mt-1">
            F1: {modelMetrics.f1Score.toFixed(3)}
          </div>
        </div>
      </div>

      {/* Recent Detections */}
      <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3 max-h-[180px] overflow-y-auto">
        <div className="text-xs text-gray-400 mb-2">Recent Classifications</div>
        <div className="space-y-2">
          {detections.map(detection => (
            <div
              key={detection.id}
              onClick={() => setSelectedDetection(detection)}
              className={`p-2 rounded cursor-pointer transition-colors ${
                selectedDetection?.id === detection.id
                  ? 'bg-cyan-900/30 border border-cyan-500/50'
                  : 'bg-[#0d1238] border border-transparent hover:bg-[#0d1238]/70'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  {detection.prediction === 'Malicious' ? (
                    <AlertTriangle className="w-3 h-3 text-red-400" />
                  ) : (
                    <CheckCircle className="w-3 h-3 text-green-400" />
                  )}
                  <span className="text-xs text-gray-300">{detection.model}</span>
                </div>
                <span className="text-xs text-gray-500">{detection.timestamp}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className={`text-xs ${
                  detection.prediction === 'Malicious' ? 'text-red-400' : 'text-green-400'
                }`}>
                  {detection.prediction}
                </span>
                <span className="text-xs text-cyan-400">
                  {(detection.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Explainability Section */}
      {selectedDetection && (
        <div className="bg-[#0a0e27] border border-cyan-900/30 rounded-lg p-3">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Info className="w-4 h-4 text-cyan-400" />
              <span className="text-xs text-cyan-400">Model Explainability</span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setExplainabilityMethod('SHAP')}
                className={`text-xs px-2 py-1 rounded transition-colors ${
                  explainabilityMethod === 'SHAP'
                    ? 'bg-cyan-900/50 text-cyan-300 border border-cyan-500/50'
                    : 'bg-[#0d1238] text-gray-400 border border-transparent hover:bg-[#0d1238]/70'
                }`}
              >
                SHAP
              </button>
              <button
                onClick={() => setExplainabilityMethod('LIME')}
                className={`text-xs px-2 py-1 rounded transition-colors ${
                  explainabilityMethod === 'LIME'
                    ? 'bg-cyan-900/50 text-cyan-300 border border-cyan-500/50'
                    : 'bg-[#0d1238] text-gray-400 border border-transparent hover:bg-[#0d1238]/70'
                }`}
              >
                LIME
              </button>
            </div>
          </div>

          {/* Feature Importance Chart */}
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={currentExplainability} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
              <XAxis type="number" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 10 }} />
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
                  border: '1px solid #0e7490',
                  borderRadius: '8px',
                  fontSize: '12px'
                }}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {currentExplainability.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getBarColor(entry.value)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div className="mt-2 text-xs text-gray-400">
            <span className="text-cyan-400">{explainabilityMethod}</span> values explain {' '}
            {explainabilityMethod === 'SHAP' ? 'global feature importance' : 'local prediction rationale'}
          </div>
        </div>
      )}
    </div>
  );
}
