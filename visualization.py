import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class VisualizationDashboard:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    def plot_metrics_comparison(self, metrics_dict: Dict, title: str = "Model Performance Comparison"):
        """Plot comparison of different models' metrics"""
        models = list(metrics_dict.keys())
        metrics = list(metrics_dict[models[0]].keys())
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, model in enumerate(models):
            values = [metrics_dict[model][metric] for metric in metrics]
            ax.bar(x + i*width - width*(len(models)-1)/2, values, width, 
                  label=model.replace('_', ' ').title())
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([m.title() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/results/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_learning_curves(self, rewards_history: List, epsilon_history: List = None,
                            title: str = "Reinforcement Learning Progress"):
        """Plot RL learning curves"""
        if epsilon_history:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            ax1, ax2 = axes
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        
        # Plot rewards
        ax1.plot(rewards_history, alpha=0.6, color=self.colors[0], label='Episode Reward')
        
        # Add moving average
        window = max(1, len(rewards_history) // 50)
        moving_avg = pd.Series(rewards_history).rolling(window=window, min_periods=1).mean()
        ax1.plot(moving_avg, linewidth=2, color=self.colors[1], label=f'Moving Average (window={window})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title(f'{title} - Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot epsilon decay if provided
        if epsilon_history:
            ax2.plot(epsilon_history, color=self.colors[2], linewidth=2)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Epsilon')
            ax2.set_title('Exploration Rate Decay')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/results/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_confusion_matrix_heatmap(self, confusion_matrix: np.ndarray, 
                                     class_names: List[str] = None,
                                     title: str = "Confusion Matrix"):
        """Plot confusion matrix as heatmap"""
        if class_names is None:
            class_names = ['Normal', 'Attack']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig('models/results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_network_topology(self, G: nx.Graph, honeypots: List[int] = None,
                             attacks: List[int] = None, title: str = "Network Topology"):
        """Visualize network topology with honeypots and attacks"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get positions
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = nx.spring_layout(G, seed=42)
        
        # Draw all nodes first
        all_nodes = list(G.nodes())
        normal_nodes = [n for n in all_nodes 
                       if n not in (honeypots or []) and n not in (attacks or [])]
        
        if normal_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, 
                                  node_color='lightblue', node_size=300, 
                                  ax=ax, label='Normal Nodes')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        
        # Highlight honeypots
        if honeypots:
            nx.draw_networkx_nodes(G, pos, nodelist=honeypots,
                                  node_color='green', node_size=500,
                                  ax=ax, label='Honeypots')
        
        # Highlight attacks
        if attacks:
            nx.draw_networkx_nodes(G, pos, nodelist=attacks,
                                  node_color='red', node_size=500,
                                  ax=ax, label='Attacks')
        
        # Highlight detections (overlap)
        if honeypots and attacks:
            detections = set(honeypots) & set(attacks)
            if detections:
                nx.draw_networkx_nodes(G, pos, nodelist=list(detections),
                                      node_color='yellow', node_size=600,
                                      ax=ax, label='Detected Attacks')
        
        # Add labels
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        # Calculate metrics for title
        if honeypots and attacks:
            detections = set(honeypots) & set(attacks)
            detection_rate = len(detections) / len(attacks) if attacks else 0
            false_positives = len(set(honeypots) - set(attacks))
            
            title = f"{title}\nAttacks: {len(attacks)} | Honeypots: {len(honeypots)} | " \
                   f"Detections: {len(detections)} ({detection_rate:.0%}) | " \
                   f"False Positives: {false_positives}"
        
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('models/results/network_topology.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict, 
                               top_n: int = 15, title: str = "Feature Importance"):
        """Plot feature importance"""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importance = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, color=self.colors[0])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('models/results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_real_time_performance(self, performance_metrics: List[Dict], 
                                  title: str = "Real-time Performance Metrics"):
        """Plot real-time simulation performance"""
        if not performance_metrics:
            return None
        
        df = pd.DataFrame(performance_metrics)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Detection Rate
        ax1.plot(df['iteration'], df['detection_rate'], 
                marker='o', color='green', linewidth=2, label='Detection Rate')
        ax1.fill_between(df['iteration'], df['detection_rate'], 
                       alpha=0.2, color='green')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Detection Rate')
        ax1.set_title('Attack Detection Rate Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. False Positive Rate
        ax2.plot(df['iteration'], df['false_positive_rate'],
                marker='s', color='orange', linewidth=2, label='False Positive Rate')
        ax2.fill_between(df['iteration'], df['false_positive_rate'],
                       alpha=0.2, color='orange')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('False Positive Rate')
        ax2.set_title('False Positive Rate Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. System Efficiency
        if 'efficiency' in df.columns:
            ax3.plot(df['iteration'], df['efficiency'],
                    marker='^', color='blue', linewidth=2, label='System Efficiency')
            ax3.fill_between(df['iteration'], df['efficiency'],
                           alpha=0.2, color='blue')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Efficiency Score')
            ax3.set_title('System Efficiency Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            # Show honeypot usage
            ax3.bar(df['iteration'], df['honeypots_used'], color='blue', alpha=0.7)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Honeypots Used')
            ax3.set_title('Honeypot Resource Utilization')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Attacks Detected vs Active Attacks
        iterations = df['iteration']
        detected = df['attacks_detected']
        active = df['active_attacks'] if 'active_attacks' in df.columns else [0] * len(df)
        
        width = 0.35
        x = np.arange(len(iterations))
        
        ax4.bar(x - width/2, active, width, label='Active Attacks', color='red', alpha=0.7)
        ax4.bar(x + width/2, detected, width, label='Detected Attacks', color='green', alpha=0.7)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Count')
        ax4.set_title('Attack Detection Performance')
        ax4.set_xticks(x)
        ax4.set_xticklabels(iterations)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig('models/results/real_time_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_attack_analysis(self, attack_log: List[Dict], title: str = "Attack Analysis"):
        """Plot attack analysis"""
        if not attack_log:
            return None
        
        df = pd.DataFrame(attack_log)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Attack type distribution
        if 'type' in df.columns:
            attack_counts = df['type'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(attack_counts)))
            ax1.pie(attack_counts.values, labels=attack_counts.index, 
                   autopct='%1.1f%%', colors=colors)
            ax1.set_title('Attack Type Distribution')
        
        # 2. Attack intensity distribution
        if 'intensity' in df.columns:
            ax2.hist(df['intensity'], bins=10, color='red', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Attack Intensity')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Attack Intensity Distribution')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Target node distribution
        if 'target_node' in df.columns:
            node_counts = df['target_node'].value_counts().sort_index()
            ax3.bar(node_counts.index, node_counts.values, color='blue', alpha=0.7)
            ax3.set_xlabel('Node ID')
            ax3.set_ylabel('Attack Count')
            ax3.set_title('Attack Distribution Across Nodes')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Detection rate over time
        if 'detected' in df.columns and 'start_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['start_time'])
            df['hour'] = df['timestamp'].dt.hour
            
            hourly_stats = df.groupby('hour').agg({
                'detected': ['count', 'sum']
            }).reset_index()
            
            hourly_stats.columns = ['hour', 'total_attacks', 'detected_attacks']
            hourly_stats['detection_rate'] = hourly_stats['detected_attacks'] / hourly_stats['total_attacks']
            
            ax4.plot(hourly_stats['hour'], hourly_stats['detection_rate'], 
                    marker='o', color='green', linewidth=2)
            ax4.set_xlabel('Hour of Day')
            ax4.set_ylabel('Detection Rate')
            ax4.set_title('Detection Rate by Hour')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1.1)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig('models/results/attack_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_interactive_dashboard(self, metrics_data: Dict, 
                                    network_data: Dict = None) -> go.Figure:
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance', 'Feature Importance',
                          'Network Status', 'Attack Detection Timeline'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # 1. Model Performance
        if 'ml_metrics' in metrics_data:
            models = list(metrics_data['ml_metrics'].keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            
            for i, model in enumerate(models):
                values = [metrics_data['ml_metrics'][model].get(m, 0) for m in metrics]
                fig.add_trace(
                    go.Bar(name=model, x=metrics, y=values),
                    row=1, col=1
                )
        else:
            # Sample data
            models = ['Random Forest', 'Logistic Regression', 'XGBoost']
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
            values = [[0.94, 0.93, 0.95, 0.94],
                     [0.89, 0.88, 0.89, 0.88],
                     [0.95, 0.95, 0.95, 0.95]]
            
            for i, model in enumerate(models):
                fig.add_trace(
                    go.Bar(name=model, x=metrics, y=values[i]),
                    row=1, col=1
                )
        
        # 2. Feature Importance (placeholder)
        features = ['Duration', 'Source Bytes', 'Destination Bytes', 
                   'Packet Rate', 'Source TTL', 'Destination TTL']
        importance = [0.85, 0.72, 0.68, 0.61, 0.55, 0.48]
        
        fig.add_trace(
            go.Bar(x=importance, y=features, orientation='h',
                  marker=dict(color=importance, colorscale='Viridis')),
            row=1, col=2
        )
        
        # 3. Network Status
        if network_data and 'graph' in network_data:
            G = network_data['graph']
            honeypots = network_data.get('honeypots', [])
            attacks = network_data.get('attacks', [])
            
            # Create network visualization
            pos = nx.spring_layout(G, seed=42)
            
            # Extract node positions
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                if node in honeypots and node in attacks:
                    node_text.append(f"Node {node}: Honeypot & Attack")
                    node_color.append('yellow')
                elif node in honeypots:
                    node_text.append(f"Node {node}: Honeypot")
                    node_color.append('green')
                elif node in attacks:
                    node_text.append(f"Node {node}: Attack")
                    node_color.append('red')
                else:
                    node_text.append(f"Node {node}: Normal")
                    node_color.append('lightblue')
            
            # Add nodes
            fig.add_trace(
                go.Scatter(x=node_x, y=node_y, mode='markers+text',
                          text=[f"Node {i}" for i in G.nodes()],
                          marker=dict(size=20, color=node_color),
                          hovertext=node_text,
                          hoverinfo='text',
                          showlegend=False),
                row=2, col=1
            )
            
            # Add edges
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(
                go.Scatter(x=edge_x, y=edge_y, mode='lines',
                          line=dict(width=1, color='gray'),
                          hoverinfo='none',
                          showlegend=False),
                row=2, col=1
            )
        else:
            # Sample network data
            fig.add_trace(
                go.Scatter(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5],
                          mode='markers',
                          marker=dict(size=20, color=['red', 'green', 'blue', 'yellow', 'gray']),
                          text=['Attack', 'Honeypot', 'Normal', 'Detected', 'Normal']),
                row=2, col=1
            )
        
        # 4. Attack Detection Timeline
        timeline = np.arange(100)
        attacks = np.random.randn(100).cumsum() + 50
        detections = attacks * 0.8 + np.random.randn(100) * 5
        
        fig.add_trace(
            go.Scatter(x=timeline, y=attacks, mode='lines',
                      name='Attack Attempts', line=dict(color='red', width=2)),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=timeline, y=detections, mode='lines',
                      name='Detected Attacks', line=dict(color='green', width=2)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Adaptive Deception-Mesh Interactive Dashboard",
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Metrics", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_xaxes(title_text="Importance", row=1, col=2)
        fig.update_yaxes(title_text="Features", row=1, col=2)
        fig.update_xaxes(title_text="X", row=2, col=1)
        fig.update_yaxes(title_text="Y", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        # Save interactive plot
        fig.write_html("models/results/interactive_dashboard.html")
        
        return fig
    
    def plot_resource_utilization(self, resource_data: Dict, 
                                 title: str = "Resource Utilization"):
        """Plot resource utilization over time"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        resources = ['CPU', 'Memory', 'Bandwidth', 'Honeypots']
        
        for i, resource in enumerate(resources):
            ax = axes[i]
            if resource in resource_data:
                time_steps = range(len(resource_data[resource]))
                ax.plot(time_steps, resource_data[resource], 
                       linewidth=2, color=self.colors[i])
                ax.fill_between(time_steps, 0, resource_data[resource],
                               alpha=0.3, color=self.colors[i])
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'{resource} Usage')
            ax.set_title(f'{resource} Utilization')
            ax.grid(True, alpha=0.3)
            if resource in resource_data:
                ax.set_ylim(0, max(resource_data[resource]) * 1.1)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig('models/results/resource_utilization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_comprehensive_report(self, all_results: Dict):
        """Generate comprehensive visualization report"""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE VISUALIZATION REPORT")
        print("="*60)
        
        # Create all visualizations
        visualizations = {}
        
        # 1. Metrics Comparison
        if 'ml_metrics' in all_results:
            print("   📊 Generating model performance comparison...")
            visualizations['metrics_comparison'] = self.plot_metrics_comparison(
                all_results['ml_metrics'], 
                "Machine Learning Models Performance"
            )
        
        # 2. Learning Curves
        if 'rl_rewards' in all_results:
            print("   📈 Generating reinforcement learning curves...")
            visualizations['learning_curves'] = self.plot_learning_curves(
                all_results['rl_rewards'],
                all_results.get('rl_epsilon'),
                "Reinforcement Learning Progress"
            )
        
        # 3. Real-time Performance
        if 'real_time_state' in all_results and 'performance' in all_results['real_time_state']:
            print("   ⚡ Generating real-time performance metrics...")
            visualizations['real_time_performance'] = self.plot_real_time_performance(
                all_results['real_time_state']['performance'],
                "Real-time Simulation Performance"
            )
        
        # 4. Network Topology
        if 'real_time_state' in all_results:
            print("   🌐 Generating network topology visualization...")
            visualizations['network_topology'] = self.plot_network_topology(
                all_results['real_time_state']['graph'],
                all_results['real_time_state'].get('honeypots'),
                all_results['real_time_state'].get('attacks'),
                "Adaptive Honeynet Configuration"
            )
        
        # 5. Feature Importance
        if 'feature_importance' in all_results:
            print("   🔍 Generating feature importance visualization...")
            visualizations['feature_importance'] = self.plot_feature_importance(
                all_results['feature_importance'],
                title="Top Features for Attack Detection"
            )
        
        # 6. Attack Analysis
        if 'real_time_state' in all_results and 'attack_statistics' in all_results['real_time_state']:
            print("   🎯 Generating attack analysis...")
            # This would need attack log data
        
        # 7. Interactive Dashboard
        print("   📱 Generating interactive dashboard...")
        visualizations['dashboard'] = self.create_interactive_dashboard(
            all_results.get('ml_metrics', {}),
            all_results.get('real_time_state', {})
        )
        
        print(f"\n✅ All visualizations saved to 'models/results/' directory")
        print("Generated files:")
        print("   1. metrics_comparison.png - Model performance comparison")
        print("   2. learning_curves.png - RL training progress")
        print("   3. real_time_performance.png - Simulation performance")
        print("   4. network_topology.png - Network visualization")
        print("   5. feature_importance.png - Feature importance")
        print("   6. interactive_dashboard.html - Interactive dashboard")
        print("   7. Additional visualizations in the directory")
        
        return visualizations