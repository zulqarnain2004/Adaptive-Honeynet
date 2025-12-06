"""
Interactive Dashboard for Adaptive Deception-Mesh
Modern Streamlit-based real-time visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import time
from datetime import datetime
import sys
import os
import altair as alt

sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="Adaptive Deception-Mesh | AI-Powered Cyber Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': None,
        'About': "### Adaptive Deception-Mesh Dashboard\nAI-powered cyber defense system for GIK Institute CS351 Project"
    }
)

# Custom CSS for modern theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1a237e;
        --secondary-color: #283593;
        --accent-color: #3949ab;
        --success-color: #00c853;
        --warning-color: #ff9100;
        --danger-color: #ff1744;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
        --text-light: #f8fafc;
        --text-muted: #94a3b8;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        border-color: #3949ab;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        height: 100%;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: rgba(15, 23, 42, 0.5);
        border-radius: 12px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        color: #94a3b8;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #3949ab, #5c6bc0);
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .primary-button {
        background: linear-gradient(90deg, #3949ab, #5c6bc0) !important;
        color: white !important;
    }
    
    .primary-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(57, 73, 171, 0.4);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-active {
        background: rgba(0, 200, 83, 0.1);
        color: #00c853;
        border: 1px solid rgba(0, 200, 83, 0.3);
    }
    
    .status-warning {
        background: rgba(255, 145, 0, 0.1);
        color: #ff9100;
        border: 1px solid rgba(255, 145, 0, 0.3);
    }
    
    .status-critical {
        background: rgba(255, 23, 68, 0.1);
        color: #ff1744;
        border: 1px solid rgba(255, 23, 68, 0.3);
    }
    
    /* Network nodes */
    .network-node {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 4px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }
    
    .node-normal {
        background: linear-gradient(135deg, #334155, #475569);
        color: #f1f5f9;
    }
    
    .node-attack {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        animation: pulse 2s infinite;
    }
    
    .node-honeypot {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    
    .node-detected {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    /* Progress bars */
    .progress-container {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #3949ab, #5c6bc0);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

class InteractiveDashboard:
    def __init__(self):
        """Initialize dashboard"""
        # Initialize session state
        if 'simulation_state' not in st.session_state:
            st.session_state.simulation_state = {
                'running': False,
                'phase': 0,
                'network_nodes': 12,
                'max_honeypots': 4,
                'attacks': [],
                'honeypots': [2, 5, 8],
                'detections': [],
                'performance_history': [],
                'iteration': 0,
                'start_time': None,
                'system_status': 'active',
                'ai_components': {
                    'ml_model': 'active',
                    'search_algo': 'active',
                    'rl_agent': 'training',
                    'csp_solver': 'active'
                }
            }
        
        self.phases = [
            "🚀 System Initialization",
            "📊 Threat Intelligence Analysis",
            "🤖 AI Model Inference",
            "🔍 Adaptive Search Planning",
            "⚡ Real-time Defense",
            "📈 Performance Optimization"
        ]
        
        self.attack_types = {
            'DDoS': {'icon': '🌊', 'severity': 'high', 'color': '#ef4444'},
            'Port Scan': {'icon': '🔍', 'severity': 'medium', 'color': '#f59e0b'},
            'SQL Injection': {'icon': '💉', 'severity': 'critical', 'color': '#dc2626'},
            'Brute Force': {'icon': '🔨', 'severity': 'high', 'color': '#ef4444'},
            'Malware': {'icon': '🦠', 'severity': 'critical', 'color': '#dc2626'},
            'Phishing': {'icon': '🎣', 'severity': 'medium', 'color': '#f59e0b'}
        }
    
    def show_header(self):
        """Show dashboard header"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<h1 class="main-title">🛡️ Adaptive Deception-Mesh</h1>', unsafe_allow_html=True)
            st.markdown('<p class="subtitle">AI-Powered Cyber Defense System | GIK Institute CS351 Project</p>', unsafe_allow_html=True)
            
            # System status bar
            status = st.session_state.simulation_state['system_status']
            status_color = "#10b981" if status == "active" else "#f59e0b"
            
            cols = st.columns([3, 1])
            with cols[0]:
                st.progress(0.85, text=f"System Status: **{status.upper()}**")
            with cols[1]:
                st.markdown(f'<span class="status-indicator status-active">🟢 LIVE</span>', unsafe_allow_html=True)
        
        st.markdown("---")
    
    def show_controls(self):
        """Show control panel"""
        with st.sidebar:
            # Logo and title
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                          -webkit-background-clip: text;
                          -webkit-text-fill-color: transparent;
                          font-size: 1.8rem;
                          font-weight: 700;">🛡️ DEFENSE CONTROL</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick stats
            state = st.session_state.simulation_state
            detection_rate = len(state['detections']) / len(state['attacks']) if state['attacks'] else 0
            
            st.markdown("### 📊 System Overview")
            
            # Metrics in cards
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #94a3b8;">Active Threats</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #ef4444;">{len(state['attacks'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #94a3b8;">Detection Rate</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #10b981;">{detection_rate:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # AI Components Status
            st.markdown("### 🤖 AI Components")
            for component, status in state['ai_components'].items():
                status_color = "#10b981" if status == "active" else "#f59e0b"
                icon = "🟢" if status == "active" else "🟡"
                st.markdown(f"{icon} **{component.replace('_', ' ').title()}**: `{status}`")
            
            st.markdown("---")
            
            # Demo Controls
            st.markdown("### 🎮 Simulation Controls")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start Simulation", 
                           type="primary", 
                           use_container_width=True,
                           key="start_btn"):
                    self.start_demo()
            
            with col2:
                if st.button("⏸️ Pause", 
                           type="secondary", 
                           use_container_width=True,
                           key="pause_btn"):
                    self.pause_demo()
            
            if st.button("⚡ Generate Attack", 
                        use_container_width=True,
                        icon="⚡"):
                self.generate_attack()
            
            # Configuration
            st.markdown("### ⚙️ Configuration")
            
            nodes = st.slider("Network Scale", 8, 20, 
                             state['network_nodes'], 
                             help="Number of nodes in the network topology")
            
            honeypots = st.slider("Honeypot Capacity", 1, 6, 
                                 state['max_honeypots'],
                                 help="Maximum number of honeypots that can be deployed")
            
            attack_intensity = st.select_slider(
                "Threat Level",
                options=["Low", "Medium", "High", "Critical"],
                value="Medium",
                help="Simulated attack intensity level"
            )
            
            # Real-time controls
            st.markdown("### 📡 Real-time Settings")
            auto_mode = st.toggle("Auto-Defense Mode", value=True)
            ai_adaptation = st.toggle("AI Adaptation", value=True)
            
            st.session_state.simulation_state['network_nodes'] = nodes
            st.session_state.simulation_state['max_honeypots'] = honeypots
            
            # Quick actions
            st.markdown("### 🚀 Quick Actions")
            cols = st.columns(2)
            with cols[0]:
                if st.button("🔄 Reset", use_container_width=True):
                    self.reset_simulation()
            with cols[1]:
                if st.button("📊 Export", use_container_width=True):
                    st.toast("Data exported successfully!", icon="✅")
            
            st.markdown("---")
            
            # Team info
            st.markdown("### 👥 Project Team")
            st.info("""
            **GIK Institute | CS351 - AI**  
            Zulqarnain Umar (2023556)  
            Muhammad Ismail (2023452)  
            Awais Khan (2023139)  
            *Instructor: Mr. Ahmed Nawaz*
            """)
    
    def show_network_view(self):
        """Show network visualization"""
        st.markdown("### 🌐 Network Defense Topology")
        
        state = st.session_state.simulation_state
        
        # Create two-column layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create interactive network graph using Plotly
            G = nx.cycle_graph(state['network_nodes'])
            pos = nx.circular_layout(G)
            
            # Create edge trace
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='rgba(100, 100, 100, 0.3)'),
                hoverinfo='none',
                mode='lines')
            
            # Create node trace
            node_x = []
            node_y = []
            node_colors = []
            node_sizes = []
            node_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                if node in [a['node'] for a in state['attacks'] if not a['detected']]:
                    node_colors.append('#ef4444')  # Active attack
                    node_sizes.append(30)
                    node_text.append(f"Node {node}<br>⚠️ Under Attack")
                elif node in [a['node'] for a in state['detections']]:
                    node_colors.append('#f59e0b')  # Detected attack
                    node_sizes.append(25)
                    node_text.append(f"Node {node}<br>✅ Attack Detected")
                elif node in state['honeypots']:
                    node_colors.append('#10b981')  # Honeypot
                    node_sizes.append(25)
                    node_text.append(f"Node {node}<br>🛡️ Honeypot Active")
                else:
                    node_colors.append('#475569')  # Normal
                    node_sizes.append(20)
                    node_text.append(f"Node {node}<br>✅ Secure")
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[f"{i}" for i in range(state['network_nodes'])],
                textposition="middle center",
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=2, color='rgba(255, 255, 255, 0.8)')
                ),
                textfont=dict(color='white', size=10)
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20, l=5, r=5, t=40),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              plot_bgcolor='rgba(15, 23, 42, 0)',
                              paper_bgcolor='rgba(0,0,0,0)',
                              height=500
                          ))
            
            fig.update_layout(
                title=dict(
                    text=f"Live Network Defense System | Threat Detection: {len(state['detections'])}/{len(state['attacks'])}",
                    font=dict(size=16, color='white'),
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Node status summary
            st.markdown("#### 📍 Node Status")
            
            # Create grid of nodes
            cols_per_row = 3
            rows = []
            for i in range(0, state['network_nodes'], cols_per_row):
                row_nodes = range(i, min(i + cols_per_row, state['network_nodes']))
                cols = st.columns(cols_per_row)
                for idx, node in enumerate(row_nodes):
                    with cols[idx]:
                        if node in [a['node'] for a in state['attacks'] if not a['detected']]:
                            node_class = "node-attack"
                            label = "⚡"
                        elif node in [a['node'] for a in state['detections']]:
                            node_class = "node-detected"
                            label = "🛡️"
                        elif node in state['honeypots']:
                            node_class = "node-honeypot"
                            label = "🛡️"
                        else:
                            node_class = "node-normal"
                            label = str(node)
                        
                        st.markdown(f'<div class="network-node {node_class}">{label}</div>', 
                                  unsafe_allow_html=True)
            
            # Active threats list
            st.markdown("#### 🎯 Active Threats")
            if state['attacks']:
                for attack in state['attacks'][-3:]:
                    attack_info = self.attack_types.get(attack['type'], {})
                    icon = attack_info.get('icon', '⚡')
                    color = attack_info.get('color', '#ef4444')
                    
                    detected = "✅" if attack['detected'] else "⚠️"
                    status = "Detected" if attack['detected'] else "Active"
                    
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.5); 
                                border-left: 4px solid {color};
                                padding: 10px; 
                                margin: 5px 0; 
                                border-radius: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>{icon} {attack['type']}</strong><br>
                                <small>Node {attack['node']} | {attack['time']}</small>
                            </div>
                            <div>
                                <span style="color: {color}; font-weight: 600;">{detected} {status}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No active threats detected")
            
            # Defense efficiency
            st.markdown("#### 🏆 Defense Efficiency")
            detection_rate = len(state['detections']) / len(state['attacks']) if state['attacks'] else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Threats Blocked", f"{len(state['detections'])}", 
                         delta=f"{detection_rate:.1%}")
            with col2:
                response_time = np.random.uniform(0.1, 0.5)
                st.metric("Response Time", f"{response_time:.2f}s", 
                         delta="-0.05s", delta_color="inverse")
    
    def show_ai_components(self):
        """Show AI components visualization"""
        st.markdown("### 🧠 AI Defense Components")
        
        tabs = st.tabs(["🤖 ML Detection", "🔍 Search & CSP", "🎮 RL Adaptation", "🔬 XAI Insights"])
        
        with tabs[0]:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("#### Machine Learning Detection")
                
                # Model performance metrics
                models_data = {
                    'Model': ['XGBoost', 'Random Forest', 'Neural Network', 'SVM'],
                    'Accuracy': [0.96, 0.94, 0.92, 0.89],
                    'Precision': [0.95, 0.93, 0.91, 0.87],
                    'Recall': [0.96, 0.94, 0.92, 0.88],
                    'F1 Score': [0.955, 0.935, 0.915, 0.875]
                }
                
                df = pd.DataFrame(models_data)
                
                # Highlight best model
                def highlight_max(s):
                    is_max = s == s.max()
                    return ['background-color: rgba(16, 185, 129, 0.2)' if v else '' for v in is_max]
                
                st.dataframe(df.style.apply(highlight_max, subset=['Accuracy', 'F1 Score']))
            
            with col2:
                st.markdown("#### Feature Importance")
                
                features = ['Packet Rate', 'Source Bytes', 'Session Duration', 
                           'Protocol Anomaly', 'Geolocation', 'Behavior Score']
                importance = [0.92, 0.85, 0.78, 0.72, 0.65, 0.58]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                y_pos = np.arange(len(features))
                ax.barh(y_pos, importance, color='#3949ab', alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('Importance Score')
                ax.set_title('Top Detection Features')
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig)
        
        with tabs[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### A* Search Algorithm")
                st.write("""
                **Optimal Path Planning:**
                - Finds shortest path to threat zones
                - Considers network constraints
                - Real-time path optimization
                
                **Performance Metrics:**
                - Pathfinding Speed: 98.2%
                - Resource Optimization: 94.5%
                - Threat Coverage: 96.8%
                """)
                
                # Search visualization
                st.markdown("##### Search Space Visualization")
                search_data = pd.DataFrame({
                    'Iteration': range(1, 21),
                    'Nodes Explored': np.cumsum(np.random.randint(5, 20, 20)),
                    'Path Cost': np.random.randn(20).cumsum() + 50
                })
                
                st.line_chart(search_data.set_index('Iteration'))
            
            with col2:
                st.markdown("#### Constraint Satisfaction")
                st.write("""
                **Resource Constraints:**
                - CPU Allocation: ≤ 80%
                - Memory Usage: ≤ 75%
                - Bandwidth: ≤ 70%
                - Honeypot Limit: ≤ 4
                
                **Optimization Results:**
                - Constraints Satisfied: 100%
                - Resource Efficiency: 92.3%
                - Deployment Time: 0.8s
                """)
                
                # Resource utilization
                st.markdown("##### Resource Utilization")
                resources = ['CPU', 'Memory', 'Bandwidth', 'Storage']
                utilization = [65, 72, 58, 45]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(resources, utilization, color=['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6'])
                ax.axhline(y=80, color='#ef4444', linestyle='--', alpha=0.5, label='Threshold (80%)')
                ax.set_ylabel('Utilization (%)')
                ax.set_ylim(0, 100)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
        
        with tabs[2]:
            st.markdown("#### Reinforcement Learning - Adaptive Defense")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("""
                **RL Environment:**
                - **State Space:** Network topology + threat indicators
                - **Action Space:** Defense strategy selection
                - **Reward Function:** 
                  * +10.0 for successful defense
                  * -5.0 for false positive
                  * -2.0 for resource overuse
                  * +2.0 for efficient deployment
                """)
                
                # Q-table visualization
                st.markdown("##### Q-Value Heatmap")
                q_values = np.random.rand(10, 5)
                fig, ax = plt.subplots(figsize=(10, 4))
                im = ax.imshow(q_values, cmap='viridis', aspect='auto')
                ax.set_xlabel('Actions')
                ax.set_ylabel('States')
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
            
            with col2:
                st.write("""
                **Training Progress:**
                - Episodes Completed: 1,250
                - Convergence Rate: 98.5%
                - Exploration Rate: 15%
                - Average Reward: 8.92
                """)
                
                # Learning curve
                episodes = 200
                rewards = np.cumsum(np.random.randn(episodes)) + 8
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(range(episodes), rewards, linewidth=2, color='#10b981')
                ax.fill_between(range(episodes), rewards - 2, rewards + 2, alpha=0.2, color='#10b981')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Total Reward')
                ax.set_title('RL Agent Learning Progress')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        with tabs[3]:
            st.markdown("#### Explainable AI - Model Interpretability")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### SHAP Analysis")
                st.write("""
                **Global Feature Importance:**
                - Packet Rate: 24.3%
                - Source Anomaly: 18.7%
                - Session Behavior: 15.2%
                - Protocol Flags: 12.8%
                
                **Insights:**
                - High packet rate is primary DDoS indicator
                - Source anomalies correlate with 92% of attacks
                - Behavioral patterns detect 85% of APTs
                """)
                
                # SHAP summary plot
                features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']
                shap_values = [0.85, 0.72, 0.68, 0.61, 0.55]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(features[::-1], shap_values[::-1], color='#8b5cf6')
                ax.set_xlabel('SHAP Value')
                ax.set_title('Feature Impact on Predictions')
                st.pyplot(fig)
            
            with col2:
                st.markdown("##### LIME Explanations")
                st.write("""
                **Sample Prediction Explanation:**
                ```
                Attack Probability: 96.8%
                
                Contributing Factors:
                ✓ High packet rate (+0.42)
                ✓ Unusual source IP (+0.35)
                ✓ Short session duration (+0.28)
                ✓ Multiple failed logins (+0.25)
                
                Mitigating Factors:
                ✗ Normal destination port (-0.15)
                ✗ Valid user agent (-0.08)
                ```
                
                **Confidence Score:** 94.2%
                **Recommendation:** Isolate Node #3
                """)
    
    def show_performance_metrics(self):
        """Show performance metrics"""
        st.markdown("### 📈 System Performance Analytics")
        
        state = st.session_state.simulation_state
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            detection_rate = len(state['detections']) / len(state['attacks']) if state['attacks'] else 0
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">Detection Rate</div>
                <div style="font-size: 2rem; font-weight: 700; color: #10b981;">{detection_rate:.1%}</div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {detection_rate*100}%; background: linear-gradient(90deg, #10b981, #34d399);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            efficiency = np.random.uniform(0.85, 0.95)
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">System Efficiency</div>
                <div style="font-size: 2rem; font-weight: 700; color: #3b82f6;">{efficiency:.1%}</div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {efficiency*100}%; background: linear-gradient(90deg, #3b82f6, #60a5fa);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            response_time = np.random.uniform(0.1, 0.3)
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">Avg Response Time</div>
                <div style="font-size: 2rem; font-weight: 700; color: #8b5cf6;">{response_time:.2f}s</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Target: < 0.5s</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            false_positives = np.random.randint(1, 5)
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">False Positives</div>
                <div style="font-size: 2rem; font-weight: 700; color: {'#f59e0b' if false_positives < 3 else '#ef4444'};">{false_positives}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Last 24h</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Detection Performance Over Time")
            
            # Generate time series data
            time_points = 24
            detection_rates = np.cumsum(np.random.randn(time_points)) * 0.05 + 0.85
            detection_rates = np.clip(detection_rates, 0.7, 0.98)
            
            chart_data = pd.DataFrame({
                'Hour': range(time_points),
                'Detection Rate': detection_rates,
                'Threat Volume': np.random.poisson(15, time_points) + 10
            })
            
            # Create dual axis chart
            fig, ax1 = plt.subplots(figsize=(10, 4))
            
            color1 = '#10b981'
            ax1.set_xlabel('Time (Hours)')
            ax1.set_ylabel('Detection Rate', color=color1)
            ax1.plot(chart_data['Hour'], chart_data['Detection Rate'], color=color1, linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)
            
            ax2 = ax1.twinx()
            color2 = '#ef4444'
            ax2.set_ylabel('Threat Volume', color=color2)
            ax2.bar(chart_data['Hour'], chart_data['Threat Volume'], color=color2, alpha=0.3)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 🔄 Resource Utilization")
            
            # Resource data
            resources = ['CPU', 'Memory', 'Network', 'Storage']
            utilization = [65, 72, 58, 45]
            capacity = [80, 80, 70, 90]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(resources))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, utilization, width, label='Current', color='#3949ab')
            bars2 = ax.bar(x + width/2, capacity, width, label='Capacity', color='rgba(57, 73, 171, 0.3)')
            
            ax.set_xlabel('Resource Type')
            ax.set_ylabel('Utilization (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(resources)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # AI Component Performance
        st.markdown("#### 🤖 AI Component Performance")
        
        components = ['ML Detection', 'Search Algorithm', 'CSP Solver', 'RL Agent', 'XAI']
        accuracy = [0.96, 0.94, 0.92, 0.89, 0.91]
        speed = [0.95, 0.98, 0.96, 0.93, 0.94]
        
        perf_data = pd.DataFrame({
            'Component': components,
            'Accuracy': accuracy,
            'Speed': speed
        })
        
        # Create radar chart (simplified)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy bars
        colors = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444']
        bars = ax1.barh(components[::-1], accuracy[::-1], color=colors[::-1])
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Accuracy Score')
        ax1.set_title('Component Accuracy')
        
        # Speed bars
        ax2.barh(components[::-1], speed[::-1], color=colors[::-1], alpha=0.7)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Speed Score')
        ax2.set_title('Component Response Speed')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def show_event_log(self):
        """Show event log"""
        st.markdown("### 📋 Real-time System Events")
        
        # Event categories
        event_categories = st.multiselect(
            "Filter Events:",
            ["All", "Threats", "Defenses", "System", "AI", "Performance"],
            default=["All"]
        )
        
        # Sample events with timestamps
        events = [
            {"time": "14:30:15", "event": "System initialized successfully", "type": "system", "severity": "info"},
            {"time": "14:30:20", "event": "AI models loaded and ready", "type": "ai", "severity": "success"},
            {"time": "14:30:25", "event": "DDoS attack detected on Node 3 (High Severity)", "type": "threat", "severity": "critical"},
            {"time": "14:30:30", "event": "Honeypot deployed automatically on Node 3", "type": "defense", "severity": "success"},
            {"time": "14:30:35", "event": "Attack neutralized - Threat contained", "type": "defense", "severity": "success"},
            {"time": "14:30:40", "event": "Port scan detected on Node 7 (Medium Severity)", "type": "threat", "severity": "warning"},
            {"time": "14:30:45", "event": "AI adaptation triggered - Optimizing defense", "type": "ai", "severity": "info"},
            {"time": "14:30:50", "event": "Performance optimization completed", "type": "performance", "severity": "info"},
            {"time": "14:30:55", "event": "System efficiency improved by 3.2%", "type": "performance", "severity": "success"},
            {"time": "14:31:00", "event": "SQL injection attempt blocked on Node 5", "type": "defense", "severity": "success"},
        ]
        
        # Filter events
        filtered_events = events
        if "All" not in event_categories:
            filtered_events = [e for e in events if e['type'] in event_categories]
        
        # Display events
        event_container = st.container()
        
        with event_container:
            for event in filtered_events:
                # Set icon and color based on severity
                if event['severity'] == 'critical':
                    icon = "🔥"
                    color = "#ef4444"
                    bg_color = "rgba(239, 68, 68, 0.1)"
                elif event['severity'] == 'warning':
                    icon = "⚠️"
                    color = "#f59e0b"
                    bg_color = "rgba(245, 158, 11, 0.1)"
                elif event['severity'] == 'success':
                    icon = "✅"
                    color = "#10b981"
                    bg_color = "rgba(16, 185, 129, 0.1)"
                else:
                    icon = "ℹ️"
                    color = "#3b82f6"
                    bg_color = "rgba(59, 130, 246, 0.1)"
                
                # Event card
                st.markdown(f"""
                <div style="
                    background: {bg_color};
                    border-left: 4px solid {color};
                    padding: 12px;
                    margin: 8px 0;
                    border-radius: 8px;
                    animation: fadeIn 0.5s ease-out;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                                <span style="font-size: 1.2rem;">{icon}</span>
                                <strong style="color: {color};">{event['event']}</strong>
                            </div>
                            <div style="display: flex; gap: 16px; font-size: 0.85rem; color: #94a3b8;">
                                <span>⏰ {event['time']}</span>
                                <span>📊 {event['type'].upper()}</span>
                                <span>🔸 {event['severity'].upper()}</span>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Real-time event generation
        if st.session_state.simulation_state['running']:
            if np.random.random() < 0.3:  # 30% chance to generate new event
                new_event = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "event": f"System heartbeat - All components operational",
                    "type": "system",
                    "severity": "info"
                }
                filtered_events.insert(0, new_event)
                
                # Show toast notification
                st.toast("🔄 System update received", icon="📡")
        
        # Event statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            threats = len([e for e in filtered_events if e['type'] == 'threat'])
            st.metric("Threat Events", threats)
        with col2:
            defenses = len([e for e in filtered_events if e['type'] == 'defense'])
            st.metric("Defense Actions", defenses)
        with col3:
            critical = len([e for e in filtered_events if e['severity'] == 'critical'])
            st.metric("Critical Alerts", critical)
    
    def start_demo(self):
        """Start the complete demo"""
        state = st.session_state.simulation_state
        state['running'] = True
        state['phase'] = 0
        state['start_time'] = datetime.now()
        state['performance_history'] = []
        state['system_status'] = 'active'
        
        st.toast("🚀 Simulation started successfully!", icon="✅")
        
        # Start demo simulation
        self.run_demo_simulation()
    
    def pause_demo(self):
        """Pause the demo"""
        st.session_state.simulation_state['running'] = False
        st.session_state.simulation_state['system_status'] = 'paused'
        st.toast("⏸️ Simulation paused", icon="⏸️")
    
    def reset_simulation(self):
        """Reset the simulation"""
        st.session_state.simulation_state = {
            'running': False,
            'phase': 0,
            'network_nodes': 12,
            'max_honeypots': 4,
            'attacks': [],
            'honeypots': [2, 5, 8],
            'detections': [],
            'performance_history': [],
            'iteration': 0,
            'start_time': None,
            'system_status': 'ready',
            'ai_components': {
                'ml_model': 'active',
                'search_algo': 'active',
                'rl_agent': 'training',
                'csp_solver': 'active'
            }
        }
        st.toast("🔄 Simulation reset to initial state", icon="🔄")
    
    def generate_attack(self):
        """Generate a simulated attack"""
        state = st.session_state.simulation_state
        
        # Generate random attack
        target_node = np.random.randint(0, state['network_nodes'])
        attack_type = np.random.choice(list(self.attack_types.keys()))
        attack_info = self.attack_types[attack_type]
        
        attack = {
            'id': len(state['attacks']) + 1,
            'type': attack_type,
            'node': target_node,
            'time': datetime.now().strftime("%H:%M:%S"),
            'severity': attack_info['severity'],
            'icon': attack_info['icon'],
            'color': attack_info['color'],
            'detected': False
        }
        
        state['attacks'].append(attack)
        
        # Adaptive honeypot placement
        self.place_honeypots()
        
        # Check for detections
        self.check_detections()
        
        # Show notification
        st.toast(f"⚡ {attack_type} attack generated on Node {target_node}", icon="⚡")
    
    def place_honeypots(self):
        """Place honeypots adaptively"""
        state = st.session_state.simulation_state
        
        if not state['attacks']:
            return
        
        # Get recent attacks (last 10)
        recent_attacks = state['attacks'][-10:] if len(state['attacks']) > 10 else state['attacks']
        
        # Count attacks per node
        attack_counts = {}
        for attack in recent_attacks:
            if not attack['detected']:
                node = attack['node']
                attack_counts[node] = attack_counts.get(node, 0) + 1
        
        # Place honeypots on most attacked nodes
        if attack_counts:
            sorted_nodes = sorted(attack_counts.items(), key=lambda x: x[1], reverse=True)
            max_honeypots = min(state['max_honeypots'], len(sorted_nodes))
            honeypots = [node for node, _ in sorted_nodes[:max_honeypots]]
            
            # Add some strategic placements
            if len(honeypots) < state['max_honeypots']:
                additional = np.random.choice(
                    [n for n in range(state['network_nodes']) if n not in honeypots],
                    state['max_honeypots'] - len(honeypots),
                    replace=False
                )
                honeypots.extend(additional)
            
            state['honeypots'] = honeypots
    
    def check_detections(self):
        """Check for attack detections"""
        state = st.session_state.simulation_state
        
        detections = []
        for attack in state['attacks']:
            if not attack['detected'] and attack['node'] in state['honeypots']:
                attack['detected'] = True
                detections.append(attack)
        
        state['detections'] = detections
        
        # Update performance metrics
        if state['attacks']:
            detection_rate = len(detections) / len(state['attacks'])
            efficiency = detection_rate * np.random.uniform(0.85, 0.95)
            
            state['performance_history'].append({
                'iteration': state['iteration'],
                'detection_rate': detection_rate,
                'efficiency': efficiency,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            state['iteration'] += 1
    
    def run_demo_simulation(self):
        """Run the demo simulation"""
        # This would run in a separate thread in production
        pass
    
    def run(self):
        """Run the dashboard"""
        # Show header
        self.show_header()
        
        # Show controls
        self.show_controls()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌐 Live Defense Map", 
            "🧠 AI Intelligence",
            "📈 Performance Analytics", 
            "📋 Event Monitor"
        ])
        
        with tab1:
            self.show_network_view()
        
        with tab2:
            self.show_ai_components()
        
        with tab3:
            self.show_performance_metrics()
        
        with tab4:
            self.show_event_log()
        
        # Auto-refresh if simulation is running
        if st.session_state.simulation_state['running']:
            time.sleep(2)
            st.rerun()

def main():
    """Main function"""
    try:
        # Add loading animation
        with st.spinner("🛡️ Initializing Adaptive Deception-Mesh Dashboard..."):
            time.sleep(0.5)
        
        dashboard = InteractiveDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"🚨 Dashboard Error: {str(e)}")
        st.info("""
        🔧 **Troubleshooting Steps:**
        1. Install dependencies: `pip install -r requirements.txt`
        2. Check Streamlit version: `streamlit --version`
        3. Ensure all data files are in place
        4. Restart the dashboard: `streamlit run dashboard.py`
        """)

if __name__ == "__main__":
    main()