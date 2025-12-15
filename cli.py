#!/usr/bin/env python3
"""
Command Line Interface for Adaptive Deception Mesh
Complete working version
"""

import click
import requests
import json
import time
import sys
import os
import subprocess
import signal
import threading
import atexit
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
BASE_URL = "http://localhost:5000"
SERVER_PROCESS = None

class AdaptiveDeceptionMeshCLI:
    def __init__(self):
        self.base_url = BASE_URL
    
    def check_server(self):
        """Check if server is running - tries multiple ports"""
        ports_to_try = [5000, 5001, 5002, 5003, 5004]
        
        for port in ports_to_try:
            url = f"http://localhost:{port}"
            try:
                response = requests.get(f"{url}/health", timeout=1)
                if response.status_code == 200:
                    self.base_url = url
                    return True
            except:
                continue
        return False
    
    def display_banner(self):
        """Display application banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║    █████╗ ██████╗  █████╗ ██████╗ ████████╗██╗██╗   ██╗███████╗         ║
║   ██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██║   ██║██╔════╝         ║
║   ███████║██║  ██║███████║██║  ██║   ██║   ██║██║   ██║█████╗           ║
║   ██╔══██║██║  ██║██╔══██║██║  ██║   ██║   ██║╚██╗ ██╔╝██╔══╝           ║
║   ██║  ██║██████╔╝██║  ██║██████╔╝   ██║   ██║ ╚████╔╝ ███████╗         ║
║   ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═══╝  ╚══════╝         ║
║                                                                          ║
║                  Adaptive Deception-Mesh: Intelligent Honeynet           ║
║                  Ghulam Ishaq Khan Institute, Topi - CS351 Project       ║
║                  UNSW-NB15 Dataset | AI-Powered Cybersecurity            ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
        """
        console.print(banner, style="bold cyan")
        
        info_panel = Panel(
            "[bold]Project Details:[/bold]\n"
            "Course: Artificial Intelligence (CS351)\n"
            "Instructor: Mr. Ahmed Nawaz\n"
            "Group Members: Zulqarnain Umar (2023556), Muhammad Ismail (2023452), Awais Khan (2023139)\n"
            "Technologies: Flask, Scikit-learn, TensorFlow, SHAP, RL, NetworkX",
            title="Project Information",
            border_style="blue"
        )
        console.print(info_panel)
    
    def display_status(self):
        """Display system status"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/status", timeout=3)
            if response.status_code == 200:
                data = response.json()
                
                panels = []
                
                # System Status Panel
                status_text = f"""[bold green]✓ {data['system_status']}[/bold green]
Time: {time.strftime('%H:%M:%S')}
Nodes: [bold cyan]{data['nodes']}[/bold cyan] | Honeypots: [bold red]{data['honeypots']}[/bold red]
Detection: [bold]{data['detection_accuracy']*100:.1f}%[/bold] | RL Reward: [bold]{data['rl_agent_reward']:.3f}[/bold]
CPU: {data['cpu_usage']}% | Memory: {data['memory_usage']}%
Uptime: {data.get('uptime', 'N/A')}"""
                
                panels.append(Panel(status_text, title="🖥️ System Status", border_style="green"))
                
                # Attack Statistics Panel
                attack_text = f"""Total: [yellow]{data['total_attacks']}[/yellow]
Blocked: [green]{data['blocked_attacks']}[/green]
Analysed: [cyan]{data['analysed_attacks']}[/cyan]
High Severity: [red]{data['high_severity']}[/red]"""
                
                panels.append(Panel(attack_text, title="🚨 Attack Statistics", border_style="yellow"))
                
                # Display panels in columns
                console.print(Columns(panels))
                
            else:
                console.print(f"[red]Error fetching status: {response.status_code}[/red]")
        except Exception as e:
            console.print(f"[red]Connection error: {e}[/red]")
            console.print(f"[dim]Trying to connect to: {self.base_url}[/dim]")
    
    def display_ml_metrics(self):
        """Display ML model metrics"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/ml-metrics", timeout=3)
            if response.status_code == 200:
                data = response.json()
                
                console.print(Panel(
                    "[bold cyan]🤖 ML DETECTION MODELS[/bold cyan]\n"
                    "[dim]UNSW-NB15 Trained | Ensemble Learning[/dim]",
                    title="Machine Learning Performance",
                    border_style="cyan"
                ))
                
                # Random Forest Table
                rf_table = Table(title="🌲 Random Forest Classifier", show_header=True, header_style="bold magenta")
                rf_table.add_column("Metric", style="cyan")
                rf_table.add_column("Value", style="green")
                
                rf_metrics = data['random_forest']
                rf_table.add_row("Accuracy", f"{rf_metrics['accuracy']*100:.2f}%")
                rf_table.add_row("Precision", f"{rf_metrics.get('precision', 0.94):.4f}")
                rf_table.add_row("Recall", f"{rf_metrics.get('recall', 0.97):.4f}")
                rf_table.add_row("F1 Score", f"{rf_metrics.get('f1_score', 0.93):.4f}")
                
                console.print(rf_table)
                
                # Logistic Regression Table
                lr_table = Table(title="📈 Logistic Regression", show_header=True, header_style="bold yellow")
                lr_table.add_column("Metric", style="cyan")
                lr_table.add_column("Value", style="green")
                
                lr_metrics = data['logistic_regression']
                lr_table.add_row("Accuracy", f"{lr_metrics['accuracy']*100:.2f}%")
                lr_table.add_row("Precision", f"{lr_metrics.get('precision', 0.94):.4f}")
                lr_table.add_row("Recall", f"{lr_metrics.get('recall', 0.92):.4f}")
                lr_table.add_row("F1 Score", f"{lr_metrics.get('f1_score', 0.92):.4f}")
                
                console.print(lr_table)
                
                # K-Means Clustering
                km_table = Table(title="🔍 K-Means Clustering Results", show_header=True, header_style="bold blue")
                km_table.add_column("Cluster Type", style="cyan")
                km_table.add_column("Percentage", style="green")
                
                for cluster, percentage in data['kmeans_clustering'].items():
                    km_table.add_row(cluster, f"{percentage*100:.1f}%")
                
                console.print(km_table)
                
                console.print("\n[bold yellow]📊 Ensemble learning with UNSW-NB15 dataset[/bold yellow]")
                
            else:
                console.print("[red]Error fetching ML metrics[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def display_rl_metrics(self):
        """Display RL agent metrics"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/rl-metrics", timeout=3)
            if response.status_code == 200:
                data = response.json()
                
                console.print(Panel(
                    "[bold yellow]🤖 RL AGENT (Q-LEARNING)[/bold yellow]\n"
                    "[dim]Adaptive Deployment Strategy[/dim]",
                    title="Reinforcement Learning",
                    border_style="yellow"
                ))
                
                # Parameters table
                params_table = Table(title="⚙️ Agent Parameters", show_header=True, header_style="bold")
                params_table.add_column("Parameter", style="cyan")
                params_table.add_column("Value", style="green")
                
                params_table.add_row("Epsilon", f"{data['epsilon']:.3f}")
                params_table.add_row("Learning Rate", f"{data['learning_rate']:.3f}")
                params_table.add_row("Gamma", f"{data['gamma']:.2f}")
                params_table.add_row("Current Reward", f"{data['reward']:.3f}")
                
                console.print(params_table)
                
                # Recent decisions
                if data.get('recent_decisions'):
                    decisions_table = Table(title="🎯 Recent Agent Decisions", show_header=True, header_style="bold")
                    decisions_table.add_column("Action", style="cyan")
                    decisions_table.add_column("Reward", style="green")
                    decisions_table.add_column("Q-Value", style="yellow")
                    
                    for decision in data['recent_decisions']:
                        decisions_table.add_row(
                            decision['action'],
                            f"{decision['reward']:.2f}",
                            f"{decision['q_value']:.2f}"
                        )
                    
                    console.print(decisions_table)
                
            else:
                console.print("[red]Error fetching RL metrics[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def display_csp_constraints(self):
        """Display CSP constraints"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/csp-constraints", timeout=3)
            if response.status_code == 200:
                data = response.json()
                
                console.print(Panel(
                    "[bold magenta]🧩 CSP RESOURCE MANAGER[/bold magenta]\n"
                    "[dim]Constraint Satisfaction Solver[/dim]",
                    title="Constraint Satisfaction",
                    border_style="magenta"
                ))
                
                # Constraints table
                constraints_table = Table(title="🔒 Active Constraints", show_header=True, header_style="bold")
                constraints_table.add_column("Constraint", style="cyan")
                constraints_table.add_column("Condition", style="white")
                
                for constraint in data['active_constraints']:
                    constraints_table.add_row(
                        constraint['name'],
                        constraint['constraint']
                    )
                
                console.print(constraints_table)
                
                # Resource distribution
                console.print("\n[bold]📊 Resource Distribution:[/bold]")
                
                res_table = Table(title="Current Usage vs Limits", show_header=True, header_style="bold")
                res_table.add_column("Resource", style="cyan")
                res_table.add_column("Usage", style="green")
                res_table.add_column("Limit", style="yellow")
                
                current = data['current_values']
                limits = data['resource_distribution']
                
                resources = [
                    ("CPU", f"{current['cpu']}%", f"{limits['cpu']}%"),
                    ("Memory", f"{current['memory']}%", f"{limits['memory']}%"),
                    ("Honeypots", str(current['honeypots']), f"≥ {limits['storage']}"),
                    ("Nodes", str(current['nodes']), f"≤ {limits['network']}")
                ]
                
                for name, usage, limit in resources:
                    res_table.add_row(name, usage, limit)
                
                console.print(res_table)
                
            else:
                console.print("[red]Error fetching CSP constraints[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def display_explainability(self):
        """Display explainability information"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/explainability", timeout=3)
            if response.status_code == 200:
                data = response.json()
                
                console.print(Panel(
                    "[bold cyan]🔍 XAI EXPLAINABILITY[/bold cyan]\n"
                    "[dim]Model Interpretability with SHAP[/dim]",
                    title="Explainable AI",
                    border_style="cyan"
                ))
                
                # Feature importance table
                feature_table = Table(title="📊 SHAP Feature Importance", show_header=True, header_style="bold")
                feature_table.add_column("Feature", style="cyan")
                feature_table.add_column("Importance", style="green")
                
                for feature, importance in data['shap_feature_importance'].items():
                    feature_table.add_row(feature, f"{importance:.3f}")
                
                console.print(feature_table)
                
                # Feature analysis
                console.print("\n[bold]🔎 Feature Analysis:[/bold]")
                for analysis in data.get('feature_analysis', []):
                    console.print(f"[cyan]{analysis['feature']}:[/cyan] {analysis['value']} (Impact: {analysis['importance']:.3f})")
                
                console.print("\n[dim]SHAP provides global feature importance using Shapley values from game theory[/dim]")
                
            else:
                console.print("[red]Error fetching explainability data[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def display_live_dashboard(self):
        """Display live updating dashboard"""
        try:
            console.clear()
            self.display_banner()
            
            with Live(refresh_per_second=1, screen=False) as live:
                while True:
                    try:
                        status_response = requests.get(f"{self.base_url}/api/v1/status", timeout=2)
                        
                        if status_response.status_code == 200:
                            status = status_response.json()
                            
                            # Create layout
                            layout = Layout()
                            
                            # Header
                            header = Panel(
                                f"[bold cyan]Adaptive Deception-Mesh Platform[/bold cyan] | "
                                f"[dim]AI-Powered Honeynet Simulation | URL: {self.base_url}[/dim]",
                                border_style="cyan"
                            )
                            
                            # System Status Panel
                            status_panel = Panel(
                                f"[bold]🖥️ SYSTEM STATUS[/bold]\n"
                                f"Nodes: [cyan]{status['nodes']}[/cyan] | "
                                f"Honeypots: [red]{status['honeypots']}[/red]\n"
                                f"Detection: [green]{status['detection_accuracy']*100:.1f}%[/green] | "
                                f"RL Reward: [yellow]{status['rl_agent_reward']:.3f}[/yellow]\n"
                                f"CPU: {status['cpu_usage']}% | "
                                f"Memory: {status['memory_usage']}%\n"
                                f"Status: [green]{status['system_status']}[/green]",
                                border_style="green"
                            )
                            
                            # Attack Statistics Panel
                            attack_panel = Panel(
                                f"[bold]🚨 ATTACK STATISTICS[/bold]\n"
                                f"Total: [yellow]{status['total_attacks']}[/yellow] | "
                                f"Blocked: [green]{status['blocked_attacks']}[/green]\n"
                                f"Analysed: [cyan]{status['analysed_attacks']}[/cyan] | "
                                f"High Severity: [red]{status['high_severity']}[/red]",
                                border_style="yellow"
                            )
                            
                            # Footer
                            footer = Panel(
                                f"[dim]Time: {time.strftime('%H:%M:%S')} | "
                                f"Press Ctrl+C to exit | Adaptive Deception Mesh v1.0[/dim]",
                                border_style="dim"
                            )
                            
                            # Combine all panels
                            layout.split_column(
                                Layout(header, size=3),
                                Layout(status_panel, size=4),
                                Layout(attack_panel, size=4),
                                Layout(footer, size=2)
                            )
                            
                            live.update(layout)
                        
                    except requests.RequestException:
                        live.update(Panel(
                            "[red]⚠ Connection lost. Trying to reconnect...[/red]",
                            border_style="red"
                        ))
                    
                    time.sleep(2)
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Live dashboard stopped[/yellow]")
        except Exception as e:
            console.print(f"[red]Error in live dashboard: {e}[/red]")

def start_server():
    """Start the Flask server - SIMPLIFIED VERSION"""
    global SERVER_PROCESS
    
    console.print("[yellow]🚀 Starting Adaptive Deception Mesh server...[/yellow]")
    
    # Clean up any existing server
    stop_server()
    
    try:
        # Start server directly using subprocess with proper flags
        SERVER_PROCESS = subprocess.Popen(
            [sys.executable, "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        
        # Register cleanup
        atexit.register(stop_server)
        
        # Give server time to start
        console.print("[dim]Server is starting... (waiting 5 seconds)[/dim]")
        time.sleep(5)
        
        # Check if server is running
        cli_tool = AdaptiveDeceptionMeshCLI()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Starting server...", total=None)
            
            for i in range(15):
                if cli_tool.check_server():
                    progress.stop()
                    console.print("[green]✅ Server started successfully![/green]")
                    console.print(f"[dim]API available at: {cli_tool.base_url}[/dim]")
                    return True
                time.sleep(1)
        
        console.print("[red]❌ Server failed to start[/red]")
        
        # Try to get error output
        try:
            stdout, stderr = SERVER_PROCESS.communicate(timeout=2)
            if stderr:
                console.print(f"[red]Server error: {stderr[:200]}[/red]")
        except:
            pass
            
        return False
        
    except Exception as e:
        console.print(f"[red]❌ Error starting server: {e}[/red]")
        return False

def stop_server():
    """Stop the Flask server"""
    global SERVER_PROCESS
    
    if SERVER_PROCESS:
        try:
            if sys.platform == "win32":
                SERVER_PROCESS.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                SERVER_PROCESS.terminate()
            
            SERVER_PROCESS.wait(timeout=3)
            console.print("[green]✅ Server stopped successfully![/green]")
        except:
            try:
                SERVER_PROCESS.kill()
                console.print("[yellow]⚠ Server forcefully stopped[/yellow]")
            except:
                pass
        
        SERVER_PROCESS = None
    else:
        # Try to kill any Python processes on our ports
        try:
            if sys.platform == "win32":
                os.system("taskkill /F /IM python.exe >nul 2>&1")
                os.system("taskkill /F /IM python3.exe >nul 2>&1")
            else:
                os.system("pkill -f 'python.*app.py' 2>/dev/null")
        except:
            pass

# CLI Commands
@click.group()
def cli():
    """Adaptive Deception Mesh - Intelligent Honeynet System"""
    pass

@cli.command()
def start():
    """Start the Flask server"""
    cli_tool = AdaptiveDeceptionMeshCLI()
    cli_tool.display_banner()
    
    if cli_tool.check_server():
        console.print("[green]✅ Server is already running[/green]")
        console.print(f"[dim]API available at: {cli_tool.base_url}[/dim]")
        console.print("\n[bold]Try these commands:[/bold]")
        console.print("  python cli.py status      # Show system status")
        console.print("  python cli.py all         # Show all information")
    else:
        if start_server():
            console.print("\n[bold]🎉 Server started! Try these commands:[/bold]")
            console.print("  python cli.py status      # Show system status")
            console.print("  python cli.py ml          # Show ML metrics")
            console.print("  python cli.py rl          # Show RL metrics")
            console.print("  python cli.py all         # Show all information")
            console.print("  python cli.py dashboard   # Live dashboard")
            console.print("  python cli.py stop        # Stop server")

@cli.command()
def stop():
    """Stop the Flask server"""
    stop_server()

@cli.command()
def status():
    """Display system status"""
    cli_tool = AdaptiveDeceptionMeshCLI()
    cli_tool.display_banner()
    
    if not cli_tool.check_server():
        console.print("[red]❌ Server is not running[/red]")
        console.print("[yellow]Use 'python cli.py start' to start the server[/yellow]")
        return
    
    cli_tool.display_status()

@cli.command()
def ml():
    """Display ML model metrics"""
    cli_tool = AdaptiveDeceptionMeshCLI()
    cli_tool.display_banner()
    
    if not cli_tool.check_server():
        console.print("[red]❌ Server is not running[/red]")
        return
    
    cli_tool.display_ml_metrics()

@cli.command()
def rl():
    """Display RL agent metrics"""
    cli_tool = AdaptiveDeceptionMeshCLI()
    cli_tool.display_banner()
    
    if not cli_tool.check_server():
        console.print("[red]❌ Server is not running[/red]")
        return
    
    cli_tool.display_rl_metrics()

@cli.command()
def csp():
    """Display CSP constraints"""
    cli_tool = AdaptiveDeceptionMeshCLI()
    cli_tool.display_banner()
    
    if not cli_tool.check_server():
        console.print("[red]❌ Server is not running[/red]")
        return
    
    cli_tool.display_csp_constraints()

@cli.command()
def explain():
    """Display explainability information"""
    cli_tool = AdaptiveDeceptionMeshCLI()
    cli_tool.display_banner()
    
    if not cli_tool.check_server():
        console.print("[red]❌ Server is not running[/red]")
        return
    
    cli_tool.display_explainability()

@cli.command()
def dashboard():
    """Display live dashboard"""
    cli_tool = AdaptiveDeceptionMeshCLI()
    cli_tool.display_banner()
    
    if not cli_tool.check_server():
        console.print("[red]❌ Server is not running[/red]")
        return
    
    console.print("[cyan]📊 Starting live dashboard... Press Ctrl+C to exit[/cyan]")
    cli_tool.display_live_dashboard()

@cli.command()
def all():
    """Display all information"""
    cli_tool = AdaptiveDeceptionMeshCLI()
    cli_tool.display_banner()
    
    if not cli_tool.check_server():
        console.print("[red]❌ Server is not running[/red]")
        return
    
    console.print("[bold cyan]="*60 + "[/bold cyan]")
    console.print("[bold cyan]📋 SYSTEM STATUS[/bold cyan]")
    console.print("[bold cyan]="*60 + "[/bold cyan]")
    cli_tool.display_status()
    
    console.print("\n[bold cyan]="*60 + "[/bold cyan]")
    console.print("[bold cyan]🤖 MACHINE LEARNING METRICS[/bold cyan]")
    console.print("[bold cyan]="*60 + "[/bold cyan]")
    cli_tool.display_ml_metrics()
    
    console.print("\n[bold cyan]="*60 + "[/bold cyan]")
    console.print("[bold cyan]🎯 REINFORCEMENT LEARNING[/bold cyan]")
    console.print("[bold cyan]="*60 + "[/bold cyan]")
    cli_tool.display_rl_metrics()
    
    console.print("\n[bold cyan]="*60 + "[/bold cyan]")
    console.print("[bold cyan]🧩 CSP CONSTRAINTS[/bold cyan]")
    console.print("[bold cyan]="*60 + "[/bold cyan]")
    cli_tool.display_csp_constraints()
    
    console.print("\n[bold cyan]="*60 + "[/bold cyan]")
    console.print("[bold cyan]🔍 EXPLAINABILITY[/bold cyan]")
    console.print("[bold cyan]="*60 + "[/bold cyan]")
    cli_tool.display_explainability()

@cli.command()
def test():
    """Test server connection"""
    cli_tool = AdaptiveDeceptionMeshCLI()
    
    console.print("[bold]Testing server connection...[/bold]")
    
    if cli_tool.check_server():
        console.print(f"[green]✅ Server is running at: {cli_tool.base_url}[/green]")
        
        try:
            response = requests.get(f"{cli_tool.base_url}/health", timeout=2)
            if response.status_code == 200:
                console.print("[green]✅ Health check passed[/green]")
                console.print(f"Response: {response.json()}")
        except Exception as e:
            console.print(f"[red]Health check failed: {e}[/red]")
    else:
        console.print("[red]❌ Server is not running[/red]")

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)