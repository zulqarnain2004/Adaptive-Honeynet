"""
Adaptive Deception Mesh — Complete Integrated Implementation
File: adaptive_deception_mesh_complete.py
Purpose: Complete, runnable reference implementation that integrates the concepts
from Project Proposal, Progress Report I-II-III:
 - Robust preprocessing (re-uses data_pipeline.py if available)
 - Baseline ML (Logistic Regression, Random Forest, XGBoost)
 - Advanced DL (FCNN classifier + Autoencoder anomaly detector)
 - Explainability (SHAP + LIME hooks with caching)
 - A* planner + CSP validator (re-uses agent_and_astar.py if available)
 - Reinforcement Learning (Gym-compatible environment + DQN agent)
 - Integration glue that ties ML outputs into the planner and RL reward
 - CLI entrypoints for: preprocess, train_models, run_agent_example, train_rl

Notes:
 - This file is intended as a reference and end-to-end demo. Tweak paths,
   hyperparameters and resource settings for production.
 - It will try to use the uploaded utility modules located at:
     /mnt/data/data_pipeline.py
     /mnt/data/agent_and_astar.py
   If found, it will import them and reuse preprocessing/A* logic. Otherwise
   this file includes fallback implementations.

Run examples:
  python adaptive_deception_mesh_complete.py preprocess
  python adaptive_deception_mesh_complete.py train_models
  python adaptive_deception_mesh_complete.py agent_example
  python adaptive_deception_mesh_complete.py train_rl

"""

import os
import sys
import time
import json
import math
import random
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

# --- ML / DL dependencies ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# try optional libs
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None

try:
    import gym
    from gym import spaces
except Exception:
    gym = None

# Try to import the user's uploaded helpers if present
UPLOADED_DATA_PIPELINE = '/mnt/data/data_pipeline.py'
UPLOADED_AGENT = '/mnt/data/agent_and_astar.py'
use_uploaded_pipeline = False
use_uploaded_agent = False
if os.path.exists(UPLOADED_DATA_PIPELINE):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('uploaded_data_pipeline', UPLOADED_DATA_PIPELINE)
        uploaded_data_pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(uploaded_data_pipeline)
        use_uploaded_pipeline = True
    except Exception as e:
        print("Could not import uploaded data_pipeline.py:", e)
if os.path.exists(UPLOADED_AGENT):
    try:
        spec = importlib.util.spec_from_file_location('uploaded_agent', UPLOADED_AGENT)
        uploaded_agent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(uploaded_agent)
        use_uploaded_agent = True
    except Exception as e:
        print("Could not import uploaded agent_and_astar.py:", e)

# -----------------------------
# Config / Paths
# -----------------------------
ROOT = os.getcwd()
MODELS_DIR = os.path.join(ROOT, 'models_adaptive_mesh')
os.makedirs(MODELS_DIR, exist_ok=True)
PREPROCESSED_TRAIN = os.path.join(ROOT, 'preprocessed_train.csv')
PREPROCESSED_TEST = os.path.join(ROOT, 'preprocessed_test.csv')

RANDOM_STATE = 42

# -----------------------------
# Utilities
# -----------------------------
def safe_save_model(model, name):
    path = os.path.join(MODELS_DIR, name)
    joblib.dump(model, path)
    return path

# -----------------------------
# 1) Preprocessing wrapper
# -----------------------------
def run_preprocessing_with_uploaded(input_path: str = None):
    """
    If uploaded data_pipeline module is available, call it to generate preprocessed
    train/test files. The uploaded pipeline writes files to Desktop by default;
    this wrapper will try to copy them into the local working directory.
    """
    if not use_uploaded_pipeline:
        raise RuntimeError('Uploaded preprocessing module not available')
    # call its main() (it writes to OUT_TRAIN/OUT_TEST to Desktop)
    uploaded_data_pipeline.main()
    # Try to locate outputs from that module by reading exported variable names
    # The uploaded module defines OUT_TRAIN, OUT_TEST variables
    try:
        out_train = getattr(uploaded_data_pipeline, 'OUT_TRAIN')
        out_test = getattr(uploaded_data_pipeline, 'OUT_TEST')
    except Exception:
        raise RuntimeError('Uploaded module did not expose OUT_TRAIN/OUT_TEST')
    # copy files into working dir
    if os.path.exists(out_train):
        df_train = pd.read_csv(out_train)
        df_train.to_csv(PREPROCESSED_TRAIN, index=False)
    else:
        raise FileNotFoundError(f'Expected train file not found: {out_train}')
    if os.path.exists(out_test):
        df_test = pd.read_csv(out_test)
        df_test.to_csv(PREPROCESSED_TEST, index=False)
    else:
        raise FileNotFoundError(f'Expected test file not found: {out_test}')
    print('Preprocessed files copied to working dir:', PREPROCESSED_TRAIN, PREPROCESSED_TEST)


def simple_preprocess_from_csv(input_csv: str, train_out=PREPROCESSED_TRAIN, test_out=PREPROCESSED_TEST):
    """
    Minimal preprocessing fallback if uploaded pipeline not available.
    Assumes CSV has a target column with name containing 'label' or 'attack'.
    Produces scaled numeric features, label_bin column as binary target, and saves 70/30 split.
    """
    df = pd.read_csv(input_csv)
    # find label
    candidates = [c for c in df.columns if 'label' in c.lower() or 'attack' in c.lower()]
    if not candidates:
        raise RuntimeError('No label-like column found in CSV')
    label_col = candidates[0]
    # binary label
    df['label_bin'] = df[label_col].apply(lambda x: 0 if str(x).lower() in ('0','normal','benign','none') else 1)
    # drop obvious id cols
    drop_cols = [c for c in df.columns if c.lower() in ('id','srcip','dstip','sourceip','destip')]
    X = df.drop(columns=drop_cols + [label_col, 'label_bin'], errors='ignore')
    # keep numeric only
    X = X.select_dtypes(include=[np.number]).fillna(0)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    Xf = pd.DataFrame(Xs, columns=X.columns)
    y = df['label_bin']
    X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    tr = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    te = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    tr.to_csv(train_out, index=False)
    te.to_csv(test_out, index=False)
    print('Saved simple preprocessed train/test to', train_out, test_out)

# -----------------------------
# 2) ML training & explainability
# -----------------------------
class ModelManager:
    def _init_(self):
        self.rf = None
        self.lr = None
        self.xgb = None
        self.feature_names = None
        self.shap_cache = None

    def load_data(self):
        if not os.path.exists(PREPROCESSED_TRAIN) or not os.path.exists(PREPROCESSED_TEST):
            raise FileNotFoundError('Preprocessed train/test not found. Run preprocess first or place preprocessed CSVs in working dir.')
        train = pd.read_csv(PREPROCESSED_TRAIN)
        test = pd.read_csv(PREPROCESSED_TEST)
        if 'label_bin' not in train.columns:
            raise RuntimeError('Expected label_bin column in preprocessed files')
        X_train = train.drop(columns=['label_bin'])
        y_train = train['label_bin']
        X_test = test.drop(columns=['label_bin'])
        y_test = test['label_bin']
        self.feature_names = list(X_train.columns)
        return X_train, X_test, y_train, y_test

    def train_classical(self, X_train, y_train, X_test, y_test):
        print('Training RandomForest...')
        self.rf = RandomForestClassifier(n_estimators=200, max_depth=25, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        self.rf.fit(X_train, y_train)
        safe_save_model(self.rf, 'random_forest.pkl')
        print('Training LogisticRegression...')
        self.lr = LogisticRegression(C=0.5, solver='liblinear', class_weight='balanced', max_iter=1000)
        self.lr.fit(X_train, y_train)
        safe_save_model(self.lr, 'logistic_regression.pkl')
        if xgb is not None:
            print('Training XGBoost...')
            self.xgb = xgb.XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
            self.xgb.fit(X_train, y_train)
            safe_save_model(self.xgb, 'xgboost.pkl')
        res = {}
        for name, model in [('RandomForest', self.rf), ('LogisticRegression', self.lr), ('XGBoost', self.xgb)]:
            if model is None:
                continue
            ypred = model.predict(X_test)
            res[name] = {
                'accuracy': float(accuracy_score(y_test, ypred)),
                'precision': float(precision_score(y_test, ypred, zero_division=0)),
                'recall': float(recall_score(y_test, ypred, zero_division=0)),
                'f1': float(f1_score(y_test, ypred, zero_division=0))
            }
        print('Classical model results:', json.dumps(res, indent=2))
        return res

    def build_and_train_fcnn(self, X_train, y_train, X_test, y_test, epochs=20):
        if torch is None:
            print('PyTorch not available; skipping FCNN training.')
            return None
        input_dim = X_train.shape[1]
        class FCNN(nn.Module):
            def _init_(self, input_dim):
                super()._init_()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 128), nn.ReLU(),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, 2)
                )
            def forward(self, x):
                return self.net(x)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FCNN(input_dim).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        ds = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long))
        loader = DataLoader(ds, batch_size=128, shuffle=True)
        for e in range(epochs):
            model.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                out = model(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()
                total += loss.item() * xb.size(0)
            print(f'FCNN epoch {e+1}/{epochs} loss={total/len(ds):.4f}')
        # evaluate
        model.eval()
        with torch.no_grad():
            Xte = torch.tensor(X_test.values, dtype=torch.float32).to(device)
            logits = model(Xte).cpu().numpy()
            ypred = np.argmax(logits, axis=1)
        res = {
            'accuracy': float(accuracy_score(y_test, ypred)),
            'precision': float(precision_score(y_test, ypred, zero_division=0)),
            'recall': float(recall_score(y_test, ypred, zero_division=0)),
            'f1': float(f1_score(y_test, ypred, zero_division=0))
        }
        # persist pytorch model
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'fcnn_state.pt'))
        print('FCNN results:', res)
        return res

    def train_autoencoder(self, X_train, epochs=40):
        if torch is None:
            print('PyTorch not available; skipping Autoencoder training.')
            return None
        input_dim = X_train.shape[1]
        class Autoencoder(nn.Module):
            def _init_(self, dim):
                super()._init_()
                self.enc = nn.Sequential(nn.Linear(dim, 90), nn.ReLU(), nn.Linear(90, 45), nn.ReLU(), nn.Linear(45, 16))
                self.dec = nn.Sequential(nn.Linear(16, 45), nn.ReLU(), nn.Linear(45, 90), nn.ReLU(), nn.Linear(90, dim))
            def forward(self, x):
                return self.dec(self.enc(x))
        model = Autoencoder(input_dim)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        ds = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=128, shuffle=True)
        for e in range(epochs):
            model.train()
            total = 0.0
            for (xb,) in loader:
                xb = xb.to(device)
                opt.zero_grad()
                recon = model(xb)
                loss = crit(recon, xb)
                loss.backward()
                opt.step()
                total += loss.item() * xb.size(0)
            print(f'AE epoch {e+1}/{epochs} loss={total/len(ds):.6f}')
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'autoencoder_state.pt'))
        return model

    def compute_shap(self, background_samples: pd.DataFrame = None):
        if shap is None or self.rf is None:
            print('SHAP or model not available — skipping shap.')
            return None
        if background_samples is None:
            # use a small sample from training features
            background_samples = pd.DataFrame(np.random.choice(0, size=(10, len(self.feature_names))), columns=self.feature_names)
        explainer = shap.TreeExplainer(self.rf)
        self.shap_cache = explainer
        return explainer

    def get_shap_values(self, X: pd.DataFrame):
        if shap is None or self.shap_cache is None:
            return None
        return self.shap_cache.shap_values(X)

    def lime_explain(self, X_train: pd.DataFrame, instance: np.ndarray):
        if LimeTabularExplainer is None or self.rf is None:
            return None
        expl = LimeTabularExplainer(X_train.values, feature_names=self.feature_names, class_names=['normal','attack'], mode='classification')
        return expl.explain_instance(instance, self.rf.predict_proba)

# -----------------------------
# 3) CSP Validator (simple backtracking)
# -----------------------------
class CSPValidator:
    def _init_(self, nodes: Dict[str, Any], cpu_limit: float, ram_limit: float = None):
        # nodes: mapping id -> dict with cost, cpu, ram etc
        self.nodes = nodes
        self.cpu_limit = cpu_limit
        self.ram_limit = ram_limit

    def validate(self, placements: List[str]) -> bool:
        total = 0.0
        for p in placements:
            total += self.nodes[p].get('cost', 0.0)
            if total > self.cpu_limit + 1e-9:
                return False
        return True

# -----------------------------
# 4) A* Agent wrapper (uses uploaded agent if present)
# -----------------------------
if use_uploaded_agent:
    DeceptionAgent = uploaded_agent.DeceptionAgent
    AStarPlanner = uploaded_agent.AStarPlanner
    NetworkNode = uploaded_agent.NetworkNode
else:
    # basic fallback using internal simple planner (not as featured as uploaded one)
    class NetworkNode:
        def _init_(self, id, vulnerability, cost, neighbors=None):
            self.id = id
            self.vulnerability = vulnerability
            self.cost = cost
            self.neighbors = neighbors or []

    class DeceptionAgent:
        def _init_(self, nodes, resource_budget, ml_suspicion=None, suspicion_weight=1.0):
            self.nodes = nodes
            self.budget = resource_budget
            self.ml_suspicion = ml_suspicion or {}
            self.suspicion_weight = suspicion_weight

        def plan_placements(self, coverage_target=None):
            # greedy fallback: pick nodes by highest (vulnerability * (1 + suspicion)) / cost until budget
            nodes_list = list(self.nodes.values())
            score_list = []
            for n in nodes_list:
                s = n.vulnerability * (1.0 + self.ml_suspicion.get(n.id, 0.0) * self.suspicion_weight)
                score_list.append((s / max(1e-6, n.cost), n))
            score_list.sort(reverse=True, key=lambda x: x[0])
            placed = set()
            total_cost = 0.0
            covered = set()
            for sc, n in score_list:
                if total_cost + n.cost > self.budget:
                    continue
                placed.add(n.id)
                total_cost += n.cost
                covered.add(n.id)
                covered.update(n.neighbors)
            return placed, total_cost, covered, {'method': 'greedy_fallback'}

# -----------------------------
# 5) Reinforcement Learning environment + simple DQN
# -----------------------------
class SimpleNetEnv:
    """
    Minimal gym-like environment to train a defender agent that can place/remove honeypots.
    State: vector of node risk scores + binary placements
    Action: for each node -> {0: no-op, 1: add honeypot, 2: remove honeypot}
    Reward: +attack_detected * weight - resource_penalty - false_positive_penalty
    This environment is intentionally small for demonstration and should be extended.
    """
    def _init_(self, node_ids: List[str], initial_risk: Dict[str, float], node_cost: Dict[str, float], attack_prob: Dict[str, float], budget: float):
        if gym is None:
            raise RuntimeError('gym not available')
        self.node_ids = node_ids
        self.n = len(node_ids)
        self.node_cost = node_cost
        self.attack_prob = attack_prob
        self.budget = budget
        # observation: [risk scores..., placements...] length = 2n
        low = np.zeros(2 * self.n, dtype=np.float32)
        high = np.ones(2 * self.n, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # actions: choose one node and apply action {0,1,2} -> discrete(n*3)
        self.action_space = spaces.Discrete(self.n * 3)
        self.reset(initial_risk)

    def reset(self, initial_risk: Dict[str, float] = None):
        self.risk = np.array([initial_risk.get(n, 0.0) for n in self.node_ids], dtype=np.float32)
        self.placements = np.zeros(self.n, dtype=np.float32)
        return np.concatenate([self.risk, self.placements])

    def step(self, action: int):
        node_idx = action // 3
        act = action % 3
        reward = 0.0
        info = {}
        # apply action
        if act == 1:  # add
            # ensure budget
            cost = self.node_cost[self.node_ids[node_idx]]
            current_cost = np.dot(self.placements, [self.node_cost[n] for n in self.node_ids])
            if current_cost + cost <= self.budget:
                self.placements[node_idx] = 1.0
                reward -= 0.1  # small action cost
            else:
                reward -= 0.5  # penalty for invalid action
        elif act == 2:  # remove
            if self.placements[node_idx] == 1.0:
                self.placements[node_idx] = 0.0
                reward -= 0.05
            else:
                reward -= 0.2
        # simulate attacker occurrence: for simplicity, each step attacker may target a random node
        detected = False
        fp = 0
        fn = 0
        for i, nid in enumerate(self.node_ids):
            attacked = random.random() < self.attack_prob.get(nid, 0.0)
            if attacked and self.placements[i] == 1.0:
                reward += 1.0  # detected
                detected = True
            elif attacked and self.placements[i] == 0.0:
                reward -= 1.0  # missed
                fn += 1
            elif not attacked and self.placements[i] == 1.0:
                # false positive-like (wasted resource)
                reward -= 0.02
                fp += 1
        # next observation slightly updates risk by small noise
        self.risk = np.clip(self.risk + np.random.normal(0, 0.01, size=self.n), 0.0, 1.0)
        obs = np.concatenate([self.risk, self.placements])
        done = False
        return obs.astype(np.float32), reward, done, {'detected': detected, 'fp': fp, 'fn': fn}

# DQN (simple) using PyTorch
class DQNAgent:
    def _init_(self, obs_dim, n_actions, hidden=128, lr=1e-3, gamma=0.99):
        if torch is None:
            raise RuntimeError('PyTorch required for DQN')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        class Net(nn.Module):
            def _init_(self, inp, outp):
                super()._init_()
                self.net = nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outp))
            def forward(self, x):
                return self.net(x)
        self.policy_net = Net(obs_dim, n_actions).to(self.device)
        self.target_net = Net(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.opt = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.replay = []
        self.replay_capacity = 5000

    def act(self, obs, eps):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(obs_t)
        return int(q.argmax().cpu().numpy()[0])

    def push(self, transition):
        self.replay.append(transition)
        if len(self.replay) > self.replay_capacity:
            self.replay.pop(0)

    def update(self, batch_size=64):
        if len(self.replay) < batch_size:
            return 0.0
        batch = random.sample(self.replay, batch_size)
        obs_b = torch.tensor([b[0] for b in batch], dtype=torch.float32).to(self.device)
        act_b = torch.tensor([b[1] for b in batch], dtype=torch.long).to(self.device)
        rew_b = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
        next_b = torch.tensor([b[3] for b in batch], dtype=torch.float32).to(self.device)
        done_b = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(self.device)
        qvals = self.policy_net(obs_b)
        q_a = qvals.gather(1, act_b.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            qnext = self.target_net(next_b).max(1)[0]
        target = rew_b + self.gamma * qnext * (1.0 - done_b)
        loss = nn.functional.mse_loss(q_a, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# -----------------------------
# 6) Integration helpers
# -----------------------------
def compute_node_suspicion_from_models(model_mgr: ModelManager, X: pd.DataFrame) -> Dict[str, float]:
    """Compute per-node suspicion scores by averaging model probabilities from RF/XGB/LR.
    Assumes X has an index with node identifiers or flow IDs. Returns dict(index->score)
    """
    probs = []
    if model_mgr.rf is not None:
        probs.append(model_mgr.rf.predict_proba(X)[:, 1])
    if model_mgr.xgb is not None:
        probs.append(model_mgr.xgb.predict_proba(X)[:, 1])
    if model_mgr.lr is not None:
        probs.append(model_mgr.lr.predict_proba(X)[:, 1])
    if not probs:
        # fallback: use random suspicion
        return {str(idx): float(random.random()*0.2) for idx in X.index}
    avg = np.mean(np.vstack(probs), axis=0)
    # map to index
    return {str(idx): float(avg_i) for idx, avg_i in zip(X.index, avg)}

# -----------------------------
# CLI Entrypoints
# -----------------------------
def cmd_preprocess(args):
    # Prefer uploaded pipeline; else run simple preprocess
    if use_uploaded_pipeline:
        run_preprocessing_with_uploaded()
    else:
        if not args:
            print('Provide a CSV path for simple preprocessing:')
            print('  python adaptive_deception_mesh_complete.py preprocess /path/to/UNSW.csv')
            return
        simple_preprocess_from_csv(args[0])


def cmd_train_models(args):
    mm = ModelManager()
    X_train, X_test, y_train, y_test = mm.load_data()
    # train classical models
    results = mm.train_classical(X_train, y_train, X_test, y_test)
    # train fcnn
    mm.build_and_train_fcnn(X_train, y_train, X_test, y_test, epochs=10)
    # train autoencoder
    ae = mm.train_autoencoder(X_train, epochs=10)
    # compute shap cache
    mm.compute_shap(background_samples=X_train.sample(min(50, len(X_train))))
    print('Training complete. Models saved in', MODELS_DIR)


def cmd_agent_example(args):
    # build nodes from a small example (Progress Report I) and run DeceptionAgent
    nodes = {
        'S1': NetworkNode('S1', vulnerability=4, cost=1.5, neighbors=['S2','S3']),
        'S2': NetworkNode('S2', vulnerability=8, cost=2.0, neighbors=['S1','S4']),
        'S3': NetworkNode('S3', vulnerability=6, cost=1.0, neighbors=['S1','S5']),
        'S4': NetworkNode('S4', vulnerability=9, cost=2.5, neighbors=['S2','S5','S6']),
        'S5': NetworkNode('S5', vulnerability=5, cost=1.0, neighbors=['S3','S4']),
        'S6': NetworkNode('S6', vulnerability=7, cost=2.0, neighbors=['S4']),
    }
    agent = DeceptionAgent(nodes, resource_budget=5.0)
    placements, cost, covered, info = agent.plan_placements()
    print('Agent recommended placements:', placements)
    print('Cost:', cost)
    print('Covered nodes:', covered)
    print('Info:', info)


def cmd_train_rl(args):
    # small RL training run using SimpleNetEnv and DQN
    if torch is None or gym is None:
        print('Torch and Gym required for RL training; missing.'); return
    # define network
    node_ids = ['N1','N2','N3','N4']
    initial_risk = {n: random.random()*0.2 for n in node_ids}
    node_cost = {n: 1.0 for n in node_ids}
    attack_prob = {n: 0.05 + (0.1 if i%2==0 else 0.02) for i,n in enumerate(node_ids)}
    budget = 2.0
    env = SimpleNetEnv(node_ids, initial_risk, node_cost, attack_prob, budget)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(obs_dim, n_actions)
    episodes = 200
    eps = 1.0
    eps_min = 0.05
    eps_decay = (eps - eps_min) / episodes
    for ep in range(episodes):
        obs = env.reset(initial_risk)
        total_reward = 0.0
        for t in range(50):
            act = agent.act(obs, eps)
            next_obs, rew, done, info = env.step(act)
            agent.push((obs, act, rew, next_obs, 0.0))
            loss = agent.update(batch_size=64)
            obs = next_obs
            total_reward += rew
        agent.sync_target()
        eps = max(eps_min, eps - eps_decay)
        if (ep+1) % 20 == 0:
            print(f'Episode {ep+1}/{episodes} total_reward={total_reward:.3f} eps={eps:.3f}')
    print('RL training completed.')

# -----------------------------
# Main
# -----------------------------
def main_entry():
    if len(sys.argv) < 2:
        print('Usage: python adaptive_deception_mesh_complete.py [preprocess|train_models|agent_example|train_rl] [args]')
        return
    cmd = sys.argv[1]
    args = sys.argv[2:]
    if cmd == 'preprocess':
        cmd_preprocess(args)
    elif cmd == 'train_models':
        cmd_train_models(args)
    elif cmd == 'agent_example':
        cmd_agent_example(args)
    elif cmd == 'train_rl':
        cmd_train_rl(args)
    else:
        print('Unknown command:', cmd)

if _name_ == '_main_':
    main_entry()