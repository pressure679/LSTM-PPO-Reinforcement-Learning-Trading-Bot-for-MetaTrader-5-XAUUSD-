import pandas as pd
import numpy as np
import os
import pickle
from io import StringIO
import random
from collections import deque
from datetime import datetime, timedelta
# import requests
# import threading
# from multiprocessing import Process
# import time
# from time import timezone
# from decimal import Decimal
# from pybit.unified_trading import HTTP
# from pybit.unified_trading import WebSocket
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics.pairwise import cosine_similarity
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import subprocess
# import glob
# import shutil
import MetaTrader5 as mt5

# ready_event = threading.Event()
# process_counter = 0
# pd.set_option('future.no_silent_downcasting', True)

ACTIONS = ['hold', 'long', 'short', 'close']

# capital = 800

def load_last_mb_xauusd(file_path="C:\\Users\\Vittus Mikiassen\\Desktop\\XAU_15m_data.csv", mb=7, delimiter=';', col_names=None):
    file_size = os.path.getsize(file_path)
    offset = max(file_size - mb * 1024 * 1024, 0)  # start position
    
    with open(file_path, 'rb') as f:
        # Seek to approximately 20 MB before EOF
        f.seek(offset)
        
        # Read to the end of file from that offset
        data = f.read().decode(errors='ignore')
        
        # If not at start of file, discard partial first line (incomplete)
        if offset > 0:
            data = data.split('\n', 1)[-1]
        
    df = pd.read_csv(StringIO(data), delimiter=delimiter, header=None, engine='python')
    
    #if col_names:
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    
    # Convert columns if needed, e.g.:
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M", errors='coerce')
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()

    # df = df.resample('15min').agg({
    #     'Open': 'first',
    #     'High': 'max',
    #     'Low': 'min',
    #     'Close': 'last'
    # }).dropna()
    
    df = df.dropna()
    
    return df

def ADX(df, period=14):
    """
    Returns +DI, -DI and ADX using Wilder's smoothing.
    Columns required: High, Low, Close
    """
    high  = df['High']
    low   = df['Low']
    close = df['Close']

    # --- directional movement -----------------------------------------
    # plus_dm  = (high.diff()  > low.diff())  * (high.diff()).clip(lower=0)
    # minus_dm = (low.diff()   > high.diff()) * (low.diff().abs()).clip(lower=0)̈́
    up  =  high.diff()
    dn  = -low.diff()

    plus_dm_array  = np.where((up  >  dn) & (up  > 0),  up,  0.0)
    minus_dm_array = np.where((dn  >  up) & (dn  > 0),  dn,  0.0)

    plus_dm = pd.Series(plus_dm_array, index=df.index) # ← wrap
    minus_dm = pd.Series(minus_dm_array, index=df.index) # ← wrap

    # --- true range ----------------------------------------------------
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    # --- Wilder smoothing ---------------------------------------------
    atr       = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di   = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di  = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di

def STOCH(df, period=14, smooth_d=3):
    """
    Returns %K and %D stochastic oscillator.

    Columns required:
    High, Low, Close
    """

    high  = df['High']
    low   = df['Low']
    close = df['Close']

    # --- highest high / lowest low ------------------------------------
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()

    # --- %K ------------------------------------------------------------
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))

    # --- %D (smoothed %K) ---------------------------------------------
    d = k.rolling(window=smooth_d).mean()

    return k, d

def EMA(df, period):
    return df['Close'].ewm(span=period, adjust=False).mean()

def add_indicators(df):
    df['adx'], df['+di'], df['-di'] = ADX(df)

    df['k'], df['k_smooth'] = STOCH(df)

    df['EMA7'] = EMA(df, 7)
    df['EMA21'] = EMA(df, 21)
    df['EMA_DIFF'] = df['EMA7'] - df['EMA21']

    df = df[["Open", "High", "Low", "Close", "k", "k_smooth", "adx", "+di", "-di", "EMA7", "EMA21", "EMA_DIFF"]].copy()
    # df = df[["Open", "High", "Low", "Close", "EMA_crossover", "macd_zone", "macd_line", "macd_signal", "macd_line_diff", "macd_signal_diff", "macd_line_slope", "macd_signal_line_slope" , "macd_osma", "macd_crossover", "bb_sma", "bb_upper", "bb_lower", "RSI_zone", "ADX_zone", "+DI_val", "-DI_val", "ATR", "order_block_type"]].copy()

    df.dropna(inplace=True)
    # ready_event.set()
    return df

class LSTMPPOAgent:
    def __init__(self, state_size, hidden_size, action_size, lr=1e-3, gamma=0.95, clip_ratio=0.2):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_epochs = 5
        self.batch_size = 32
        self.entropy_coef = 0.01

        # Initialize weights
        self.model = {
            # LSTM
            'Wx': np.random.randn(4 * hidden_size, state_size) * 0.1,
            'Wh': np.random.randn(4 * hidden_size, hidden_size) * 0.1,
            'b': np.zeros(4 * hidden_size),

            # Policy head
            'W_policy': np.random.randn(action_size, hidden_size) * 0.1,
            'W_policy_2': np.random.randn(1, hidden_size) * 0.1,
            'b_policy': np.zeros(action_size),
            'b_policy_2': np.zeros(1),

            # Value head
            'W_value': np.random.randn(1, hidden_size) * 0.1,
            'b_value': np.zeros(1),
        }

        self.reset_state()
        self.trajectory = []

    def reset_state(self):
        self.h = np.zeros((self.hidden_size,))
        self.c = np.zeros((self.hidden_size,))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))  # prevent overflow

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def lstm_forward(self, x_seq):
        h, c = self.h.copy(), self.c.copy()
        for x in x_seq:
            x = np.asarray(x).reshape(-1)  # ensure shape (state_size,)
            assert x.shape[0] == self.state_size, f"x shape {x.shape} does not match state_size {self.state_size}"
            z = np.dot(self.model['Wx'], x) + np.dot(self.model['Wh'], h) + self.model['b']
            i = self.sigmoid(z[0:self.hidden_size])
            f = self.sigmoid(z[self.hidden_size:2*self.hidden_size])
            o = self.sigmoid(z[2*self.hidden_size:3*self.hidden_size])
            g = self.tanh(z[3*self.hidden_size:])
            c = f * c + i * g
            h = o * self.tanh(c)
        self.h, self.c = h, c
        return h

    def forward(self, x_seq):
        h = self.lstm_forward(x_seq)
        logits = np.dot(self.model['W_policy'], h) + self.model['b_policy']
        value = np.dot(self.model['W_value'], h) + self.model['b_value']
        probs = self.softmax(logits)
        return probs, value[0]

    def sl_tp_forward(self, x_seq):
        h = self.lstm_forward(x_seq)
        # logits = np.dot(self.model['W_policy'], h) + self.model['b_policy']
        logits = np.dot(self.model['W_policy_2'], h) + self.model['b_policy_2']
        # probs = self.softmax(logits)
        value = np.dot(self.model['W_value'], h) + self.model['b_value']
        continuous_action = self.sigmoid(logits)
        # return probs, value[0]
        return continuous_action, value[0] # (tp_raw, sl_raw)

    def scale_action(self, tp_raw, sl_raw):
        tp_min, tp_max = 0.0002, 0.002  # 0.5% to 5%, 0.2% to 2% with 50x leverage
        sl_min, sl_max = 0.0001, 0.001 # 0.2% to 3%, 
        tp_pct = tp_min + (tp_max - tp_min) * tp_raw
        sl_pct = sl_min + (sl_max - sl_min) * sl_raw
        return tp_pct, sl_pct

    def select_action(self, state_seq, in_position):
        valid_actions = []
        try:
            logits, value = self.forward(state_seq)

            # Convert logits to probabilities using softmax
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)

            # Handle invalid probabilities
            if np.any(np.isnan(probs)) or np.sum(probs) == 0:
                # print("⚠️ Warning: NaN or zero-sum probabilities. Using uniform distribution.")
                probs = np.ones(self.action_size) / self.action_size

            probs = probs / np.sum(probs)  # Normalize again just in case

            if np.argmax(probs) >= 0.9:
                action = np.argmax(probs)
                logprob = np.log(probs[action] + 1e-8)
            else:
                action = np.random.choice(self.action_size, p=probs)
                logprob = np.log(probs[action] + 1e-8)

            # if in_position:
            #     action = 0
            if in_position:
                valid_actions = [0, 1]  # Hold or Close
            else:
                valid_actions = [0, 2, 3]  # Hold, Long, Short
                
            # Mask invalid actions:
            masked_probs = np.array([probs[a] if a in valid_actions else 0 for a in range(self.action_size)])
            if masked_probs.sum() == 0:
                masked_probs = np.ones(self.action_size) / self.action_size
            else:
                masked_probs = masked_probs / masked_probs.sum()
                    
            if np.argmax(masked_probs) >= 0.9:
                action = np.argmax(masked_probs)
            else:
                action = np.random.choice(self.action_size, p=masked_probs)
            logprob = np.log(masked_probs[action] + 1e-8)

            # if last_ema_crossover == current_ema_crossover:
            #     action = 0

            return action, logprob, value
        except Exception as e:
            # print(f"❌ select_action error: {e}")
            return None

    def store_transition(self, state_seq, action, logprob, value, reward, done):
        self.trajectory.append((state_seq, action, logprob, value, reward, done))

    def store_reward(self, reward):
        if self.trajectory:
            last = self.trajectory[-1]
            self.trajectory[-1] = (*last, reward)

    def _discount_rewards(self, rewards):
        discounted = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted.insert(0, R)
        return discounted

    def update_policy_and_value(self, states_seq, actions, old_action_probs, advantages, returns, lr=1e-3, epsilon=0.1):
        for i in range(len(states_seq)):
            state_seq = states_seq[i]
            action = actions[i]
            old_prob = old_action_probs[i]
            advantage = advantages[i]
            target_value = returns[i]

            # Initialize LSTM hidden state
            h = np.zeros((self.hidden_size, 1))

            # Forward pass through LSTM for the state sequence
            for x in state_seq:
                x = np.array(x, dtype=np.float32).reshape(self.state_size, 1)
                z = np.dot(self.model['Wx'], x) + np.dot(self.model['Wh'], h) + self.model['b'].reshape(-1, 1)
                # LSTM gates split
                i_gate = self.sigmoid(z[0:self.hidden_size])
                f_gate = self.sigmoid(z[self.hidden_size:2*self.hidden_size])
                o_gate = self.sigmoid(z[2*self.hidden_size:3*self.hidden_size])
                g_gate = np.tanh(z[3*self.hidden_size:4*self.hidden_size])

                # LSTM cell state update (you might want to maintain c state if you have it, here simplified)
                c = f_gate * np.zeros_like(h) + i_gate * g_gate  # Assume c initialized as zeros for simplicity
                h = o_gate * np.tanh(c)

            # Policy logits and probabilities
            logits = np.dot(self.model['W_policy'], h) + self.model['b_policy'].reshape(-1, 1)
            probs = self.softmax(logits.flatten())
            new_prob = probs[action]

            # Value prediction
            value = (np.dot(self.model['W_value'], h) + self.model['b_value']).item()

            # Value loss gradient
            v_error = value - target_value
            grad_W_value = v_error * h.T
            grad_b_value = v_error

            # PPO policy loss gradient
            ratio = new_prob / (old_prob + 1e-10)
            clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
            policy_grad_coef = -min(ratio * advantage, clipped_ratio * advantage)

            grad_logits = probs.copy()
            grad_logits[action] -= 1  # dSoftmax cross-entropy grad
            grad_logits *= policy_grad_coef

            grad_W_policy = np.dot(grad_logits.reshape(-1, 1), h.T)
            grad_b_policy = grad_logits.reshape(-1, 1)

            # Gradient descent update
            self.model['W_policy'] -= lr * grad_W_policy
            self.model['b_policy'] -= lr * grad_b_policy.flatten()
            self.model['W_value'] -= lr * grad_W_value
            self.model['b_value'] -= lr * grad_b_value

    def ppo_policy_loss(self, old_probs, new_probs, advantages, epsilon=0.1):
        old_probs = np.array(old_probs)
        new_probs = np.array(new_probs)
        advantages = np.array(advantages)
        
        ratios = new_probs / (old_probs + 1e-10)
        clipped = np.clip(ratios, 1 - epsilon, 1 + epsilon)
        loss = -np.mean(np.minimum(ratios * advantages, clipped * advantages))
        return loss
    def value_loss(self, values, returns):
        values = np.array(values)
        returns = np.array(returns)
        return 0.5 * np.mean((returns - values) ** 2)

    def compute_gae(self, rewards, values, dones, gamma=0.95, lam=0.95):
        advantages = []
        gae = 0
        values = np.append(values, 0)  # Bootstrap value after last step
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return np.array(advantages)

    def train(self):
        # if len(self.trajectory) < 2:
        #     return  # Not enough data to train
        if self.trajectory:
            # Unpack trajectory
            states, actions, logprobs_old, values, rewards, dones = zip(*self.trajectory)

            # Convert to arrays
            values = np.array(values)
            rewards = np.array(rewards)

            # Normalize rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Compute GAE
            advantages = self.compute_gae(rewards, values, dones, gamma=0.95, lam=0.95)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages
            returns = advantages + values

            # Clip returns and values
            returns = np.clip(returns, -1000, 1000)
            values = np.clip(values, -1000, 1000)
            advantages = np.clip(advantages, -10, 10)

            # Train for multiple epochs (optional)
            for _ in range(self.train_epochs):
                for i in range(len(states)):
                    state_seq = states[i]
                    action = actions[i]
                    old_logprob = logprobs_old[i]
                    # print(f"len(states): {len(states)}")
                    # print(f"len(actions): {len(actions)}")
                    # print(f"len(logprobs_old): {len(logprobs_old)}")
                    # print(f"len(values): {len(values)}")
                    # print(f"len(rewards): {len(rewards)}")
                    # print(f"len(dones): {len(dones)}")
                    # print(f"length of advantages: {len(advantages)}, i: {i}, length of states: {len(states)}")
                    advantage = advantages[i]
                    target_value = returns[i]

                    # Forward pass
                    probs, value = self.forward(state_seq)

                    # Entropy bonus
                    entropy = -np.sum(probs * np.log(probs + 1e-8))

                    # Policy loss
                    logprob = np.log(probs[action] + 1e-8)
                    ratio = np.exp(logprob - old_logprob)
                    clipped_ratio = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    policy_loss = -min(ratio * advantage, clipped_ratio * advantage)

                    # Value loss
                    v_loss = 0.5 * ((target_value - value) ** 2)

                    # Total loss
                    loss = policy_loss + 0.5 * v_loss - 0.01 * entropy

                    # Gradient descent step (simplified)
                    for k in self.model:
                        self.model[k] -= self.lr * loss
                        self.model[k] = np.clip(self.model[k], -1000, 1000)  # Clamp weights

                self.trajectory.clear()

    def savecheckpoint(self, symbol):
        os.makedirs("LSTM-PPO-saves", exist_ok=True)
        filename = f"LSTM-PPO-saves/{datetime.now().strftime('%Y-%m-%d')}-{symbol}.checkpoint.lstm-ppo.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def loadcheckpoint(self, symbol):
        files = sorted(os.listdir("LSTM-PPO-saves"))
        files = [f for f in files if f.endswith(".checkpoint.lstm-ppo.pkl") and symbol in f]
        if not files:
            print(f"[!] No checkpoint found for {symbol}")
            return

        latest = os.path.join("LSTM-PPO-saves", files[-1])
        with open(latest, "rb") as f:
            self.model = pickle.load(f)

class WinRateKNN:
    def __init__(self, symbol, k=10):
        self.k = k
        self.symbol = symbol
        self.states = []
        self.labels = []  # 1 = win, 0 = loss
        self.model = None

    def add(self, state, is_win):
        try:
            state = np.array(state, dtype=np.float32).flatten()  # Force all elements to float
        except Exception as e:
            # print("❌ Could not convert state to float:", state, "| Error:", e)
            return

        if not np.all(np.isfinite(state)):
            # print("⚠️ Skipping state with NaN or Inf:", state)
            return

        self.states.append(state)
        self.labels.append(1 if is_win else 0)

        if len(self.states) >= 100:
            self._remove_redundant_neighbor()
            # self.states.pop(0)
            # self.labels.pop(0)

        if len(self.states) >= self.k:
            self._fit()

    def _remove_redundant_neighbor(self):
        if len(self.states) < 2:
            return  # Nothing to remove

        X = np.array(self.states)

        # Compute pairwise similarity (cosine, or use euclidean if you prefer)
        sim_matrix = cosine_similarity(X)

        # Zero out diagonal (self-similarity)
        np.fill_diagonal(sim_matrix, 0)

        # Compute average similarity for each row (how redundant each entry is)
        redundancy_scores = sim_matrix.mean(axis=1)

        # Remove the most redundant (highest avg similarity)
        idx_to_remove = np.argmax(redundancy_scores)

        del self.states[idx_to_remove]
        del self.labels[idx_to_remove]
    def _fit(self):
        """
        Fit the KNN model with stored data.
        """
        if len(self.states) < 1:
            # print("⚠️ Not enough data to fit KNN.")
            return

        # Safety check
        k_neighbors = max(1, min(self.k, len(self.states)))

        self.model = NearestNeighbors(n_neighbors=k_neighbors, algorithm="kd_tree")
        self.model.fit(self.states)

    def predict_win_rate(self, state_seq, k_near=5, k_far=5):
        """
        Return the win rate based on k nearest neighbors of the input state.
        """
        # if not self.model or len(self.states) < self.k:
        if len(self.states) < 1000:
            # return True  # Not enough data
            return 1  # Not enough data

        # Find the 100 nearest neighbors
        distances, indices = self.model.kneighbors(state.reshape(1, -1), n_neighbors=50)
        distances = distances[0]
        indices = indices[0]

        # Split into nearest and farthest groups
        nearest_idx = indices[:k_near]
        nearest_dist = distances[:k_near]

        farthest_idx = indices[-k_far:]
        farthest_dist = distances[-k_far:]

        # Combine indices and distances
        combined_idx = np.concatenate([nearest_idx, farthest_idx])
        combined_dist = np.concatenate([nearest_dist, farthest_dist])

        # Get win/loss labels for selected neighbors
        selected_labels = np.array([self.labels[i] for i in combined_idx])

        # Calculate weights (closer gets higher weight)
        weights = 1 / (combined_dist + 1e-6)  # Add epsilon to avoid div-by-zero

        # Normalize weights
        weights /= weights.sum()

        # Compute weighted win rate
        win_rate = np.dot(selected_labels, weights)

        return win_rate
    def save(self):
        """
        Save the KNN model to disk.
        """
        path = f"LSTM-PPO-saves/{datetime.now().strftime('%Y-%m-%d')}-{self.symbol}.win_rate_knn.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "states": self.states,
                "labels": self.labels,
                "model": self.model
            }, f)

    def load(self):
        """
        Load the KNN model from disk.
        """
        # path = f"LSTM-PPO-saves/win_rate_knn-{self.symbol}.pkl"
        files = sorted(os.listdir("LSTM-PPO-saves"))
        files = [f for f in files if f.endswith(".win_rate_knn.pkl") and self.symbol in f]
        if not files:
            print(f"[!] No checkpoint found for {self.symbol}")
            return

        latest = os.path.join("LSTM-PPO-saves", files[-1])

        try:
            with open(latest, "rb") as f:
                data = pickle.load(f)
                self.states = data["states"]
                self.labels = data["labels"]
                self.model = data["model"]
                # print(f"✅ Loaded WinRateKNN from {latest}")
        except FileNotFoundError:
            print(f"⚠️ No saved KNN found at {path}. Starting fresh.")
    
def sharpe_ratio(returns, risk_free_rate=0.0):
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0
    return (mean_ret - risk_free_rate) / std_ret

def sortino_ratio(returns, risk_free_rate=0.0):
    mean_ret = np.mean(returns)
    # Downside deviation: only consider returns below risk-free rate, and their square differences
    downside_diff = [(r - risk_free_rate)**2 for r in returns if r < risk_free_rate]
    
    if len(downside_diff) == 0:
        return 0  # Or float('inf') if you'd rather signal perfect performance
    
    downside_std = np.sqrt(np.mean(downside_diff))
    
    if downside_std == 0:
        return 0
    
    return (mean_ret - risk_free_rate) / downside_std

def calc_lot_size(risk_amount, stop_loss_pips=300, pip_value_per_lot=0.01):
    if stop_loss_pips <= 0:
        raise ValueError("stop_loss_pips must be positive")
    if pip_value_per_lot <= 0:
        raise ValueError("pip_value_per_lot must be positive")
    if risk_amount <= 0:
        return 0.0

    lots = (risk_amount / (stop_loss_pips * pip_value_per_lot))
    # print(f"lots: {round(lots, 0)} with risk amount ${risk_amount}")
    lot_size = round(lots, 0) * 0.01
    # print(f"lot size: {lot_size} with risk amount ${risk_amount}")
    return lot_size

def max_drawdown(returns):

    if len(returns) == 0:
        return 0

    equity = np.cumsum(returns)

    peak = equity[0]
    max_dd = 0

    for value in equity:

        peak = max(peak, value)

        dd = peak - value

        max_dd = max(max_dd, dd)

    return max_dd

def train_bot(df, symbol="XAUUSD"):

    SEQ_LEN = 32

    FEATURES = [
        "Open",
        "High",
        "Low",
        "Close",
        "k",
        "k_smooth",
        "adx",
        "+di",
        "-di",
        "EMA7",
        "EMA21",
        "EMA_DIFF"
    ]

    agent = LSTMPPOAgent(
        state_size=len(FEATURES),
        hidden_size=64,
        action_size=4
    )

    # knn = WinRateKNN(symbol)

    try:
        agent.loadcheckpoint(symbol)
        # knn.load()
        print(f"[{symbol}] Loaded checkpoint")
    except:
        print(f"[{symbol}] Starting fresh")

    save_counter = 0

    in_position = False
    position_type = None

    # entry_price = 0
    # sl_price = 0
    # tp_price = 0

    entry_price = 0
    sl_price = 0

    tp1_price = 0
    tp2_price = 0
    tp3_price = 0
    tp4_price = 0

    position_size = 0.0
    realized_reward = 0.0

    tp1_hit = False
    tp2_hit = False
    tp3_hit = False
    tp4_hit = False

    tp1_sl_moved = False
    tp2_sl_moved = False
    tp3_sl_moved = False

    trade_returns = []

    # STANDARD_SL_PIPS = 100
    # RR_RATIO = 2.0
    # SPREAD_AND_COMMISSION = 1.2

    # SL_PIPS = 50

    # TP1_PIPS = 50
    # TP2_PIPS = 100
    # TP3_PIPS = 150
    # TP4_PIPS = 200

    PIP_VALUE = 0.1

    SPREAD_AND_COMMISSION = 1.2


    state_buffer = deque(maxlen=SEQ_LEN)

    # preload sequence
    for i in range(SEQ_LEN):
        row = df.iloc[i][FEATURES].values.astype(np.float32)
        state_buffer.append(row)

    for i in range(SEQ_LEN, len(df)):

        current = df.iloc[i]

        current_price = current["Close"]
        high = current["High"]
        low = current["Low"]
        SL_PIPS = round(current_price * 0.00125 * 10, 0)
        # SL_PIPS = 50
        TP1_PIPS = SL_PIPS
        TP2_PIPS = round(SL_PIPS * 2, 0)
        TP3_PIPS = round(SL_PIPS * 3, 0)
        TP4_PIPS = round(SL_PIPS * 4, 0)
        SL_MOVE_BUFFER = round(SL_PIPS / 3, 0)

        state = current[FEATURES].values.astype(np.float32)

        state_buffer.append(state)

        if len(state_buffer) < SEQ_LEN:
            continue

        state_seq = np.array(state_buffer)

        # === Select action ============================================
        result = agent.select_action(state_seq, in_position)

        if result is None:
            continue

        action, logprob, value = result

        reward = 0
        done = False

        # ==============================================================
        # OPEN LONG
        # ==============================================================

        if action == 2 and not in_position:

            in_position = True
            position_type = "long"

            # entry_price = current_price

            # sl_price = entry_price - (STANDARD_SL_PIPS * 0.1)
            # tp_price = entry_price + (
            #     STANDARD_SL_PIPS * RR_RATIO * 0.1
            # )

            entry_price = current_price

            sl_price = entry_price - (SL_PIPS * PIP_VALUE)

            tp1_price = entry_price + (TP1_PIPS * PIP_VALUE)
            tp2_price = entry_price + (TP2_PIPS * PIP_VALUE)
            tp3_price = entry_price + (TP3_PIPS * PIP_VALUE)
            tp4_price = entry_price + (TP4_PIPS * PIP_VALUE)

            position_size = 1.0
            realized_reward = 0.0

            tp1_hit = False
            tp2_hit = False
            tp3_hit = False
            tp4_hit = False

            tp1_sl_moved = False
            tp2_sl_moved = False
            tp3_sl_moved = False

            # print('opened long')

        # ==============================================================
        # OPEN SHORT
        # ==============================================================

        elif action == 3 and not in_position:

            in_position = True
            position_type = "short"

            # entry_price = current_price

            # sl_price = entry_price + (STANDARD_SL_PIPS * 0.1)
            # tp_price = entry_price - (
            #     STANDARD_SL_PIPS * RR_RATIO * 0.1
            # )

            entry_price = current_price

            sl_price = entry_price + (SL_PIPS * PIP_VALUE)

            tp1_price = entry_price - (TP1_PIPS * PIP_VALUE)
            tp2_price = entry_price - (TP2_PIPS * PIP_VALUE)
            tp3_price = entry_price - (TP3_PIPS * PIP_VALUE)
            tp4_price = entry_price - (TP4_PIPS * PIP_VALUE)

            position_size = 1.0
            realized_reward = 0.0

            tp1_hit = False
            tp2_hit = False
            tp3_hit = False
            tp4_hit = False

            tp1_sl_moved = False
            tp2_sl_moved = False
            tp3_sl_moved = False

            # print('opened short')

        # ==============================================================
        # MANAGE POSITION
        # ==============================================================
        if in_position:

            trade_closed = False

            # ==================================================
            # LONG
            # ==================================================

            if position_type == "long":

                if not tp1_hit and high >= tp1_price:

                    realized_reward += SL_PIPS
                    position_size -= 0.25

                    tp1_hit = True

                    # sl_price = entry_price

                if not tp2_hit and high >= tp2_price:

                    realized_reward += SL_PIPS * 2
                    position_size -= 0.25

                    tp2_hit = True

                    # sl_price = tp1_price

                if not tp3_hit and high >= tp3_price:

                    realized_reward += SL_PIPS * 3
                    position_size -= 0.25

                    tp3_hit = True

                    # sl_price = tp2_price

                if not tp4_hit and high >= tp4_price:

                    realized_reward += SL_PIPS * 4

                    reward = realized_reward

                    trade_closed = True

                if not trade_closed and low <= sl_price:

                    remaining_pips = (
                        (sl_price - entry_price)
                        / PIP_VALUE
                    )

                    realized_reward += (
                        remaining_pips * position_size
                    )

                    reward = realized_reward

                    trade_closed = True

                # TP1 reached +10 pips
                if tp1_hit and not tp1_sl_moved:

                    if high >= tp1_price + (SL_MOVE_BUFFER / PIP_VALUE):

                        sl_price = tp1_price

                        tp1_sl_moved = True

                # TP2 reached +10 pips
                if tp2_hit and not tp1_sl_moved:

                    if high >= tp2_price + (SL_MOVE_BUFFER / PIP_VALUE):

                        sl_price = tp2_price

                        tp2_sl_moved = True
                
                # TP1 reached +10 pips
                if tp3_hit and not tp3_sl_moved:

                    if high >= tp3_price + (SL_MOVE_BUFFER / PIP_VALUE):

                        sl_price = tp3_price

                        tp3_sl_moved = True

            # ==================================================
            # SHORT
            # ==================================================

            elif position_type == "short":

                if not tp1_hit and low <= tp1_price:

                    realized_reward += SL_PIPS
                    position_size -= 0.25

                    tp1_hit = True

                    # sl_price = entry_price

                if not tp2_hit and low <= tp2_price:

                    realized_reward += SL_PIPS * 2
                    position_size -= 0.25

                    tp2_hit = True

                    # sl_price = tp1_price

                if not tp3_hit and low <= tp3_price:

                    realized_reward += SL_PIPS * 3
                    position_size -= 0.25

                    tp3_hit = True

                    # sl_price = tp2_price

                if not tp4_hit and low <= tp4_price:

                    realized_reward += SL_PIPS * 4

                    reward = realized_reward

                    trade_closed = True

                if not trade_closed and high >= sl_price:

                    remaining_pips = (
                        (entry_price - sl_price)
                        / PIP_VALUE
                    )

                    realized_reward += (
                        remaining_pips * position_size
                    )

                    reward = realized_reward

                    trade_closed = True

                # TP1 reached +10 pips
                if tp1_hit and not tp1_sl_moved:

                    if low <= tp1_price + (SL_MOVE_BUFFER / PIP_VALUE):

                        sl_price = tp1_price

                        tp1_sl_moved = True

                # TP2 reached +10 pips
                if tp2_hit and not tp1_sl_moved:

                    if low <= tp2_price + (SL_MOVE_BUFFER / PIP_VALUE):

                        sl_price = tp2_price

                        tp2_sl_moved = True
                
                # TP1 reached +10 pips
                if tp3_hit and not tp3_sl_moved:

                    if low <= tp3_price + (SL_MOVE_BUFFER / PIP_VALUE):

                        sl_price = tp3_price

                        tp3_sl_moved = True

            if trade_closed:

                in_position = False

                trade_returns.append(reward)

        # ==============================================================
        # STORE PPO TRANSITION
        # ==============================================================

        agent.store_transition(
            state_seq,
            action,
            logprob,
            value,
            reward,
            done
        )

        save_counter += 1

        # ==============================================================
        # WEEKLY TRAINING
        # ==============================================================

        if save_counter % 672 == 0:

            print(
                f"[{symbol}] "
                f"[INFO] Training PPO on step "
                f"{save_counter}..."
            )

            agent.train()
            agent.savecheckpoint(symbol)
            # knn._fit()
            # knn.save()

            # ==========================================================
            # WEEKLY STATS
            # ==========================================================

            if len(trade_returns) > 5:

                wins = [r for r in trade_returns if r > 0]
                losses = [r for r in trade_returns if r < 0]

                weekly_pnl = np.sum(trade_returns)

                winrate = (
                    len(wins) / len(trade_returns)
                    if len(trade_returns) > 0 else 0
                )

                mean_win = (
                    np.mean(wins)
                    if len(wins) > 0 else 0
                )

                mean_loss = (
                    np.mean(losses)
                    if len(losses) > 0 else 0
                )

                sharpe = sharpe_ratio(trade_returns)
                sortino = sortino_ratio(trade_returns)

                gross_profit = sum(wins)
                gross_loss = abs(sum(losses))
                profit_factor = (
                    gross_profit / gross_loss
                    if gross_loss > 0
                    else float("inf")
                )
                max_dd = max_drawdown(trade_returns)
                R_pnl = weekly_pnl / 4 / SL_PIPS

                print()
                print("================================================")
                print(f"[{symbol}] WEEKLY PPO TRAINING")
                print("================================================")
                print(f"Trades:          {len(trade_returns)}")
                print(f"Weekly PnL:      {weekly_pnl/4:.2f} pips")
                print(f"Winrate:         {winrate*100:.2f}%")
                print(f"Mean Win:        {mean_win*4:.2f} pips")
                print(f"Mean Loss:       {mean_loss*4:.2f} pips")
                print(f"Max DD:          {max_dd/(SL_PIPS*4):.2f}R")
                print(f"PF:              {profit_factor:.2f}")
                print(f"Weekly R profit: {R_pnl:.2f}")
                print(f"Sharpe:          {sharpe:.2f}")
                print(f"Sortino:         {sortino:.2f}")
                print("================================================")
                print()

                trade_returns = []

    # ==============================================================
    # FINAL TRAINING
    # ==============================================================

    agent.train()

    agent.savecheckpoint(symbol)

    # knn._fit()

    # knn.save()

    print(f"[{symbol}] Training complete.")

    return agent

def open_long(symbol, lot_size):

    tick = mt5.symbol_info_tick(symbol)

    entry = tick.ask

    sl = entry - 5

    tp1 = entry + 5
    tp2 = entry + 10
    tp3 = entry + 15
    tp4 = entry + 20

    tps = [tp1, tp2, tp3, tp4]

    for tp in tps:

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": entry,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "bot trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)

        print(result)

def open_short(symbol, lot_size):

    tick = mt5.symbol_info_tick(symbol)

    entry = tick.bid

    sl = entry + 5

    tp1 = entry - 5
    tp2 = entry - 10
    tp3 = entry - 15
    tp4 = entry - 20

    tps = [tp1, tp2, tp3, tp4]

    for tp in tps:

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_SELL,
            "price": entry,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "bot trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)

        print(result)

def open_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    positions = [
        p
        for p in positions
        if p.magic == 123456
    ]
    return len(positions)

def get_ppo_positions(symbol):

    positions = mt5.positions_get(symbol=symbol)

    return [
        p
        for p in positions
        if p.magic == 123456
    ]

def move_all_stops(symbol, new_sl):

    positions = get_ppo_positions(symbol)

    for pos in positions:

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "sl": new_sl,
            "tp": pos.tp
        }

        result = mt5.order_send(request)

        print(
            f"SL moved ticket "
            f"{pos.ticket} -> "
            f"{new_sl}"
        )

def manage_positions(symbol, SL_MOVE_BUFFER):

    positions = get_ppo_positions(symbol)

    count = len(positions)

    if count <= 1:
        return

    positions.sort(key=lambda p: p.tp)

    entry = positions[0].price_open

    tick = mt5.symbol_info_tick(symbol)

    direction = positions[0].type

    if direction == mt5.ORDER_TYPE_BUY:

        current_price = tick.bid

        tp1 = positions[0].tp

        if count == 3 and current_price >= tp1 + SL_MOVE_BUFFER:

            move_all_stops(
                symbol,
                entry
            )

        elif count == 2:

            tp2 = positions[1].tp

            if current_price >= tp2 + SL_MOVE_BUFFER:

                move_all_stops(
                    symbol,
                    tp1
                )

        elif count == 1:

            tp3 = positions[0].tp

            if current_price >= tp3 + SL_MOVE_BUFFER:

                move_all_stops(
                    symbol,
                    tp2
                )

    else:

        current_price = tick.ask

        positions.sort(
            key=lambda p: p.tp,
            reverse=True
        )

        tp1 = positions[0].tp

        if count == 3 and current_price <= tp1 - SL_MOVE_BUFFER:

            move_all_stops(
                symbol,
                entry
            )

        elif count == 2:

            tp2 = positions[1].tp

            if current_price <= tp2 - SL_MOVE_BUFFER:

                move_all_stops(
                    symbol,
                    tp1
                )

        elif count == 1:

            tp3 = positions[0].tp

            if current_price <= tp3 - SL_MOVE_BUFFER:

                move_all_stops(
                    symbol,
                    tp2
                )

def test_bot(symbol="XAUUSD"):
    SEQ_LEN = 32
    account = mt5.account_info()

    balance = account.balance
    RISK = 0.0025
    # risk_per_position = max(balance * RISK / 500 / 4, 0.01)

    tick = mt5.symbol_info_tick(symbol)
    SL_PIPS = tick.bid * 0.00125 * 10
    risk_per_position = min(
        max(round(balance * RISK / SL_PIPS * 10 / 4, 2), 0.01),
        100.0
    )

    FEATURES = [
        "Open",
        "High",
        "Low",
        "Close",
        "k",
        "k_smooth",
        "adx",
        "+di",
        "-di",
        "EMA7",
        "EMA21",
        "EMA_DIFF"
    ]

    last_m15 = None
    last_m1 = None

    agent = LSTMPPOAgent(
        state_size=len(FEATURES),
        hidden_size=64,
        action_size=4
    )

    agent.loadcheckpoint("XAUUSD")

    # ==========================================================
    # INITIAL LOAD
    # ==========================================================

    rates_m15 = mt5.copy_rates_from_pos(
        symbol,
        mt5.TIMEFRAME_M15,
        0,
        200
    )

    rates_m1 = mt5.copy_rates_from_pos(
        symbol,
        mt5.TIMEFRAME_M15,
        0,
        10
    )

    last_m1 = rates_m1[-1]["time"]
    last_m15 = df.iloc[-1]["time"]

    df = pd.DataFrame(rates_m15)

    df = add_indicators(df)

    last_m15 = None
    last_m1 = None

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    while True:

        # ======================================================
        # MANAGE POSITIONS EVERY NEW M1 CANDLE
        # ======================================================

        # rates_m1 = mt5.copy_rates_from_pos(
        #     symbol,
        #     mt5.TIMEFRAME_M1,
        #     0,
        #     2
        # )

        # current_m1 = rates_m1[-1]["time"]

        # if current_m1 != last_m1:

           #  last_m1 = current_m1

        manage_positions(symbol, round(SL_PIPS / 10 / 3, 2))

        # ======================================================
        # CHECK FOR NEW M15 CANDLE
        # ======================================================

        new_m15 = mt5.copy_rates_from_pos(
            symbol,
            mt5.TIMEFRAME_M15,
            0,
            1
        )

        current_m15 = new_m15[0]["time"]

        if current_m15 != last_m15:

            last_m15 = current_m15

            # ==================================================
            # APPEND NEW CANDLE
            # ==================================================

            new_row = pd.DataFrame(new_m15)

            if new_row.iloc[0]["time"] != df.iloc[-1]["time"]:

                df = pd.concat(
                    [df, new_row],
                    ignore_index=True
                )

                df = (
                    df.tail(200)
                    .reset_index(drop=True)
                )

                df = add_indicators(df)

            # ==================================================
            # BUILD STATE SEQUENCE
            # ==================================================

            state_seq = (
                df[FEATURES]
                .tail(SEQ_LEN)
                .values
                .astype(np.float32)
            )

            # ==================================================
            # POSITION CHECK
            # ==================================================

            open_pos = open_positions(symbol)

            # ==================================================
            # PPO DECISION
            # ==================================================

            action, _, _ = agent.select_action(
                state_seq,
                open_pos > 0
            )

            # ==================================================
            # OPEN NEW TRADE
            # ==================================================

            if open_pos == 0:

                account = mt5.account_info()

                balance = account.balance

                risk_per_position = max(
                    balance * RISK / 500 / 4,
                    0.01
                )

                if action == 2:

                    # print(
                    #     f"[{symbol}] PPO BUY"
                    # )

                    open_long(
                        symbol,
                        risk_per_position
                    )

                elif action == 3:

                    # print(
                    #     f"[{symbol}] PPO SELL"
                    # )

                    open_short(
                        symbol,
                        risk_per_position
                    )

                else:

                    print(
                        f"[{symbol}] PPO HOLD"
                    )

        now = datetime.now()
        sleep_seconds = 60 - now.second - now.microsecond / 1_000_000
        time.sleep(sleep_seconds)

def main():

    df = load_last_mb_xauusd()

    # print(f'loaded df: {df}')

    df = add_indicators(df)

    train_bot(df, "XAUUSD")

main()
