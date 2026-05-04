import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from config import (FEATURE_SIZE, GAMMA, LR, BATCH_SIZE, EPSILON_START, EPSILON_MIN,
                    EPSILON_DECAY, TAU, PER_ALPHA, PER_BETA_START, PER_BETA_STEPS, REPLAY_CAPACITY)


# ─── Sum Tree (O(log n) priority sampling) ────────────────────────────────────

class SumTree:
    """Binary sum tree backing the prioritized replay buffer."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        # Iterative propagation — avoids Python recursion limit on large buffers
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, idx, s):
        # Iterative traversal — consistent with _propagate, avoids recursion limit on large buffers
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ─── Prioritized Experience Replay Buffer ────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    PER buffer: samples experiences proportional to their TD error magnitude.
    Importance-sampling (IS) weights correct for the introduced bias.
    Beta is annealed from beta_start → 1.0 over training to reduce bias correction
    gradually as the value estimates improve.
    """

    def __init__(self, capacity, alpha=PER_ALPHA, beta_start=PER_BETA_START, beta_steps=PER_BETA_STEPS):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_steps
        self.epsilon = 1e-5       # Prevents zero priority
        self.max_priority = 1.0

    def add(self, experience):
        # New experiences start with maximum priority so they are trained on at least once
        self.tree.add(self.max_priority ** self.alpha, experience)

    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            if data is None:
                # Fallback: pick random valid entry
                valid = [d for d in self.tree.data if d is not None]
                data = random.choice(valid) if valid else None
            priorities.append(max(priority, self.epsilon))
            batch.append(data)
            idxs.append(idx)

        # IS weights: w_i = (N · P(i))^{-β} / max_j w_j
        total = self.tree.total
        probs = np.array(priorities, dtype=np.float64) / total
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)

        self.beta = min(1.0, self.beta + self.beta_increment)
        return batch, idxs, weights

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            priority = (abs(float(err)) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# ─── Q-Network ────────────────────────────────────────────────────────────────

class DQNetwork(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_size, 128), nn.ReLU(),
            nn.Linear(128, 128),          nn.ReLU(),
            nn.Linear(128, 64),           nn.ReLU(),
            nn.Linear(64, 1)  # Single Q-value per candidate neighbour
        )

    def forward(self, x):
        return self.net(x)


# ─── Agent ────────────────────────────────────────────────────────────────────

class NeuralRouterAgent:
    """
    Double DQN agent with Prioritized Experience Replay and soft target updates.

    Improvements over vanilla DQN:
      - Double DQN:  online net selects best next action; target net evaluates it
                     → reduces Q-value overestimation bias
      - PER:         samples high-TD-error experiences more frequently
                     → faster learning on rare / difficult transitions
      - Soft update: θ_target = τ·θ_online + (1-τ)·θ_target each step
                     → smoother convergence than hard periodic copies
      - Huber loss:  robust to the large −50 malicious-crash outlier reward
    """

    def __init__(self, feature_size=FEATURE_SIZE):
        self.feature_size = feature_size
        self.memory = PrioritizedReplayBuffer(REPLAY_CAPACITY)
        self.gamma         = GAMMA
        self.epsilon       = EPSILON_START
        self.epsilon_min   = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.lr            = LR
        self.batch_size    = BATCH_SIZE
        self.tau           = TAU

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model        = DQNetwork(feature_size).to(self.device)
        self.target_model = DQNetwork(feature_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Huber loss with per-element reduction for PER weight scaling
        self.loss_fn  = nn.SmoothL1Loss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.step_count = 0

    def remember(self, action_feature, reward, next_features, done):
        self.memory.add((action_feature, reward, next_features, done))

    def act(self, valid_actions, features):
        if not valid_actions:
            return None, None
        if random.random() < self.epsilon:
            idx = random.randint(0, len(valid_actions) - 1)
            return valid_actions[idx], features[idx]

        feat_tensor = torch.FloatTensor(np.array(features)).to(self.device)
        with torch.no_grad():
            q_values = self.model(feat_tensor).cpu().numpy().flatten()
        best_idx = int(np.argmax(q_values))
        return valid_actions[best_idx], features[best_idx]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch, idxs, weights = self.memory.sample(self.batch_size)
        weights_tensor = torch.FloatTensor(weights).to(self.device)

        state_feats, rewards, dones, next_q_values = [], [], [], []

        for item in batch:
            if item is None:
                continue
            state_feat, reward, next_feats, done = item
            state_feats.append(state_feat)
            rewards.append(reward)
            dones.append(done)

            if done or len(next_feats) == 0:
                next_q_values.append(0.0)
            else:
                with torch.no_grad():
                    nft = torch.FloatTensor(np.array(next_feats)).to(self.device)
                    # Double DQN: online net picks best action index
                    best_idx = int(self.model(nft).argmax().item())
                    # Target net evaluates that action's value
                    next_q_values.append(self.target_model(nft)[best_idx].item())

        if not state_feats:
            return 0.0

        S  = torch.FloatTensor(np.array(state_feats)).to(self.device)
        R  = torch.FloatTensor(rewards).to(self.device)
        D  = torch.FloatTensor(dones).to(self.device)
        NQ = torch.FloatTensor(next_q_values).to(self.device)

        current_q = self.model(S).squeeze(1)
        target_q  = R + self.gamma * NQ * (1 - D)

        # IS-weighted Huber loss
        per_loss = self.loss_fn(current_q, target_q.detach())
        w = weights_tensor[:len(per_loss)]
        loss = (w * per_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update PER priorities with raw TD errors
        self.memory.update_priorities(idxs[:len(per_loss)], per_loss.cpu().detach().numpy())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Soft target update
        for tp, op in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

        self.step_count += 1
        return loss.item()