"""
Standalone training script for the Neural Router DQN Agent.

Usage:
    python train.py                   # uses defaults from config.py
    python train.py --episodes 3000   # custom episode count

Outputs:
    router_model.pth      — saved model weights
    training_curves.png   — reward, success-rate, and epsilon-decay plots
"""

import os
import random
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')   # Non-interactive backend (works without a display)
import matplotlib.pyplot as plt

from config import (NUM_NODES, MALICIOUS_RATIO, FEATURE_SIZE,
                    TRAIN_EPISODES, MODEL_PATH)
from network_env import NetworkEnvironment
from neural_router import NeuralRouterAgent

ROLLING_WINDOW = 50
PLOT_PATH = "training_curves.png"


def run_training(episodes: int):
    print("=" * 58)
    print("  Neural Router — DQN Training")
    print(f"  Episodes  : {episodes}")
    print(f"  Nodes     : {NUM_NODES}  |  Malicious ratio: {MALICIOUS_RATIO:.0%}")
    print(f"  Device    : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 58)

    env   = NetworkEnvironment(num_nodes=NUM_NODES, malicious_ratio=MALICIOUS_RATIO, randomize_graph=True)
    agent = NeuralRouterAgent(feature_size=FEATURE_SIZE)

    episode_rewards  = []
    episode_successes = []   # 1 = success, 0 = anything else
    episode_losses   = []

    for ep in range(episodes):
        valid_actions, features = env.reset(episode=ep)
        done = False
        total_reward = 0.0
        ep_losses = []
        info = {"reason": "step"}

        while not done:
            action, action_feature = agent.act(valid_actions, features)
            if action is None:
                break
            (next_actions, next_features), reward, done, info = env.step(action)
            agent.remember(action_feature, reward, next_features, done)
            loss = agent.replay()
            if loss:
                ep_losses.append(loss)
            valid_actions, features = next_actions, next_features
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_successes.append(1 if info["reason"] == "success" else 0)
        episode_losses.append(np.mean(ep_losses) if ep_losses else 0.0)

        if (ep + 1) % 100 == 0:
            acc = np.mean(episode_successes[-100:]) * 100
            rwd = np.mean(episode_rewards[-100:])
            print(f"  Ep {ep+1:>5}/{episodes}  |  Acc(100): {acc:5.1f}%  "
                  f"|  AvgReward: {rwd:7.2f}  |  ε: {agent.epsilon:.3f}")

    torch.save(agent.model.state_dict(), MODEL_PATH)
    print(f"\n  ✅ Model saved → {MODEL_PATH}")

    _plot_curves(episode_rewards, episode_successes, episode_losses, episodes)
    print(f"  📈 Training curves saved → {PLOT_PATH}")


def _plot_curves(rewards, successes, losses, total_eps):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle('Neural Router — DQN Training Curves', fontsize=14, fontweight='bold', y=0.98)

    eps_range = range(1, total_eps + 1)
    rolling = lambda arr, w: np.convolve(arr, np.ones(w) / w, mode='valid')

    # ── Reward ────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(eps_range, rewards, alpha=0.2, color='steelblue', linewidth=0.7, label='Episode Reward')
    if total_eps >= ROLLING_WINDOW:
        ax.plot(range(ROLLING_WINDOW, total_eps + 1), rolling(rewards, ROLLING_WINDOW),
                color='steelblue', linewidth=2, label=f'{ROLLING_WINDOW}-ep rolling avg')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Reward')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Success rate ──────────────────────────────────────────────────────────
    ax = axes[1]
    if total_eps >= ROLLING_WINDOW:
        rolling_acc = rolling(successes, ROLLING_WINDOW) * 100
        ax.plot(range(ROLLING_WINDOW, total_eps + 1), rolling_acc,
                color='seagreen', linewidth=2, label=f'{ROLLING_WINDOW}-ep success rate')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(-5, 105)
    ax.set_title('Routing Success Rate')
    ax.axhline(y=90, color='seagreen', linestyle='--', alpha=0.4, linewidth=1, label='90% target')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Epsilon decay ─────────────────────────────────────────────────────────
    ax = axes[2]
    epsilons = [max(0.01, 1.0 * (0.995 ** i)) for i in range(total_eps)]
    ax.plot(eps_range, epsilons, color='tomato', linewidth=1.5, label='Epsilon (ε)')
    ax.set_ylabel('Epsilon')
    ax.set_xlabel('Episode')
    ax.set_title('Exploration Rate (ε) Decay')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Neural Router DQN agent.')
    parser.add_argument('--episodes', type=int, default=TRAIN_EPISODES,
                        help=f'Number of training episodes (default: {TRAIN_EPISODES})')
    args = parser.parse_args()
    run_training(args.episodes)
