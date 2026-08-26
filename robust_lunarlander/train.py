"""DQN/Double-DQN baseline for LunarLander with stochastic action failures."""
from __future__ import annotations

import argparse
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class ActionFailureWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, probability: float, seed: int = 7):
        super().__init__(env); self.probability = probability; self.rng = np.random.default_rng(seed)

    def step(self, action):
        executed = self.action_space.sample() if self.rng.random() < self.probability else action
        return self.env.step(executed)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, actions: int):
        super().__init__(); self.layers = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, actions))
    def forward(self, states): return self.layers(states)


def train(double_dqn: bool, failure_probability: float, episodes: int) -> None:
    env = ActionFailureWrapper(gym.make("LunarLander-v3"), failure_probability)
    online = QNetwork(env.observation_space.shape[0], env.action_space.n)
    target = QNetwork(env.observation_space.shape[0], env.action_space.n); target.load_state_dict(online.state_dict())
    optimiser, memory = torch.optim.Adam(online.parameters(), lr=1e-3), deque(maxlen=50_000)
    gamma, batch_size, epsilon = 0.99, 64, 1.0
    for episode in range(episodes):
        state, _ = env.reset(seed=episode); reward_sum = 0.0
        while True:
            action = env.action_space.sample() if random.random() < epsilon else int(online(torch.tensor(state, dtype=torch.float32)).argmax())
            nxt, reward, terminated, truncated, _ = env.step(action)
            memory.append((state, action, reward, nxt, terminated or truncated)); state, reward_sum = nxt, reward_sum + reward
            if len(memory) >= batch_size:
                states, actions, rewards, next_states, dones = map(np.array, zip(*random.sample(memory, batch_size)))
                states_t, next_t = torch.tensor(states, dtype=torch.float32), torch.tensor(next_states, dtype=torch.float32)
                q = online(states_t).gather(1, torch.tensor(actions).unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_actions = online(next_t).argmax(1) if double_dqn else target(next_t).argmax(1)
                    target_q = torch.tensor(rewards, dtype=torch.float32) + gamma * (1 - torch.tensor(dones, dtype=torch.float32)) * target(next_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q, target_q); optimiser.zero_grad(); loss.backward(); optimiser.step()
            if terminated or truncated: break
        epsilon = max(0.05, epsilon * 0.995)
        if episode % 20 == 0: target.load_state_dict(online.state_dict()); print(f"episode={episode:4d} reward={reward_sum:7.1f} epsilon={epsilon:.2f}")
    torch.save(online.state_dict(), "lunarlander_qnet.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--double-dqn", action="store_true"); parser.add_argument("--failure-probability", type=float, default=0.15); parser.add_argument("--episodes", type=int, default=400)
    args = parser.parse_args(); train(args.double_dqn, args.failure_probability, args.episodes)
