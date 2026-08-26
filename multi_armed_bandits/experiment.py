"""Compare epsilon-greedy, UCB, and Thompson Sampling on Gaussian bandits."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GaussianBandit:
    means: np.ndarray
    rng: np.random.Generator

    def pull(self, arm: int) -> float:
        return float(self.rng.normal(self.means[arm], 1.0))


def run(strategy: str, means: np.ndarray, steps: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    bandit = GaussianBandit(means, rng)
    values, counts = np.zeros(len(means)), np.zeros(len(means))
    rewards, optimal = np.zeros(steps), np.zeros(steps)
    posterior_mean, posterior_precision = np.zeros(len(means)), np.ones(len(means))
    for t in range(steps):
        if strategy == "epsilon":
            arm = rng.integers(len(means)) if rng.random() < 0.1 else int(values.argmax())
        elif strategy == "ucb":
            arm = int(np.argmax(values + 2.0 * np.sqrt(np.log(t + 1) / (counts + 1e-9))))
        else:
            arm = int(np.argmax(rng.normal(posterior_mean, 1 / np.sqrt(posterior_precision))))
        reward = bandit.pull(arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        posterior_precision[arm] += 1
        posterior_mean[arm] += (reward - posterior_mean[arm]) / posterior_precision[arm]
        rewards[t] = reward
        optimal[t] = arm == int(means.argmax())
    return rewards, optimal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    results = {name: [] for name in ("epsilon", "ucb", "thompson")}
    for _ in range(args.runs):
        means = rng.normal(0, 1, 10)
        for name in results:
            rewards, _ = run(name, means, args.steps, rng)
            results[name].append(rewards)
    for name, runs in results.items():
        plt.plot(np.mean(runs, axis=0).cumsum(), label=name.replace("epsilon", "ε-greedy"))
    plt.title("Cumulative reward across bandit strategies")
    plt.xlabel("Step"); plt.ylabel("Cumulative reward"); plt.legend(); plt.show()
