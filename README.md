# Deep Reinforcement Learning Systems

Two practical reinforcement-learning implementations focused on learning under uncertainty and robustness.

## Projects

- `multi_armed_bandits/`: epsilon-greedy, UCB, and Thompson Sampling comparison on Gaussian bandits.
- `robust_lunarlander/`: stochastic-action-failure wrapper with DQN and Double DQN training support.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run either module from the repository root. LunarLander downloads its environment assets on first use.
