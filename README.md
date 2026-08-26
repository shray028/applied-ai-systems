# Applied AI Systems

Four standalone AI implementations focused on planning, reinforcement learning, robustness, and natural-language generation.

## Projects

- `drone_rescue/`: value-iteration planner for a grid-based emergency rescue route.
- `multi_armed_bandits/`: epsilon-greedy, UCB, and Thompson Sampling comparison.
- `robust_lunarlander/`: stochastic-action-failure wrapper and DQN/Double DQN training baseline.
- `news_summarization/`: abstractive summarization with T5, BART, or PEGASUS.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Each project can be run from the repository root. See its module README for the command and options.

## Notes

The LunarLander and summarization examples download their required environment/model assets on first use. The remaining modules run with NumPy and Matplotlib only.
