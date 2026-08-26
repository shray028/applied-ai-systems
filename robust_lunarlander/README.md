# Robust LunarLander Reinforcement Learning

The wrapper injects stochastic action failures into LunarLander. The training script supports both DQN and Double DQN targets.

```bash
python robust_lunarlander/train.py --double-dqn --failure-probability 0.15
```
