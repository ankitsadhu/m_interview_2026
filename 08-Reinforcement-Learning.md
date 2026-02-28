# Reinforcement Learning

## RL Basics

**Q: Explain the RL framework.**

A: Agent interacts with environment to maximize cumulative reward.

Components:
- State (s): current situation
- Action (a): agent's choice
- Reward (r): feedback from environment
- Policy (π): strategy mapping states to actions
- Value function (V): expected cumulative reward from state
- Q-function (Q): expected reward for state-action pair

Goal: Learn optimal policy π* that maximizes expected return

## Q-Learning

**Q: How does Q-learning work?**

A: Model-free, off-policy algorithm to learn optimal Q-function.

Update rule:
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

- α: learning rate
- γ: discount factor
- Explores using ε-greedy: random action with probability ε

Converges to optimal Q* with sufficient exploration.

Deep Q-Network (DQN):
- Use neural network to approximate Q-function
- Experience replay: store transitions, sample randomly
- Target network: stabilize training
- Applications: Atari games, robotics

## Policy Gradient

**Q: Explain policy gradient methods.**

A: Directly optimize policy parameters to maximize expected return.

REINFORCE algorithm:
- Sample trajectories using current policy
- Compute returns for each action
- Update policy to increase probability of high-reward actions

Advantage: can handle continuous action spaces

Actor-Critic:
- Actor: policy network
- Critic: value network
- Critic reduces variance of policy gradient
- Examples: A3C, PPO, SAC

## Exploration vs Exploitation

**Q: How do you balance exploration and exploitation?**

A:

Strategies:
- ε-greedy: random action with probability ε
- Softmax: sample actions proportional to Q-values
- Upper Confidence Bound (UCB): favor uncertain actions
- Thompson sampling: Bayesian approach
- Intrinsic motivation: reward for visiting new states

Decay exploration over time as agent learns.
