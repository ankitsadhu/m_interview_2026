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

## Markov Decision Process (MDP)

**Q: What is an MDP and why is it important in RL?**

A: Formal framework for sequential decision-making under uncertainty.

**Components**:
- States (S): set of all possible states
- Actions (A): set of all possible actions
- Transition function: P(s'|s,a) - probability of next state
- Reward function: R(s,a,s') - immediate reward
- Discount factor (γ): importance of future rewards

**Markov property**: Future depends only on current state, not history
- P(s_{t+1}|s_t, a_t, s_{t-1}, ..., s_0) = P(s_{t+1}|s_t, a_t)

**Return**: G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... = Σ γ^k r_{t+k}

**Why discount factor**:
- γ = 0: only immediate reward (myopic)
- γ = 1: all rewards equal (may not converge)
- γ ∈ (0,1): balance immediate and future rewards
- Typical values: 0.9, 0.95, 0.99

**Goal**: Find policy π that maximizes expected return

## Value Functions

**Q: Explain state-value and action-value functions.**

A:

**State-value function**: V^π(s) = E_π[G_t | s_t = s]
- Expected return starting from state s, following policy π
- "How good is this state?"

**Action-value function**: Q^π(s,a) = E_π[G_t | s_t = s, a_t = a]
- Expected return starting from state s, taking action a, then following π
- "How good is this action in this state?"

**Relationship**: V^π(s) = Σ_a π(a|s) Q^π(s,a)

**Bellman equations**:

State-value:
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]

Action-value:
Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ Σ_{a'} π(a'|s')Q^π(s',a')]

**Optimal value functions**:
- V*(s) = max_π V^π(s)
- Q*(s,a) = max_π Q^π(s,a)

**Optimal policy**: π*(s) = argmax_a Q*(s,a)

## Dynamic Programming

**Q: Explain value iteration and policy iteration.**

A:

**Value Iteration**:
1. Initialize V(s) arbitrarily
2. Repeat until convergence:
   - V(s) ← max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]
3. Extract policy: π(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]

**Policy Iteration**:
1. Initialize policy π arbitrarily
2. Repeat until convergence:
   - Policy Evaluation: compute V^π
   - Policy Improvement: π'(s) = argmax_a Q^π(s,a)

**Comparison**:
- Value iteration: simpler, one step per iteration
- Policy iteration: faster convergence, two steps per iteration
- Both require model (transition probabilities)

**Limitations**:
- Need complete model of environment
- Computationally expensive for large state spaces
- Not practical for real-world problems

## Monte Carlo Methods

**Q: How do Monte Carlo methods work in RL?**

A: Learn from complete episodes (sample trajectories).

**Algorithm**:
1. Generate episode using policy π
2. For each state s visited:
   - Compute return G from that point
   - Update V(s) ← V(s) + α[G - V(s)]

**First-visit vs Every-visit**:
- First-visit: update only first occurrence of state
- Every-visit: update all occurrences

**Advantages**:
- Model-free (don't need transition probabilities)
- Can learn from experience
- Unbiased estimates

**Disadvantages**:
- High variance
- Requires complete episodes
- Slow convergence

**Monte Carlo Control**:
- Use ε-greedy for exploration
- Update Q(s,a) instead of V(s)
- Improve policy greedily

## Temporal Difference Learning

**Q: What is TD learning and how does it differ from Monte Carlo?**

A: Learn from incomplete episodes using bootstrapping.

**TD(0) update**:
V(s) ← V(s) + α[r + γV(s') - V(s)]

**TD target**: r + γV(s')
**TD error**: δ = r + γV(s') - V(s)

**Comparison with Monte Carlo**:

| Aspect | Monte Carlo | TD Learning |
|--------|-------------|-------------|
| Update | After episode | After each step |
| Bias | Unbiased | Biased |
| Variance | High | Low |
| Convergence | Slower | Faster |
| Episodes | Complete | Incomplete OK |

**Advantages of TD**:
- Online learning (update every step)
- Works with incomplete episodes
- Lower variance than MC
- Faster convergence

**SARSA** (on-policy TD control):
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- Update using action actually taken (a')

**Q-Learning** (off-policy TD control):
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
- Update using best action (max)

**On-policy vs Off-policy**:
- On-policy: learn about policy being followed (SARSA)
- Off-policy: learn about different policy (Q-learning)

## Deep Q-Networks (DQN)

**Q: Explain DQN and its key innovations.**

A: Use deep neural network to approximate Q-function.

**Challenges with neural networks**:
- Correlated samples (sequential data)
- Non-stationary targets (Q-values change)
- Instability and divergence

**Key innovations**:

1. **Experience Replay**:
   - Store transitions (s, a, r, s') in replay buffer
   - Sample random mini-batches for training
   - Breaks correlation between samples
   - Improves data efficiency

2. **Target Network**:
   - Separate network for computing targets
   - Update target network periodically (every C steps)
   - Stabilizes training
   - Reduces oscillations

**Algorithm**:
```
Initialize Q-network with weights θ
Initialize target network with weights θ⁻ = θ
Initialize replay buffer D

For each episode:
    For each step:
        Select action: a = argmax_a Q(s,a;θ) with ε-greedy
        Execute action, observe r, s'
        Store (s,a,r,s') in D
        
        Sample mini-batch from D
        Compute target: y = r + γ max_a' Q(s',a';θ⁻)
        Update Q-network: minimize (y - Q(s,a;θ))²
        
        Every C steps: θ⁻ ← θ
```

**Improvements**:
- Double DQN: reduce overestimation
- Dueling DQN: separate value and advantage streams
- Prioritized Experience Replay: sample important transitions more
- Rainbow: combines multiple improvements

## Policy Gradient Methods

**Q: Derive the policy gradient theorem.**

A:

**Objective**: Maximize expected return
J(θ) = E_π[G_t] = E_π[Σ γ^k r_{t+k}]

**Policy Gradient Theorem**:
∇_θ J(θ) = E_π[∇_θ log π(a|s;θ) Q^π(s,a)]

**Intuition**: Increase probability of actions with high Q-values

**REINFORCE algorithm**:
1. Generate episode using π(a|s;θ)
2. For each step t:
   - Compute return G_t
   - Update: θ ← θ + α∇_θ log π(a_t|s_t;θ) G_t

**Advantages**:
- Can handle continuous action spaces
- Can learn stochastic policies
- Better convergence properties

**Disadvantages**:
- High variance
- Sample inefficient
- Slow convergence

**Variance reduction**:
- Baseline: subtract baseline b(s) from return
  - ∇_θ J(θ) = E_π[∇_θ log π(a|s;θ) (Q^π(s,a) - b(s))]
  - Common baseline: V^π(s)
- Advantage function: A^π(s,a) = Q^π(s,a) - V^π(s)
  - Measures how much better action a is than average

## Actor-Critic Methods

**Q: Explain actor-critic architecture.**

A: Combine policy gradient (actor) with value function (critic).

**Components**:
- Actor: policy network π(a|s;θ)
- Critic: value network V(s;w) or Q(s,a;w)

**Algorithm**:
1. Actor selects action: a ~ π(a|s;θ)
2. Execute action, observe r, s'
3. Critic evaluates: TD error δ = r + γV(s';w) - V(s;w)
4. Update critic: w ← w + α_w δ ∇_w V(s;w)
5. Update actor: θ ← θ + α_θ δ ∇_θ log π(a|s;θ)

**Advantages**:
- Lower variance than REINFORCE (critic provides baseline)
- Online learning (update every step)
- Faster convergence

**Disadvantages**:
- Biased estimates (critic is approximate)
- Two networks to train
- Hyperparameter tuning

**Variants**:
- A2C (Advantage Actor-Critic): synchronous
- A3C (Asynchronous Advantage Actor-Critic): parallel workers
- DDPG (Deep Deterministic Policy Gradient): continuous actions
- TD3 (Twin Delayed DDPG): improvements over DDPG
- SAC (Soft Actor-Critic): maximum entropy RL

## Proximal Policy Optimization (PPO)

**Q: Why is PPO popular and how does it work?**

A: State-of-the-art policy gradient method with stability and simplicity.

**Problem with vanilla policy gradient**:
- Large policy updates can be destructive
- Performance collapse
- Difficult to recover

**Solution**: Constrain policy updates

**PPO-Clip objective**:
L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

Where:
- r_t(θ) = π(a|s;θ) / π(a|s;θ_old) (probability ratio)
- Â_t: advantage estimate
- ε: clipping parameter (typically 0.2)

**Intuition**:
- If advantage positive: increase probability (but not too much)
- If advantage negative: decrease probability (but not too much)
- Clip prevents large updates

**Advantages**:
- Simple to implement
- Stable training
- Good sample efficiency
- Few hyperparameters
- Works well in practice

**Used in**: OpenAI Five (Dota 2), ChatGPT (RLHF)

## Trust Region Policy Optimization (TRPO)

**Q: How does TRPO ensure stable policy updates?**

A: Constrain policy updates using KL divergence.

**Objective**:
maximize E[π(a|s;θ)/π(a|s;θ_old) A(s,a)]
subject to E[KL(π_old || π_new)] ≤ δ

**Intuition**: Maximize improvement while staying close to old policy

**Implementation**:
- Use conjugate gradient for optimization
- Line search for step size
- Computationally expensive

**Advantages**:
- Monotonic improvement guarantee
- Stable training
- Theoretical guarantees

**Disadvantages**:
- Complex implementation
- Computationally expensive
- PPO is simpler alternative

## Multi-Armed Bandits

**Q: Explain the multi-armed bandit problem.**

A: Simplified RL: single state, multiple actions, immediate rewards.

**Problem**: Balance exploration (try new actions) vs exploitation (use best known action)

**Algorithms**:

1. **ε-greedy**:
   - Exploit: choose best action with probability 1-ε
   - Explore: random action with probability ε

2. **Upper Confidence Bound (UCB)**:
   - a_t = argmax_a [Q(a) + c√(ln t / N(a))]
   - Exploration bonus for uncertain actions
   - c: exploration parameter

3. **Thompson Sampling**:
   - Bayesian approach
   - Sample from posterior distribution
   - Choose action with highest sample
   - Optimal for many problems

**Regret**: Difference between optimal and actual reward
- Goal: minimize cumulative regret

**Applications**:
- A/B testing
- Clinical trials
- Ad placement
- Recommendation systems

## Model-Based RL

**Q: Compare model-free and model-based RL.**

A:

**Model-free**: Learn policy/value function directly from experience
- Examples: Q-learning, policy gradient
- Pros: Simple, no model bias
- Cons: Sample inefficient

**Model-based**: Learn model of environment, use for planning
- Model: P(s'|s,a) and R(s,a)
- Use model to simulate experience
- Plan using model

**Approaches**:

1. **Learn model, then plan**:
   - Learn transition and reward models
   - Use dynamic programming or tree search
   - Example: AlphaZero (MCTS + learned model)

2. **Dyna architecture**:
   - Learn model and policy simultaneously
   - Use real experience to update model and policy
   - Use simulated experience (from model) to update policy
   - Improves sample efficiency

**Advantages**:
- Sample efficient (reuse experience)
- Can plan ahead
- Transfer learning (reuse model)

**Disadvantages**:
- Model errors compound
- Computationally expensive
- Model bias

**When to use**:
- Expensive real-world interactions
- Need sample efficiency
- Environment is learnable

## Inverse Reinforcement Learning

**Q: What is inverse RL and when is it useful?**

A: Learn reward function from expert demonstrations.

**Problem**: Reward engineering is hard
- Difficult to specify reward function
- May lead to unintended behavior

**Inverse RL**: Given expert trajectories, infer reward function

**Approach**:
1. Observe expert demonstrations
2. Find reward function that makes expert optimal
3. Use learned reward to train policy

**Applications**:
- Imitation learning
- Apprenticeship learning
- Robotics (learn from human demonstrations)
- Autonomous driving

**Challenges**:
- Reward ambiguity (multiple rewards explain behavior)
- Requires expert demonstrations
- Computationally expensive

## Imitation Learning

**Q: Explain behavioral cloning and its limitations.**

A:

**Behavioral Cloning**: Supervised learning from expert demonstrations
- Treat as supervised learning: (state, action) pairs
- Train policy to mimic expert

**Algorithm**:
1. Collect expert demonstrations
2. Train policy: minimize loss(π(s), a_expert)
3. Deploy policy

**Advantages**:
- Simple
- No reward function needed
- Fast training

**Limitations**:
- Distribution shift: training states ≠ test states
- Compounding errors
- No exploration
- Requires many demonstrations

**DAgger** (Dataset Aggregation):
1. Train policy on expert data
2. Run policy, collect states
3. Get expert labels for those states
4. Retrain on combined data
5. Repeat

**Fixes distribution shift problem**

## Hierarchical RL

**Q: Why use hierarchical RL?**

A: Decompose complex tasks into subtasks.

**Motivation**:
- Long time horizons
- Sparse rewards
- Complex tasks

**Approaches**:

1. **Options framework**:
   - Option: (initiation set, policy, termination condition)
   - Learn options (reusable skills)
   - High-level policy selects options

2. **Feudal RL**:
   - Manager: sets goals for workers
   - Workers: achieve goals
   - Hierarchical structure

**Advantages**:
- Faster learning
- Transfer learning (reuse skills)
- Interpretability

**Challenges**:
- How to discover subtasks
- Credit assignment across levels

## Offline RL

**Q: What is offline RL and why is it important?**

A: Learn from fixed dataset without environment interaction.

**Motivation**:
- Expensive/dangerous to interact (healthcare, robotics)
- Have historical data
- Safety concerns

**Challenges**:
- Distribution shift: policy may visit unseen states
- Overestimation: Q-values for unseen actions
- No exploration

**Approaches**:

1. **Conservative Q-Learning (CQL)**:
   - Penalize Q-values for unseen actions
   - Conservative estimates

2. **Behavior regularization**:
   - Keep policy close to behavior policy
   - Avoid out-of-distribution actions

**Applications**:
- Healthcare (learn from medical records)
- Robotics (learn from demonstrations)
- Recommendation systems (learn from logs)


## RL Scenarios and Applications

### Scenario 1: Design a Recommendation System with RL

**Q: How would you use RL for a recommendation system?**

A:

**Problem formulation**:
- State: User context (history, demographics, session info)
- Action: Recommend item
- Reward: User engagement (click, watch time, purchase)
- Policy: Recommendation strategy

**Challenges**:

1. **Delayed rewards**: User may engage later
2. **Sparse rewards**: Most recommendations not clicked
3. **Large action space**: Millions of items
4. **Exploration-exploitation**: Show diverse items vs popular items
5. **Non-stationarity**: User preferences change

**Approach**:

**1. Contextual Bandits** (simplified RL):
- Treat each recommendation as independent
- No sequential dependency
- Faster, simpler
- Use: LinUCB, Thompson Sampling

**2. Full RL** (consider long-term):
- Model user session as MDP
- Maximize lifetime value
- Use: DQN, policy gradient

**Architecture**:

```
State representation:
- User features: demographics, history embeddings
- Context: time, device, location
- Session: items viewed, time spent

Action space reduction:
- Candidate generation: retrieve top-K items
- RL selects from candidates
- Reduces action space from millions to hundreds

Reward shaping:
- Immediate: click (+1), no click (0)
- Delayed: watch time, purchase
- Negative: skip (-0.1)
- Diversity bonus: recommend from different categories

Model:
- DQN with dueling architecture
- Actor-critic for continuous optimization
- Offline RL to learn from logs
```

**Training**:
1. Collect data from existing system (behavior policy)
2. Train offline RL model
3. Evaluate offline (counterfactual evaluation)
4. Deploy with exploration (ε-greedy or Thompson sampling)
5. Collect online data
6. Retrain periodically

**Evaluation**:
- Offline: Replay evaluation, importance sampling
- Online: A/B test (CTR, engagement, revenue)
- Long-term: User retention, lifetime value

**Production considerations**:
- Real-time inference (<100ms)
- Exploration strategy (avoid filter bubbles)
- Fairness (diverse recommendations)
- Cold start (new users/items)

### Scenario 2: Autonomous Vehicle Control

**Q: Design an RL system for autonomous driving.**

A:

**Problem formulation**:
- State: Sensor data (camera, LiDAR, GPS, speed)
- Action: Steering, acceleration, braking
- Reward: Safe, efficient driving
- Policy: Driving strategy

**Challenges**:
- Safety-critical (no room for error)
- Continuous state and action spaces
- Multi-objective (safety, efficiency, comfort)
- Sim-to-real gap
- Interpretability

**Approach**:

**State representation**:
- Raw sensors: camera images, LiDAR point clouds
- Processed: lane detection, object detection
- Vehicle state: speed, position, orientation
- Map information: road layout, traffic rules

**Action space**:
- Continuous: steering angle, acceleration
- Discrete: lane change decisions
- Hierarchical: high-level (route) + low-level (control)

**Reward function**:
```
R = w1 * safety + w2 * efficiency + w3 * comfort + w4 * rule_following

Safety: -1000 for collision, -10 for close calls
Efficiency: +speed (within limit), -fuel consumption
Comfort: -jerk (sudden changes)
Rules: -violations (red light, speed limit)
```

**Training strategy**:

1. **Simulation first**:
   - Train in simulator (CARLA, SUMO)
   - Safe exploration
   - Diverse scenarios
   - Parallel training

2. **Imitation learning**:
   - Learn from human drivers
   - Behavioral cloning as initialization
   - Reduces exploration needed

3. **RL fine-tuning**:
   - PPO or SAC for continuous control
   - Curriculum learning (easy → hard scenarios)
   - Multi-task learning (different weather, traffic)

4. **Sim-to-real transfer**:
   - Domain randomization
   - Domain adaptation
   - Real-world fine-tuning (carefully)

**Safety measures**:
- Constrained RL (safety constraints)
- Shielding (safety controller overrides)
- Formal verification
- Extensive testing

**Evaluation**:
- Simulation: success rate, collision rate
- Closed track: controlled environment
- Real-world: gradual deployment, human supervision

### Scenario 3: Dynamic Pricing with RL

**Q: Use RL for dynamic pricing in e-commerce.**

A:

**Problem formulation**:
- State: Inventory, demand, competitor prices, time
- Action: Set price
- Reward: Revenue or profit
- Policy: Pricing strategy

**Challenges**:
- Delayed feedback (purchase decision takes time)
- Competitor reactions
- Customer sensitivity
- Inventory constraints
- Seasonality

**Approach**:

**State features**:
- Product: category, cost, inventory level
- Demand: historical sales, trends, seasonality
- Competition: competitor prices, market share
- Customer: segment, price sensitivity
- Context: time of day, day of week, holidays

**Action space**:
- Discrete: price buckets (e.g., $10, $15, $20)
- Continuous: price in range [cost, max_price]

**Reward function**:
```
R = (price - cost) * sales - holding_cost * inventory

Or maximize:
- Revenue: price * sales
- Profit: (price - cost) * sales
- Market share: sales / total_market_sales
```

**Model**:
- Contextual bandits (if treating independently)
- Full RL (if considering inventory dynamics)
- Multi-armed bandits (simple version)

**Algorithm**:
1. Thompson Sampling for exploration
2. LinUCB for linear models
3. DQN for complex state spaces
4. Policy gradient for continuous prices

**Training**:
- Offline RL on historical data
- Counterfactual evaluation
- Careful online deployment (start with small subset)
- A/B testing

**Constraints**:
- Minimum price (above cost)
- Maximum price (market ceiling)
- Price stability (avoid frequent changes)
- Fairness (similar customers, similar prices)

**Evaluation**:
- Revenue increase
- Profit margin
- Sales volume
- Customer satisfaction
- Competitor response

### Scenario 4: Chatbot with RL

**Q: Design a conversational agent using RL.**

A:

**Problem formulation**:
- State: Conversation history, user intent, context
- Action: Response (from candidate set)
- Reward: User satisfaction, task completion
- Policy: Conversation strategy

**Challenges**:
- Sparse rewards (satisfaction at end)
- Large action space (many possible responses)
- Natural language understanding
- Multi-turn conversations
- Subjective rewards

**Approach**:

**State representation**:
- Conversation history: embeddings of previous turns
- User intent: classified intent (question, complaint, etc.)
- Entities: extracted information
- Context: user profile, session info

**Action space**:
- Template-based: select template + fill slots
- Retrieval-based: select from response database
- Generative: generate response (with RL fine-tuning)

**Reward function**:
```
Immediate rewards:
- User continues conversation: +0.1
- User provides information: +0.5
- User expresses frustration: -1.0

Terminal rewards:
- Task completed: +10
- User satisfied (survey): +5
- User abandons: -5

Auxiliary rewards:
- Response relevance (semantic similarity)
- Response diversity (avoid repetition)
- Response length (not too long/short)
```

**Training**:

1. **Supervised pre-training**:
   - Train on human conversations
   - Behavioral cloning

2. **RL fine-tuning**:
   - Policy gradient (REINFORCE)
   - Actor-critic
   - Reward from user feedback

3. **Reward learning**:
   - Learn reward model from user ratings
   - Inverse RL from expert conversations

**Evaluation**:
- Task success rate
- Average turns to completion
- User satisfaction scores
- Engagement metrics
- Human evaluation

**Production**:
- Hybrid approach: RL for strategy, templates for safety
- Human handoff for complex cases
- Continuous learning from interactions

### Scenario 5: Resource Allocation with RL

**Q: Use RL for cloud resource allocation.**

A:

**Problem formulation**:
- State: Current resource usage, workload, SLA
- Action: Allocate resources (CPU, memory, instances)
- Reward: Cost savings + SLA compliance
- Policy: Allocation strategy

**Challenges**:
- Multi-objective (cost vs performance)
- Constraints (SLA requirements)
- Non-stationary (workload changes)
- High-dimensional state/action spaces

**Approach**:

**State features**:
- Resource usage: CPU, memory, network, disk
- Workload: request rate, queue length
- Performance: latency, throughput, error rate
- Time: hour of day, day of week (seasonality)
- Predictions: forecasted workload

**Action space**:
- Discrete: number of instances (scale up/down)
- Continuous: resource allocation per instance
- Multi-dimensional: CPU + memory + instances

**Reward function**:
```
R = -cost - penalty_SLA_violation

cost = price_per_instance * num_instances * time
penalty = large_value if latency > SLA_threshold

Or multi-objective:
R = w1 * (-cost) + w2 * performance + w3 * (-SLA_violations)
```

**Algorithm**:
- DQN for discrete actions
- DDPG/TD3/SAC for continuous actions
- PPO for stability
- Constrained RL for SLA guarantees

**Training**:
1. Simulate workload patterns
2. Train offline on historical data
3. Safe exploration (start conservative)
4. Gradual deployment

**Safety measures**:
- Minimum resources (always meet SLA)
- Maximum resources (cost ceiling)
- Gradual changes (avoid thrashing)
- Fallback to rule-based system

**Evaluation**:
- Cost reduction
- SLA compliance rate
- Resource utilization
- Response time to workload changes

## RL Interview Questions

### Conceptual Questions

**Q: When would you use RL vs supervised learning?**

A:

**Use RL when**:
- Sequential decision-making
- Delayed rewards
- Need to explore environment
- No labeled data, only rewards
- Examples: games, robotics, resource allocation

**Use supervised learning when**:
- Have labeled data
- Direct input-output mapping
- No sequential dependency
- Immediate feedback
- Examples: classification, regression

**Hybrid approaches**:
- Imitation learning: supervised pre-training + RL fine-tuning
- Reward learning: supervised learning of reward function

**Q: Explain the credit assignment problem in RL.**

A:

**Problem**: Which actions were responsible for the reward?

**Challenges**:
- Delayed rewards: reward comes many steps later
- Sparse rewards: reward only at end of episode
- Stochastic environment: same action, different outcomes

**Solutions**:
- Discount factor: recent actions more important
- Eligibility traces: track recent state-action pairs
- Advantage function: compare to baseline
- Reward shaping: provide intermediate rewards
- Hindsight experience replay: learn from failures

**Q: How do you handle continuous action spaces?**

A:

**Approaches**:

1. **Discretization**:
   - Divide continuous space into bins
   - Simple but loses precision
   - Curse of dimensionality

2. **Policy gradient**:
   - Directly output continuous actions
   - Gaussian policy: μ(s), σ(s)
   - Sample action from distribution

3. **Actor-critic**:
   - Actor outputs continuous action
   - Critic evaluates action
   - DDPG, TD3, SAC

4. **Deterministic policy gradient**:
   - Policy outputs deterministic action
   - Add noise for exploration

**Q: What is the exploration-exploitation dilemma?**

A:

**Dilemma**: Should agent try new actions (explore) or use best known action (exploit)?

**Too much exploration**: Waste time on suboptimal actions
**Too much exploitation**: Miss better actions

**Strategies**:
- ε-greedy: explore with probability ε
- Softmax: sample proportional to Q-values
- UCB: exploration bonus for uncertain actions
- Thompson sampling: Bayesian approach
- Optimism in face of uncertainty
- Intrinsic motivation: curiosity-driven exploration

**Decay exploration**: Start high, decrease over time

**Q: How do you evaluate an RL agent?**

A:

**Metrics**:
- Average return: mean cumulative reward
- Success rate: % of episodes achieving goal
- Sample efficiency: performance vs training steps
- Convergence speed: time to reach performance
- Robustness: performance across different scenarios

**Evaluation methods**:
- Hold-out test episodes
- Different random seeds
- Ablation studies (remove components)
- Comparison to baselines
- Human evaluation (for subjective tasks)

**Challenges**:
- High variance: need multiple runs
- Environment stochasticity
- Hyperparameter sensitivity
- Reproducibility issues

### Practical Questions

**Q: How do you debug an RL agent that's not learning?**

A:

**Checklist**:

1. **Reward function**:
   - Is reward signal correct?
   - Too sparse? Add intermediate rewards
   - Reward scale appropriate?
   - Check for bugs in reward computation

2. **Exploration**:
   - Is agent exploring enough?
   - Increase ε or temperature
   - Check if stuck in local optimum

3. **Learning rate**:
   - Too high: unstable, divergence
   - Too low: slow learning
   - Try different values

4. **Network architecture**:
   - Too small: underfitting
   - Too large: overfitting, slow
   - Check activation functions

5. **Hyperparameters**:
   - Discount factor γ
   - Batch size
   - Replay buffer size
   - Target network update frequency

6. **Environment**:
   - Is task learnable?
   - Try simpler version first
   - Check for bugs in environment

7. **Baseline**:
   - Compare to random policy
   - Compare to simple heuristic
   - Sanity check: can overfit to single episode?

**Debugging tools**:
- Plot learning curves
- Visualize Q-values
- Log episode returns
- Check gradient magnitudes
- Visualize agent behavior

**Q: How do you handle sparse rewards?**

A:

**Strategies**:

1. **Reward shaping**:
   - Add intermediate rewards
   - Potential-based shaping (preserves optimality)
   - Domain knowledge

2. **Curriculum learning**:
   - Start with easier tasks
   - Gradually increase difficulty
   - Transfer learning

3. **Hindsight experience replay (HER)**:
   - Relabel failed episodes
   - "What if that was the goal?"
   - Learn from failures

4. **Exploration bonuses**:
   - Intrinsic motivation
   - Curiosity-driven exploration
   - Count-based exploration

5. **Imitation learning**:
   - Learn from demonstrations
   - Provides initial policy
   - Reduces exploration needed

6. **Hierarchical RL**:
   - Break into subtasks
   - Learn skills separately
   - Compose skills

**Q: How do you ensure safety in RL?**

A:

**Approaches**:

1. **Constrained RL**:
   - Add safety constraints
   - Optimize subject to constraints
   - Lagrangian methods

2. **Safe exploration**:
   - Start with safe policy
   - Limit exploration
   - Risk-sensitive RL

3. **Shielding**:
   - Safety controller monitors agent
   - Overrides unsafe actions
   - Formal verification

4. **Simulation first**:
   - Train in simulation
   - Extensive testing
   - Sim-to-real transfer

5. **Human oversight**:
   - Human-in-the-loop
   - Approval for critical actions
   - Gradual autonomy

6. **Robust RL**:
   - Train on diverse scenarios
   - Adversarial training
   - Worst-case optimization

**Q: Compare on-policy and off-policy methods.**

A:

**On-policy** (SARSA, PPO, A3C):
- Learn about policy being followed
- Update using current policy's experience
- More stable
- Sample inefficient (can't reuse old data)

**Off-policy** (Q-learning, DQN, DDPG):
- Learn about different policy (target policy)
- Can use experience from any policy
- Sample efficient (replay buffer)
- Less stable (importance sampling)

**When to use**:
- On-policy: Stability important, can generate data easily
- Off-policy: Sample efficiency important, expensive data collection

## Key Takeaways

**Core concepts**:
- MDP framework
- Value functions and Bellman equations
- Policy vs value-based methods
- Exploration vs exploitation

**Algorithms to know**:
- Q-learning and DQN
- Policy gradient and REINFORCE
- Actor-critic (A3C, PPO)
- Model-based methods

**Practical considerations**:
- Reward design is critical
- Exploration strategies
- Sample efficiency
- Stability and convergence
- Safety and constraints

**Applications**:
- Recommendation systems
- Robotics and control
- Resource allocation
- Game playing
- Autonomous systems

**Interview tips**:
- Understand fundamentals deeply
- Know when to use RL vs other methods
- Discuss trade-offs
- Practical experience matters
- Be ready to design RL systems
