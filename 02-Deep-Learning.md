# Deep Learning - Complete Guide

## Table of Contents
1. Neural Network Fundamentals
2. Backpropagation
3. Activation Functions
4. Optimization Algorithms
5. Regularization Techniques
6. Convolutional Neural Networks
7. Recurrent Neural Networks
8. Attention Mechanisms
9. Batch Normalization & Layer Normalization
10. Advanced Architectures

---

## 1. NEURAL NETWORK FUNDAMENTALS

### Architecture Components

**Neuron (Perceptron)**:
```
output = activation(Σ(w_i · x_i) + b)
```

Components:
- Inputs (x): features
- Weights (w): learned parameters
- Bias (b): learned offset
- Activation: non-linearity

**Layer Types**:

**Input Layer**: Receives raw features

**Hidden Layers**: Learn representations
- Fully connected (dense)
- Convolutional
- Recurrent
- Attention

**Output Layer**: Produces predictions
- Binary: 1 neuron + sigmoid
- Multi-class: n neurons + softmax
- Regression: 1 neuron + linear

### Universal Approximation Theorem

**Statement**: A neural network with:
- Single hidden layer
- Sufficient neurons
- Non-linear activation

Can approximate any continuous function

**Implications**:
- Depth helps efficiency (fewer neurons needed)
- Width vs depth trade-off
- Practical: deep networks work better

### Forward Propagation

**Process**: Compute output from input

**Layer l**:
```
z^[l] = W^[l] · a^[l-1] + b^[l]
a^[l] = g^[l](z^[l])
```

Where:
- z: pre-activation
- a: activation (output)
- W: weights
- b: bias
- g: activation function

**Example** (3-layer network):
```
Input: x
Layer 1: z^[1] = W^[1]x + b^[1], a^[1] = ReLU(z^[1])
Layer 2: z^[2] = W^[2]a^[1] + b^[2], a^[2] = ReLU(z^[2])
Output: z^[3] = W^[3]a^[2] + b^[3], ŷ = sigmoid(z^[3])
```

### Loss Functions

**Binary Classification**: Binary cross-entropy
```
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

**Multi-class**: Categorical cross-entropy
```
L = -Σ y_i log(ŷ_i)
```

**Regression**: MSE
```
L = (y - ŷ)²
```

### Initialization

**Why it matters**:
- All zeros: neurons learn same features (symmetry)
- Too large: exploding gradients
- Too small: vanishing gradients

**Xavier/Glorot Initialization**:
```
W ~ N(0, √(2/(n_in + n_out)))
```
- For tanh, sigmoid
- Maintains variance across layers

**He Initialization**:
```
W ~ N(0, √(2/n_in))
```
- For ReLU
- Accounts for ReLU killing half the neurons

**Best practices**:
- Use He for ReLU
- Use Xavier for tanh/sigmoid
- Bias: initialize to zero
- Batch norm: reduces sensitivity to initialization

---

## 2. BACKPROPAGATION

### The Algorithm

**Goal**: Compute gradients of loss w.r.t. all parameters

**Key insight**: Chain rule applied recursively

**Process**:
1. Forward pass: compute activations
2. Compute loss
3. Backward pass: compute gradients
4. Update parameters

### Mathematical Derivation

**Output layer** (layer L):
```
dL/dz^[L] = dL/dŷ · dŷ/dz^[L]
```

For binary cross-entropy + sigmoid:
```
dL/dz^[L] = ŷ - y
```

**Hidden layer** l:
```
dL/dz^[l] = (W^[l+1])^T · dL/dz^[l+1] ⊙ g'(z^[l])
```

Where ⊙ is element-wise multiplication

**Parameter gradients**:
```
dL/dW^[l] = dL/dz^[l] · (a^[l-1])^T
dL/db^[l] = dL/dz^[l]
```

### Computational Graph

**Nodes**: Operations (add, multiply, activation)
**Edges**: Data flow

**Forward**: Compute values
**Backward**: Compute gradients

**Example**: y = (x + 2) * 3
```
Forward:
  a = x + 2
  y = a * 3

Backward:
  dy/da = 3
  dy/dx = dy/da · da/dx = 3 · 1 = 3
```

### Gradient Checking

**Purpose**: Verify backprop implementation

**Method**: Numerical gradient approximation
```
(f(θ + ε) - f(θ - ε)) / (2ε)
```

**Process**:
1. Compute gradients with backprop
2. Compute numerical gradients
3. Check relative difference < 10^-7

**Note**: Only for debugging, too slow for training

### Vanishing Gradients

**Problem**: Gradients become very small in early layers

**Causes**:
- Sigmoid/tanh: derivatives < 1
- Deep networks: gradients multiply
- Poor initialization

**Symptoms**:
- Early layers learn slowly
- Network doesn't converge

**Solutions**:
- ReLU activation
- Batch normalization
- Residual connections
- Better initialization
- Gradient clipping

### Exploding Gradients

**Problem**: Gradients become very large

**Causes**:
- Poor initialization
- Deep networks
- Recurrent networks

**Symptoms**:
- NaN in loss
- Unstable training
- Oscillating loss

**Solutions**:
- Gradient clipping: clip norm or value
- Better initialization
- Batch normalization
- Lower learning rate

### Gradient Clipping

**By value**:
```
g = max(min(g, threshold), -threshold)
```

**By norm**:
```
if ||g|| > threshold:
    g = g · threshold / ||g||
```

**When to use**: RNNs, unstable training

---

## 3. ACTIVATION FUNCTIONS

### Sigmoid

**Formula**: σ(x) = 1 / (1 + e^-x)

**Range**: (0, 1)

**Derivative**: σ'(x) = σ(x)(1 - σ(x))

**Pros**:
- Smooth, differentiable
- Output interpretable as probability
- Historically popular

**Cons**:
- Vanishing gradients (derivative max 0.25)
- Not zero-centered
- Expensive (exponential)

**When to use**: Binary classification output layer only

### Tanh

**Formula**: tanh(x) = (e^x - e^-x) / (e^x + e^-x)

**Range**: (-1, 1)

**Derivative**: tanh'(x) = 1 - tanh²(x)

**Pros**:
- Zero-centered (better than sigmoid)
- Smooth, differentiable

**Cons**:
- Vanishing gradients (derivative max 1)
- Expensive

**When to use**: RNNs (historically), rarely used now

### ReLU (Rectified Linear Unit)

**Formula**: ReLU(x) = max(0, x)

**Range**: [0, ∞)

**Derivative**: 
- 1 if x > 0
- 0 if x ≤ 0

**Pros**:
- No vanishing gradient for x > 0
- Computationally efficient
- Sparse activation (some neurons off)
- Empirically works well

**Cons**:
- Dying ReLU: neurons can get stuck at 0
- Not zero-centered
- Unbounded output

**When to use**: Default choice for hidden layers

**Dying ReLU problem**:
- Large negative bias → always negative input
- Gradient always 0 → no learning
- Solution: Leaky ReLU, He initialization, lower learning rate

### Leaky ReLU

**Formula**: LeakyReLU(x) = max(αx, x)

**Typical α**: 0.01

**Derivative**:
- 1 if x > 0
- α if x ≤ 0

**Pros**:
- Fixes dying ReLU
- Small gradient for negative values

**Cons**:
- Extra hyperparameter (α)

**When to use**: When ReLU causes dying neurons

### Parametric ReLU (PReLU)

**Formula**: PReLU(x) = max(αx, x)

**Difference**: α is learned parameter

**Pros**: Adaptive, can learn optimal α

**Cons**: More parameters

### ELU (Exponential Linear Unit)

**Formula**:
- x if x > 0
- α(e^x - 1) if x ≤ 0

**Pros**:
- Smooth everywhere
- Negative values push mean toward zero
- No dying neurons

**Cons**:
- Expensive (exponential)

### SELU (Scaled ELU)

**Formula**: λ · ELU(x, α)

**Special property**: Self-normalizing
- Specific α, λ values
- Maintains mean 0, variance 1

**Pros**:
- Self-normalizing (no batch norm needed)
- Strong theoretical guarantees

**Cons**:
- Requires specific initialization
- Sensitive to hyperparameters

**When to use**: Fully connected networks, want self-normalization

### Swish / SiLU

**Formula**: Swish(x) = x · sigmoid(x)

**Pros**:
- Smooth, non-monotonic
- Empirically better than ReLU in some cases
- Used in EfficientNet

**Cons**:
- More expensive than ReLU

### GELU (Gaussian Error Linear Unit)

**Formula**: GELU(x) = x · Φ(x)

Where Φ is Gaussian CDF

**Approximation**: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

**Pros**:
- Smooth, stochastic
- State-of-the-art in transformers (BERT, GPT)

**When to use**: Transformers, modern architectures

### Softmax

**Formula**: softmax(x_i) = e^x_i / Σ e^x_j

**Properties**:
- Outputs sum to 1
- Converts logits to probabilities
- Differentiable

**When to use**: Multi-class classification output

**Numerical stability**:
```
softmax(x_i) = e^(x_i - max(x)) / Σ e^(x_j - max(x))
```

### Activation Function Selection Guide

**Hidden layers**:
- Default: ReLU
- Dying ReLU issues: Leaky ReLU, ELU
- Transformers: GELU
- Self-normalizing: SELU

**Output layer**:
- Binary classification: Sigmoid
- Multi-class classification: Softmax
- Regression: Linear (no activation)
- Multi-label: Sigmoid (per class)

---

## 4. OPTIMIZATION ALGORITHMS

### Learning Rate

**Most important hyperparameter**

**Too high**:
- Divergence
- Oscillation
- Overshooting minima

**Too low**:
- Slow convergence
- Stuck in local minima
- Wasted computation

**Typical values**: 0.1, 0.01, 0.001, 0.0001

**Finding good LR**:
1. Learning rate range test
2. Start small, increase exponentially
3. Plot loss vs LR
4. Choose LR before loss explodes

### SGD (Stochastic Gradient Descent)

**Update**: θ = θ - α∇J(θ)

**Variants**:
- Batch: use all data
- Stochastic: use one sample
- Mini-batch: use batch of samples

**Pros**:
- Simple
- Memory efficient
- Can escape local minima (noise)

**Cons**:
- Slow convergence
- Sensitive to LR
- Same LR for all parameters

**When to use**: Simple problems, want best generalization

### Momentum

**Update**:
```
v = βv + ∇J(θ)
θ = θ - αv
```

**Intuition**: Rolling ball accumulates velocity

**Benefits**:
- Accelerates in consistent directions
- Dampens oscillations
- Faster convergence
- Helps escape local minima

**Hyperparameter**: β (typically 0.9)

**Physical analogy**: Ball rolling down hill with friction

### Nesterov Momentum (NAG)

**Update**:
```
v = βv + ∇J(θ - βv)
θ = θ - αv
```

**Difference**: Look ahead before computing gradient

**Benefits**:
- More responsive to changes
- Better convergence
- Corrects momentum if wrong direction

**When to use**: Improvement over standard momentum

### AdaGrad

**Update**:
```
G = G + (∇J(θ))²
θ = θ - α/(√G + ε) · ∇J(θ)
```

**Intuition**: Adapt LR per parameter based on history

**Benefits**:
- No manual LR tuning per parameter
- Good for sparse data
- Frequent parameters get smaller LR
- Infrequent parameters get larger LR

**Drawbacks**:
- LR decays too aggressively
- May stop learning
- Accumulates all gradients

**When to use**: Sparse features, NLP, early deep learning

### RMSprop

**Update**:
```
G = βG + (1-β)(∇J(θ))²
θ = θ - α/(√G + ε) · ∇J(θ)
```

**Difference from AdaGrad**: Exponential moving average

**Benefits**:
- Fixes AdaGrad's aggressive decay
- Adapts LR per parameter
- Works well in practice
- Good for RNNs

**Hyperparameters**:
- α: 0.001
- β: 0.9
- ε: 1e-8

**When to use**: RNNs, non-stationary problems

### Adam (Adaptive Moment Estimation)

**Update**:
```
m = β₁m + (1-β₁)∇J(θ)          # First moment (mean)
v = β₂v + (1-β₂)(∇J(θ))²        # Second moment (variance)
m̂ = m / (1 - β₁^t)              # Bias correction
v̂ = v / (1 - β₂^t)              # Bias correction
θ = θ - α · m̂ / (√v̂ + ε)
```

**Intuition**: Momentum + RMSprop + bias correction

**Benefits**:
- Adaptive LR per parameter
- Momentum for acceleration
- Bias correction for early iterations
- Works well out-of-the-box
- Most popular optimizer

**Hyperparameters**:
- α: 0.001 (or 0.0001)
- β₁: 0.9
- β₂: 0.999
- ε: 1e-8

**Drawbacks**:
- May converge to sharp minima (poor generalization)
- Can fail on some problems

**When to use**: Default choice, transformers, CNNs

### AdamW

**Modification**: Decouples weight decay from gradient

**Update**:
```
θ = θ - α · (m̂ / (√v̂ + ε) + λθ)
```

**Benefits**:
- Better regularization
- Improved generalization
- Fixes Adam's weight decay issue

**When to use**: Transformers (BERT, GPT), modern deep learning

### Nadam

**Combination**: Nesterov + Adam

**Benefits**: Faster convergence than Adam

### Comparison

| Optimizer | Speed | Generalization | Tuning | Use Case |
|-----------|-------|----------------|--------|----------|
| SGD | Slow | Best | Hard | Final model |
| SGD+Momentum | Medium | Excellent | Medium | Production |
| Adam | Fast | Good | Easy | Experiments |
| AdamW | Fast | Better | Easy | Transformers |
| RMSprop | Fast | Good | Easy | RNNs |

### Learning Rate Schedules

**Step Decay**:
```
α = α₀ · decay^(epoch / step_size)
```
- Drop LR every N epochs
- Typical: decay=0.5, step_size=10

**Exponential Decay**:
```
α = α₀ · e^(-kt)
```
- Smooth decay
- k controls decay rate

**1/t Decay**:
```
α = α₀ / (1 + kt)
```
- Inverse time decay

**Cosine Annealing**:
```
α = α_min + 0.5(α_max - α_min)(1 + cos(πt/T))
```
- Smooth decay
- Reaches minimum at T

**Warm Restarts (SGDR)**:
```
Cosine annealing + periodic resets
```
- Helps escape local minima
- Multiple cycles

**Warm-up**:
```
Linearly increase LR for first N steps
```
- Prevents instability early in training
- Common in transformers

**Reduce on Plateau**:
```
If validation loss doesn't improve for N epochs:
    α = α · factor
```
- Adaptive to training progress

**Cyclical Learning Rates**:
```
Cycle between min and max LR
```
- Helps exploration
- Faster convergence

**One Cycle Policy**:
```
1. Warm-up to max LR
2. Anneal to min LR
3. Final anneal to very low LR
```
- Fast convergence
- Good generalization

### Choosing Optimizer & Schedule

**Quick experiments**: Adam with default settings

**Best performance**: SGD + momentum + schedule

**Transformers**: AdamW + warm-up + cosine decay

**RNNs**: RMSprop or Adam

**Fine-tuning**: Lower LR (1/10 to 1/100 of training)

**Large batch**: Scale LR linearly with batch size

---


## 5. REGULARIZATION TECHNIQUES

### L1 and L2 Regularization

**L2 (Ridge, Weight Decay)**:
```
Loss_total = Loss + λ Σ w²
```

**Effects**:
- Penalizes large weights
- Weights shrink toward zero
- Smooth weight distribution
- Prefers many small weights

**Gradient**: ∇w = ∇Loss + 2λw

**When to use**: Default regularization, all features important

**L1 (Lasso)**:
```
Loss_total = Loss + λ Σ |w|
```

**Effects**:
- Penalizes absolute weights
- Produces sparse solutions
- Feature selection
- Some weights exactly zero

**Gradient**: ∇w = ∇Loss + λ · sign(w)

**When to use**: Feature selection, interpretability

**Elastic Net**: Combines L1 and L2

**λ selection**: Cross-validation, typical values 0.0001-0.1

### Dropout

**Training**: Randomly set neurons to zero with probability p

**Testing**: Use all neurons, scale by (1-p)

**Inverted Dropout**: Scale during training instead
```
a = a / (1-p)  # During training
```

**Why it works**:
- Prevents co-adaptation
- Ensemble effect
- Forces redundancy
- Noise injection

**Best practices**:
- p = 0.5 for hidden layers
- p = 0.2 for input layer
- Don't use on output
- Don't use with batch norm (redundant)
- Higher p for larger networks

**Variants**:
- DropConnect: Drop connections, not neurons
- Spatial Dropout: Drop entire feature maps (CNNs)
- Variational Dropout: Same mask across time (RNNs)

**When to use**: Fully connected layers, overfitting

### Early Stopping

**Method**:
1. Monitor validation loss
2. Save best model
3. Stop if no improvement for N epochs (patience)
4. Restore best weights

**Benefits**:
- Simple, effective
- No hyperparameter (except patience)
- Prevents overfitting

**Patience**: Typically 5-20 epochs

**Considerations**:
- May stop too early
- Validation set must be representative
- Can combine with other regularization

### Data Augmentation

**Purpose**: Artificially increase training data

**Computer Vision**:
- Geometric: rotation, flip, crop, zoom, shear
- Color: brightness, contrast, saturation, hue
- Noise: Gaussian, salt-pepper
- Cutout: mask random patches
- Mixup: blend images and labels
- CutMix: paste patches from other images
- AutoAugment: learned policies
- RandAugment: random augmentation

**NLP**:
- Synonym replacement
- Random insertion/deletion/swap
- Back-translation
- Paraphrasing
- Contextual word embeddings

**Audio**:
- Time stretching
- Pitch shifting
- Adding noise
- SpecAugment (for spectrograms)

**Benefits**:
- Improves generalization
- Reduces overfitting
- Increases effective dataset size

**Best practices**:
- Apply online (during training)
- Don't augment validation/test
- Domain-appropriate augmentations
- Not too aggressive (preserve labels)

### Batch Normalization

**Formula**:
```
μ = (1/m) Σ x_i                    # Batch mean
σ² = (1/m) Σ (x_i - μ)²           # Batch variance
x̂ = (x - μ) / √(σ² + ε)           # Normalize
y = γx̂ + β                         # Scale and shift
```

**Parameters**:
- γ, β: learned (scale and shift)
- ε: numerical stability (1e-5)

**Benefits**:
- Reduces internal covariate shift
- Allows higher learning rates
- Reduces sensitivity to initialization
- Acts as regularization
- Faster convergence

**Where to apply**: After linear layer, before activation

**Training vs Inference**:
- Training: use batch statistics
- Inference: use running statistics (exponential moving average)

**Considerations**:
- Batch size matters (too small → noisy estimates)
- Different behavior train/test
- Not ideal for RNNs (use Layer Norm)
- Can conflict with dropout

**When to use**: CNNs, fully connected networks

### Layer Normalization

**Difference from Batch Norm**: Normalize across features, not batch

**Formula**: Same as batch norm, but compute μ, σ² across features

**Benefits**:
- Independent of batch size
- Same behavior train/test
- Works with RNNs
- Works with batch size 1

**When to use**: RNNs, transformers, small batches

### Other Normalization

**Instance Normalization**: Normalize each sample independently
- Use: Style transfer, GANs

**Group Normalization**: Normalize groups of channels
- Use: Small batches, object detection

**Weight Normalization**: Normalize weight vectors
- Use: GANs, RL

### Max-Norm Constraints

**Method**: Constrain weight vector magnitude
```
if ||w|| > c:
    w = w · c / ||w||
```

**Benefits**: Prevents exploding weights

**When to use**: RNNs, with dropout

### Noise Injection

**Input noise**: Add Gaussian noise to inputs

**Weight noise**: Add noise to weights during training

**Benefits**: Regularization, robustness

**When to use**: Small datasets, want robustness

### Label Smoothing

**Method**: Soften one-hot labels
```
y_smooth = y(1 - ε) + ε/K
```

**Example**: [1, 0, 0] → [0.9, 0.05, 0.05] with ε=0.1

**Benefits**:
- Prevents overconfidence
- Better 