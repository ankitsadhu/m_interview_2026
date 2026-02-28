# ML Fundamentals - Complete Guide

## Table of Contents
1. Learning Paradigms
2. Bias-Variance Tradeoff
3. Overfitting & Regularization
4. Loss Functions
5. Evaluation Metrics
6. Feature Engineering
7. Cross-Validation
8. Gradient Descent
9. Ensemble Methods
10. Class Imbalance

---

## 1. LEARNING PARADIGMS

### Supervised Learning

**Definition**: Learning from labeled data (input-output pairs) to predict outputs for new inputs.

**Types**:

**Classification**: Predict discrete labels
- Binary: spam/not spam, fraud/legitimate
- Multi-class: digit recognition (0-9), image classification
- Multi-label: tag prediction (multiple tags per item)

**Regression**: Predict continuous values
- Linear regression: house prices, temperature
- Non-linear: stock prices, demand forecasting

**Common Algorithms**:
- Linear models: Linear/Logistic Regression, SVM
- Tree-based: Decision Trees, Random Forest, Gradient Boosting
- Neural Networks: MLP, CNN, RNN
- Instance-based: KNN, k-means
- Probabilistic: Naive Bayes, Gaussian Processes

**When to use**: Have labeled data, clear input-output relationship

---

### Unsupervised Learning

**Definition**: Learning patterns from unlabeled data without explicit targets.

**Types**:

**Clustering**: Group similar data points
- K-means: partition into k clusters
- Hierarchical: build cluster tree
- DBSCAN: density-based, finds arbitrary shapes
- Gaussian Mixture Models: probabilistic clustering

**Dimensionality Reduction**: Reduce feature space
- PCA: linear, preserves variance
- t-SNE: non-linear, for visualization
- UMAP: faster than t-SNE, preserves global structure
- Autoencoders: neural network approach

**Anomaly Detection**: Find outliers
- Isolation Forest
- One-class SVM
- Autoencoders (reconstruction error)

**Association Rules**: Find relationships
- Market basket analysis
- Apriori algorithm

**When to use**: No labels, exploratory analysis, preprocessing

---

### Semi-Supervised Learning

**Definition**: Leverage both labeled (small) and unlabeled (large) data.

**Techniques**:
- Self-training: train on labeled, predict on unlabeled, add confident predictions
- Co-training: multiple views of data, models teach each other
- Multi-view learning
- Pseudo-labeling

**Use cases**: Labeling is expensive (medical imaging, speech recognition)

---

### Reinforcement Learning

**Definition**: Agent learns by interacting with environment, receiving rewards/penalties.

**Components**:
- State, Action, Reward, Policy, Value function
- Goal: maximize cumulative reward

**Algorithms**: Q-learning, SARSA, Policy Gradients, Actor-Critic

**Use cases**: Game playing, robotics, recommendation systems, autonomous vehicles

---

## 2. BIAS-VARIANCE TRADEOFF

### Mathematical Foundation

**Total Error** = Bias² + Variance + Irreducible Error

**Bias**: Error from wrong assumptions
- High bias → underfitting
- Model too simple, misses patterns
- Example: linear model for non-linear data

**Variance**: Error from sensitivity to training data
- High variance → overfitting
- Model too complex, learns noise
- Example: deep tree on small dataset

**Irreducible Error**: Noise in data, cannot be reduced

### The Tradeoff

```
Simple Model (Linear)     Complex Model (Deep NN)
High Bias                 Low Bias
Low Variance              High Variance
Underfits                 Overfits
```

**Sweet Spot**: Balance complexity to minimize total error

### Practical Implications

**Detecting High Bias**:
- Low training accuracy
- Low validation accuracy
- Training and validation errors similar

**Solutions**:
- Increase model complexity
- Add more features
- Reduce regularization
- Train longer

**Detecting High Variance**:
- High training accuracy
- Low validation accuracy
- Large gap between train and validation error

**Solutions**:
- Get more training data
- Reduce model complexity
- Add regularization
- Feature selection
- Early stopping

### Learning Curves

Plot training/validation error vs training set size:
- High bias: both errors high, converge quickly
- High variance: large gap, more data helps

---

## 3. OVERFITTING & REGULARIZATION

### Understanding Overfitting

**Signs**:
- Perfect training accuracy, poor test accuracy
- Model memorizes training data
- Doesn't generalize to new data
- Complex decision boundaries

**Causes**:
- Too many parameters relative to data
- Training too long
- Noisy data
- No regularization

### Regularization Techniques

#### L2 Regularization (Ridge)

**Formula**: Loss + λ Σ w²

**Properties**:
- Penalizes large weights
- Weights shrink toward zero (but not exactly zero)
- Closed-form solution exists
- Handles multicollinearity
- Smooth weight distribution

**When to use**: Want all features, prevent large weights

#### L1 Regularization (Lasso)

**Formula**: Loss + λ Σ |w|

**Properties**:
- Penalizes absolute weights
- Produces sparse solutions (weights become exactly zero)
- Automatic feature selection
- No closed-form solution
- Some features completely eliminated

**When to use**: Feature selection, interpretability

#### Elastic Net

**Formula**: Loss + λ₁ Σ |w| + λ₂ Σ w²

**Properties**:
- Combines L1 and L2
- Sparse solutions + grouping effect
- Best of both worlds

**When to use**: Many correlated features, want sparsity

#### Dropout

**Mechanism**: Randomly set neurons to zero during training (probability p)

**Why it works**:
- Prevents co-adaptation of neurons
- Forces redundant representations
- Ensemble effect (training many sub-networks)
- Acts as regularization

**Best practices**:
- p = 0.5 for hidden layers
- p = 0.2 for input layer
- Don't use on output layer
- Scale activations at test time: multiply by (1-p)
- Or use inverted dropout (scale during training)

#### Early Stopping

**Method**: Monitor validation error, stop when it starts increasing

**Implementation**:
- Track best validation error
- Patience parameter: wait N epochs before stopping
- Restore best weights

**Advantages**: Simple, effective, no hyperparameter tuning

#### Data Augmentation

**Purpose**: Artificially increase training data

**Computer Vision**:
- Rotation, flipping, scaling
- Color jittering, brightness
- Random crops, cutout
- Mixup, CutMix

**NLP**:
- Synonym replacement
- Back-translation
- Random insertion/deletion
- Paraphrasing

**Benefits**: Improves generalization, reduces overfitting

### Other Techniques

**Batch Normalization**: Normalizes layer inputs, acts as regularization

**Weight Decay**: Equivalent to L2 in SGD

**Max-norm Constraints**: Constrain weight vector magnitude

**Noise Injection**: Add noise to inputs or weights

---

## 4. LOSS FUNCTIONS

### Regression Losses

#### Mean Squared Error (MSE)

**Formula**: (1/n) Σ (y - ŷ)²

**Properties**:
- Penalizes large errors heavily (squared term)
- Differentiable everywhere
- Sensitive to outliers
- Corresponds to Gaussian likelihood (MLE)

**When to use**: Default for regression, no outliers

#### Mean Absolute Error (MAE)

**Formula**: (1/n) Σ |y - ŷ|

**Properties**:
- Linear penalty
- Robust to outliers
- Not differentiable at zero
- Corresponds to Laplacian likelihood

**When to use**: Data has outliers

#### Huber Loss

**Formula**: 
- |y - ŷ| ≤ δ: 0.5(y - ŷ)²
- |y - ŷ| > δ: δ|y - ŷ| - 0.5δ²

**Properties**:
- Quadratic for small errors (like MSE)
- Linear for large errors (like MAE)
- Robust to outliers
- Differentiable everywhere

**When to use**: Balance between MSE and MAE

#### Log-Cosh Loss

**Formula**: Σ log(cosh(y - ŷ))

**Properties**:
- Smooth approximation of MAE
- Twice differentiable
- Less sensitive to outliers than MSE

### Classification Losses

#### Binary Cross-Entropy

**Formula**: -[y log(ŷ) + (1-y) log(1-ŷ)]

**Properties**:
- For binary classification
- Output: sigmoid activation
- Penalizes confident wrong predictions heavily
- Corresponds to Bernoulli likelihood

**When to use**: Binary classification (spam, fraud)

#### Categorical Cross-Entropy

**Formula**: -Σ y_i log(ŷ_i)

**Properties**:
- For multi-class classification
- Output: softmax activation
- One-hot encoded labels
- Measures KL divergence

**When to use**: Multi-class, mutually exclusive classes

#### Sparse Categorical Cross-Entropy

**Same as categorical but**:
- Labels are integers (not one-hot)
- More memory efficient
- Same gradients

#### Focal Loss

**Formula**: -(1-p)^γ log(p)

**Properties**:
- Addresses class imbalance
- Down-weights easy examples
- Focuses on hard examples
- γ controls focusing (typically 2)

**When to use**: Severe class imbalance (object detection)

#### Hinge Loss

**Formula**: max(0, 1 - y·ŷ)

**Properties**:
- For SVM
- y ∈ {-1, +1}
- Margin-based loss
- Sparse solutions

**When to use**: SVM, maximum margin classification

### Ranking Losses

#### Triplet Loss

**Formula**: max(0, d(a,p) - d(a,n) + margin)

**Components**:
- Anchor, Positive, Negative
- d: distance metric (L2, cosine)
- margin: separation threshold

**When to use**: Face recognition, similarity learning

#### Contrastive Loss

**Formula**: 
- Similar: d²
- Dissimilar: max(0, margin - d)²

**When to use**: Siamese networks, metric learning

---

## 5. EVALUATION METRICS

### Classification Metrics

#### Confusion Matrix

```
                Predicted
              Pos    Neg
Actual Pos    TP     FN
       Neg    FP     TN
```

**Derived Metrics**:

**Accuracy** = (TP + TN) / Total
- Overall correctness
- Misleading with imbalanced data
- Use when: balanced classes, all errors equal cost

**Precision** = TP / (TP + FP)
- Of predicted positives, how many correct
- "How precise are positive predictions?"
- Use when: false positives costly (spam filter)

**Recall (Sensitivity)** = TP / (TP + FN)
- Of actual positives, how many found
- "How complete is positive detection?"
- Use when: false negatives costly (disease detection)

**Specificity** = TN / (TN + FP)
- True negative rate
- Important in medical testing

**F1 Score** = 2 · (Precision · Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- Balances both metrics
- Use when: need balance, imbalanced data

**F-beta Score** = (1 + β²) · (Precision · Recall) / (β² · Precision + Recall)
- β > 1: favor recall
- β < 1: favor precision
- F2: weighs recall higher
- F0.5: weighs precision higher

#### ROC Curve & AUC

**ROC** (Receiver Operating Characteristic):
- Plot: TPR (recall) vs FPR (1-specificity)
- Shows performance across all thresholds
- Diagonal line: random classifier
- Top-left corner: perfect classifier

**AUC** (Area Under Curve):
- Single number summary
- 1.0: perfect
- 0.5: random
- Interpretation: probability model ranks random positive higher than random negative

**When to use**:
- Compare models
- Threshold-independent evaluation
- Balanced or moderately imbalanced data

**Limitations**:
- Optimistic with severe imbalance
- Doesn't show calibration

#### Precision-Recall Curve & AP

**PR Curve**:
- Plot: Precision vs Recall
- Better for imbalanced data than ROC
- Shows trade-off between precision and recall

**Average Precision (AP)**:
- Area under PR curve
- Weighted mean of precisions at each threshold
- Used in object detection (mAP)

**When to use**: Imbalanced data, care about positive class

#### Log Loss (Cross-Entropy)

**Formula**: -(1/n) Σ [y log(p) + (1-y) log(1-p)]

**Properties**:
- Measures probability calibration
- Penalizes confident wrong predictions
- Lower is better

**When to use**: Need probability estimates, not just class labels

#### Matthews Correlation Coefficient (MCC)

**Formula**: (TP·TN - FP·FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]

**Properties**:
- Range: [-1, 1]
- Balanced measure even with imbalanced data
- 1: perfect, 0: random, -1: total disagreement

**When to use**: Imbalanced data, single metric needed

### Regression Metrics

#### R² (Coefficient of Determination)

**Formula**: 1 - (SS_res / SS_tot)

**Interpretation**:
- Proportion of variance explained
- 1: perfect fit
- 0: model no better than mean
- Negative: worse than mean

**Limitations**:
- Can be misleading with non-linear relationships
- Always increases with more features

#### Adjusted R²

**Formula**: 1 - [(1-R²)(n-1)/(n-p-1)]

**Properties**:
- Penalizes additional features
- Better for model comparison
- Use when: comparing models with different features

#### RMSE (Root Mean Squared Error)

**Formula**: √[(1/n) Σ (y - ŷ)²]

**Properties**:
- Same units as target
- Penalizes large errors
- Sensitive to outliers

#### MAPE (Mean Absolute Percentage Error)

**Formula**: (100/n) Σ |y - ŷ| / |y|

**Properties**:
- Scale-independent
- Interpretable (percentage)
- Undefined when y = 0
- Asymmetric (penalizes over-prediction more)

**When to use**: Compare across different scales

### Ranking Metrics

#### NDCG (Normalized Discounted Cumulative Gain)

**DCG**: Σ (2^rel_i - 1) / log₂(i + 1)

**NDCG**: DCG / IDCG (ideal DCG)

**Properties**:
- Considers position and relevance
- Range: [0, 1]
- Higher positions weighted more

**When to use**: Search ranking, recommendation

#### MAP (Mean Average Precision)

**Average Precision**: Mean of precision at each relevant result

**MAP**: Mean of AP across queries

**When to use**: Information retrieval, ranking

#### MRR (Mean Reciprocal Rank)

**Reciprocal Rank**: 1 / (rank of first relevant result)

**MRR**: Mean across queries

**When to use**: Single relevant result per query

---


## 6. FEATURE ENGINEERING

### Numerical Features

#### Scaling & Normalization

**Min-Max Scaling**: x' = (x - min) / (max - min)
- Range: [0, 1]
- Preserves relationships
- Sensitive to outliers
- Use when: bounded range needed, neural networks

**Standardization (Z-score)**: x' = (x - μ) / σ
- Mean = 0, Std = 1
- Not bounded
- Less sensitive to outliers
- Use when: features have different scales, linear models, SVM

**Robust Scaling**: x' = (x - median) / IQR
- Uses median and interquartile range
- Robust to outliers
- Use when: data has outliers

**Log Transform**: x' = log(x + 1)
- Handles skewed distributions
- Reduces impact of outliers
- Use when: exponential relationships, right-skewed data

**Box-Cox Transform**: Finds optimal power transformation
- Generalizes log transform
- Makes data more normal
- Use when: need normality for statistical tests

#### Binning/Discretization

**Equal-width**: Divide range into equal intervals
**Equal-frequency**: Each bin has same number of samples
**Custom**: Domain-specific boundaries

**Benefits**:
- Handles outliers
- Captures non-linear relationships
- Makes model more robust

**Drawbacks**:
- Loss of information
- Arbitrary boundaries

#### Polynomial Features

**Create**: x₁, x₂ → x₁, x₂, x₁², x₂², x₁x₂

**Benefits**:
- Capture non-linear relationships
- Interaction terms

**Drawbacks**:
- Exponential feature growth
- Overfitting risk
- Multicollinearity

### Categorical Features

#### One-Hot Encoding

**Method**: Create binary column for each category

**Example**: Color = {Red, Blue, Green}
- Red: [1, 0, 0]
- Blue: [0, 1, 0]
- Green: [0, 0, 1]

**Pros**: No ordinal assumption, works with all algorithms

**Cons**: High dimensionality with many categories, sparse

**When to use**: Nominal categories, tree-based models

#### Label Encoding

**Method**: Assign integer to each category

**Example**: {Red: 0, Blue: 1, Green: 2}

**Pros**: Compact, memory efficient

**Cons**: Implies ordering (Red < Blue < Green)

**When to use**: Ordinal categories, tree-based models

#### Target Encoding (Mean Encoding)

**Method**: Replace category with mean of target for that category

**Example**: 
- Red appears in samples with target [0, 1, 1] → encode as 0.67
- Blue appears in samples with target [0, 0, 1] → encode as 0.33

**Pros**: Captures relationship with target, handles high cardinality

**Cons**: Risk of overfitting, data leakage

**Best practices**:
- Use cross-validation (encode using other folds)
- Add smoothing: (n·mean + m·global_mean) / (n + m)
- Add noise for regularization

**When to use**: High cardinality, tree-based models

#### Frequency Encoding

**Method**: Replace with frequency of category

**Pros**: Simple, handles high cardinality

**Cons**: Different categories may have same frequency

#### Binary Encoding

**Method**: Convert to binary, split into columns

**Example**: 5 categories → 3 binary columns (2³ = 8 > 5)

**Pros**: More compact than one-hot

**When to use**: High cardinality, need compression

#### Embedding

**Method**: Learn dense vector representation

**Pros**: Captures similarity, compact, powerful

**Cons**: Requires training, more complex

**When to use**: Deep learning, high cardinality (user IDs, product IDs)

### Handling Missing Data

#### Understanding Missingness

**MCAR** (Missing Completely At Random):
- Missingness independent of data
- Safe to remove or impute
- Example: sensor malfunction

**MAR** (Missing At Random):
- Missingness depends on observed data
- Can model missingness
- Example: income missing for unemployed

**MNAR** (Missing Not At Random):
- Missingness depends on unobserved data
- Hardest to handle
- Example: high earners don't report income

#### Imputation Strategies

**Simple Imputation**:
- Mean/Median: for numerical
- Mode: for categorical
- Constant: domain-specific value
- Forward/Backward fill: for time series

**Pros**: Fast, simple
**Cons**: Ignores relationships, reduces variance

**KNN Imputation**:
- Use k nearest neighbors to impute
- Considers feature relationships
- Computationally expensive

**Model-based Imputation**:
- Train model to predict missing values
- Iterative imputation (MICE)
- Most accurate but complex

**Multiple Imputation**:
- Create multiple imputed datasets
- Train model on each
- Combine predictions
- Accounts for uncertainty

**Indicator Variable**:
- Add binary feature: is_missing
- Useful if missingness is informative
- Use with any imputation method

**When to remove**:
- > 50% missing in feature: consider dropping feature
- > 50% missing in sample: consider dropping sample
- MCAR and sufficient data

### Feature Selection

#### Filter Methods

**Correlation**:
- Remove highly correlated features (> 0.9)
- Reduces multicollinearity
- Fast, model-agnostic

**Variance Threshold**:
- Remove low-variance features
- Constant or near-constant features

**Statistical Tests**:
- Chi-square: categorical features
- ANOVA F-test: numerical features
- Mutual information: any type

**Pros**: Fast, model-agnostic
**Cons**: Ignores feature interactions

#### Wrapper Methods

**Forward Selection**:
1. Start with no features
2. Add feature that improves most
3. Repeat until no improvement

**Backward Elimination**:
1. Start with all features
2. Remove feature that hurts least
3. Repeat until performance drops

**Recursive Feature Elimination (RFE)**:
1. Train model
2. Remove least important feature
3. Repeat until desired number

**Pros**: Considers feature interactions
**Cons**: Computationally expensive, overfitting risk

#### Embedded Methods

**L1 Regularization (Lasso)**:
- Automatically zeros out features
- Built into training

**Tree-based Feature Importance**:
- Random Forest, XGBoost
- Importance from splits
- Fast, handles interactions

**Pros**: Efficient, considers interactions
**Cons**: Model-specific

#### Dimensionality Reduction

**PCA** (Principal Component Analysis):
- Linear combinations of features
- Orthogonal components
- Preserves variance

**LDA** (Linear Discriminant Analysis):
- Supervised version of PCA
- Maximizes class separation

**Autoencoders**:
- Neural network compression
- Non-linear relationships

### Feature Crosses

**Definition**: Combine multiple features

**Examples**:
- [latitude, longitude] → geohash
- [hour, day_of_week] → hour_of_week
- [age, income] → age_income_bucket

**Benefits**:
- Capture interactions
- Non-linear relationships
- Domain knowledge

**Drawbacks**:
- Exponential growth
- Sparsity

### Time-based Features

**From Timestamp**:
- Hour, day, month, year
- Day of week, weekend
- Quarter, season
- Time since event
- Cyclical encoding: sin/cos for hour, month

**Lag Features**:
- Previous values: t-1, t-7, t-30
- Rolling statistics: mean, std, min, max
- Exponential moving average

**Domain-specific**:
- Holiday indicators
- Business hours
- Peak times

### Text Features

**Bag of Words**: Count of each word
**TF-IDF**: Term frequency × inverse document frequency
**N-grams**: Sequences of n words
**Word Embeddings**: Word2Vec, GloVe, FastText
**Sentence Embeddings**: BERT, Sentence-BERT

### Feature Engineering Best Practices

1. **Understand domain**: Domain knowledge is crucial
2. **Start simple**: Basic features first
3. **Iterate**: Add complexity gradually
4. **Validate**: Use cross-validation
5. **Monitor**: Track feature importance
6. **Document**: Keep track of transformations
7. **Automate**: Build reproducible pipelines
8. **Test**: Ensure no data leakage

---

## 7. CROSS-VALIDATION

### K-Fold Cross-Validation

**Process**:
1. Split data into k equal folds
2. For each fold:
   - Train on k-1 folds
   - Validate on remaining fold
3. Average results across k folds

**Typical k**: 5 or 10

**Pros**:
- Uses all data for training and validation
- Reduces variance in performance estimate
- More reliable than single train-test split

**Cons**:
- k times more expensive
- Not suitable for time series

**When to use**: Limited data, need robust estimate

### Stratified K-Fold

**Modification**: Preserve class distribution in each fold

**Why**: Ensures each fold is representative

**When to use**: Classification with imbalanced classes

### Leave-One-Out (LOO)

**Process**: k = n (each sample is a fold)

**Pros**: Maximum use of data, deterministic

**Cons**: Very expensive, high variance

**When to use**: Very small datasets (< 100 samples)

### Time Series Cross-Validation

**Problem**: Cannot shuffle time series data (data leakage)

**Solution**: Time-based splits

**Forward Chaining**:
```
Fold 1: train [1:100], test [101:150]
Fold 2: train [1:150], test [151:200]
Fold 3: train [1:200], test [201:250]
```

**Sliding Window**:
```
Fold 1: train [1:100], test [101:150]
Fold 2: train [51:150], test [151:200]
Fold 3: train [101:200], test [201:250]
```

**When to use**: Time series, sequential data

### Group K-Fold

**Problem**: Samples from same group should stay together

**Solution**: Split by groups, not samples

**Example**: Patient data - all samples from same patient in same fold

**When to use**: Hierarchical data, prevent data leakage

### Nested Cross-Validation

**Purpose**: Unbiased model selection and evaluation

**Structure**:
- Outer loop: model evaluation
- Inner loop: hyperparameter tuning

**Process**:
1. Outer fold: split into train and test
2. Inner folds: split train into train and validation
3. Tune hyperparameters on inner folds
4. Evaluate on outer test fold
5. Repeat for all outer folds

**When to use**: Need both model selection and unbiased evaluation

### Cross-Validation Best Practices

1. **Stratify**: For classification
2. **Shuffle**: Unless time series
3. **Same splits**: For fair model comparison
4. **Preprocessing**: Fit on train, transform on validation (avoid leakage)
5. **Sufficient data**: Each fold should be representative
6. **Computational cost**: Balance k with resources
7. **Variance**: Higher k → lower variance, higher cost

---

## 8. GRADIENT DESCENT

### Batch Gradient Descent

**Update**: θ = θ - α ∇J(θ)

**Process**: Use entire dataset for each update

**Pros**:
- Stable convergence
- Guaranteed to converge to minimum (convex) or local minimum (non-convex)
- Smooth trajectory

**Cons**:
- Slow for large datasets
- Stuck in local minima
- Requires entire dataset in memory

**When to use**: Small datasets, convex problems

### Stochastic Gradient Descent (SGD)

**Update**: Use single sample for each update

**Pros**:
- Fast updates
- Can escape local minima (noise helps)
- Online learning possible
- Memory efficient

**Cons**:
- Noisy updates
- Unstable convergence
- Requires learning rate decay
- May not converge exactly

**When to use**: Large datasets, online learning

### Mini-Batch Gradient Descent

**Update**: Use batch of samples (typically 32-256)

**Pros**:
- Balance between batch and stochastic
- Efficient on GPUs (parallel computation)
- Stable convergence
- Faster than batch GD

**Cons**:
- Hyperparameter: batch size

**When to use**: Deep learning (most common)

**Batch size considerations**:
- Larger: more stable, better GPU utilization, may generalize worse
- Smaller: more noise, regularization effect, may generalize better
- Typical: 32, 64, 128, 256

### Momentum

**Update**:
- v = βv + ∇J(θ)
- θ = θ - αv

**Intuition**: Accumulate velocity, accelerate in consistent directions

**Benefits**:
- Faster convergence
- Dampens oscillations
- Helps escape local minima
- Navigates ravines better

**Hyperparameter**: β (typically 0.9)

### Nesterov Accelerated Gradient (NAG)

**Update**:
- v = βv + ∇J(θ - βv)
- θ = θ - αv

**Difference**: Look ahead before computing gradient

**Benefits**: More responsive, better convergence

### AdaGrad

**Update**:
- G = G + (∇J(θ))²
- θ = θ - α/(√G + ε) · ∇J(θ)

**Intuition**: Adapt learning rate per parameter based on history

**Benefits**:
- No manual learning rate tuning
- Good for sparse data
- Different learning rates per parameter

**Drawbacks**:
- Learning rate decays too aggressively
- May stop learning too early

**When to use**: Sparse features, NLP

### RMSprop

**Update**:
- G = βG + (1-β)(∇J(θ))²
- θ = θ - α/(√G + ε) · ∇J(θ)

**Difference from AdaGrad**: Exponential moving average (fixes aggressive decay)

**Benefits**:
- Adapts learning rate
- Works well in practice
- Good for RNNs

**Hyperparameter**: β (typically 0.9)

### Adam (Adaptive Moment Estimation)

**Update**:
- m = β₁m + (1-β₁)∇J(θ)  [first moment]
- v = β₂v + (1-β₂)(∇J(θ))²  [second moment]
- m̂ = m/(1-β₁ᵗ)  [bias correction]
- v̂ = v/(1-β₂ᵗ)  [bias correction]
- θ = θ - α·m̂/(√v̂ + ε)

**Intuition**: Combines momentum and RMSprop

**Benefits**:
- Adaptive learning rates
- Momentum for acceleration
- Bias correction for early iterations
- Works well in practice
- Good default choice

**Hyperparameters**:
- α: 0.001 (default)
- β₁: 0.9 (momentum)
- β₂: 0.999 (RMSprop)
- ε: 1e-8

**Drawbacks**:
- May generalize worse than SGD on some tasks
- Can converge to sharp minima

### AdamW

**Modification**: Decouples weight decay from gradient

**Benefits**: Better regularization, improved generalization

**When to use**: Transformers, modern deep learning

### Learning Rate Scheduling

**Step Decay**: Reduce by factor every N epochs

**Exponential Decay**: α = α₀ · e^(-kt)

**1/t Decay**: α = α₀ / (1 + kt)

**Cosine Annealing**: α = α_min + 0.5(α_max - α_min)(1 + cos(πt/T))

**Warm Restarts**: Periodic resets with cosine annealing

**Warm-up**: Start with small LR, gradually increase

**When to use**: Long training, fine-tuning

### Choosing an Optimizer

**SGD + Momentum**:
- Best final performance
- Requires careful tuning
- Use when: have time to tune, want best results

**Adam**:
- Fast convergence
- Less tuning needed
- Good default
- Use when: quick experiments, transformers

**RMSprop**:
- Good for RNNs
- Use when: recurrent networks

**AdaGrad**:
- Use when: sparse features, NLP

---

## 9. ENSEMBLE METHODS

### Bagging (Bootstrap Aggregating)

**Idea**: Train multiple models on different subsets, average predictions

**Process**:
1. Create bootstrap samples (sample with replacement)
2. Train model on each sample
3. Average predictions (regression) or vote (classification)

**Benefits**:
- Reduces variance
- Prevents overfitting
- Parallelizable

**Example**: Random Forest

### Random Forest

**Idea**: Bagging + random feature selection

**Process**:
1. Create bootstrap samples
2. For each tree:
   - At each split, consider random subset of features
   - Choose best split from subset
3. Average predictions

**Hyperparameters**:
- n_estimators: number of trees (100-1000)
- max_depth: tree depth (None or 10-50)
- max_features: features per split (√n for classification, n/3 for regression)
- min_samples_split: minimum samples to split (2-10)

**Benefits**:
- Handles non-linear relationships
- Feature importance
- Robust to outliers
- Little hyperparameter tuning
- Handles missing values

**Drawbacks**:
- Less interpretable
- Slower prediction
- Memory intensive

**When to use**: Tabular data, baseline model, feature importance

### Boosting

**Idea**: Train models sequentially, each correcting previous errors

**Types**: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost

### AdaBoost

**Process**:
1. Initialize sample weights uniformly
2. For each iteration:
   - Train weak learner on weighted samples
   - Compute error
   - Update sample weights (increase for misclassified)
   - Compute model weight
3. Combine models with weighted vote

**Benefits**:
- Simple
- Less prone to overfitting than other boosting

**Drawbacks**:
- Sensitive to noise and outliers

### Gradient Boosting

**Idea**: Fit new model to residuals of previous model

**Process**:
1. Initialize with constant prediction
2. For each iteration:
   - Compute residuals (negative gradient of loss)
   - Fit new tree to residuals
   - Add to ensemble with learning rate
3. Final prediction: sum of all trees

**Hyperparameters**:
- n_estimators: number of trees
- learning_rate: shrinkage (0.01-0.1)
- max_depth: tree depth (3-10)
- subsample: fraction of samples per tree
- min_samples_split, min_samples_leaf

**Benefits**:
- Powerful
- Handles non-linear relationships
- Feature importance

**Drawbacks**:
- Prone to overfitting
- Requires careful tuning
- Sequential (not parallelizable)

### XGBoost

**Improvements over Gradient Boosting**:
- Regularization (L1, L2)
- Parallel tree construction
- Handling missing values
- Tree pruning
- Built-in cross-validation

**Additional hyperparameters**:
- reg_alpha: L1 regularization
- reg_lambda: L2 regularization
- gamma: minimum loss reduction for split
- colsample_bytree: feature sampling

**When to use**: Kaggle competitions, tabular data, need best performance

### LightGBM

**Improvements**:
- Leaf-wise growth (vs level-wise)
- Histogram-based algorithm
- Faster training
- Lower memory

**When to use**: Large datasets, need speed

### CatBoost

**Improvements**:
- Native categorical feature handling
- Ordered boosting (reduces overfitting)
- Symmetric trees

**When to use**: Many categorical features, less tuning needed

### Stacking

**Idea**: Train meta-model on predictions of base models

**Process**:
1. Split data into train and holdout
2. Train base models on train set
3. Generate predictions on holdout set
4. Train meta-model on holdout predictions
5. Final prediction: meta-model output

**Base models**: Diverse algorithms (RF, XGBoost, Neural Net)

**Meta-model**: Simple (Logistic Regression, Linear Regression)

**Benefits**:
- Combines strengths of different models
- Often wins competitions

**Drawbacks**:
- Complex
- Risk of overfitting
- Computationally expensive

**Best practices**:
- Use diverse base models
- Use cross-validation for meta-features
- Keep meta-model simple

### Blending

**Similar to stacking but**:
- Single holdout set (not cross-validation)
- Simpler, faster
- More prone to overfitting

### Voting

**Hard Voting**: Majority vote (classification)

**Soft Voting**: Average probabilities (classification)

**Averaging**: Mean of predictions (regression)

**Weighted**: Assign weights to models

**When to use**: Simple ensemble, diverse models

---

## 10. CLASS IMBALANCE

### Understanding the Problem

**Definition**: One class significantly outnumbers others

**Examples**:
- Fraud detection: 0.1% fraud
- Disease diagnosis: 1% positive
- Click prediction: 0.5% clicks

**Why it matters**:
- Model biased toward majority class
- High accuracy but poor recall on minority
- Misleading metrics

### Detection

**Signs**:
- High accuracy but low F1
- Model predicts majority class always
- Poor recall on minority class

### Resampling Techniques

#### Oversampling

**Random Oversampling**:
- Duplicate minority samples
- Simple but risk of overfitting

**SMOTE** (Synthetic Minority Over-sampling Technique):
- Create synthetic samples
- Interpolate between minority samples
- k-nearest neighbors approach

**ADASYN** (Adaptive Synthetic Sampling):
- Focus on harder-to-learn samples
- More synthetic samples in difficult regions

**Pros**: Increases minority representation
**Cons**: Overfitting risk, longer training

#### Undersampling

**Random Undersampling**:
- Remove majority samples
- Simple but loses information

**Tomek Links**:
- Remove majority samples close to minority
- Cleans decision boundary

**NearMiss**:
- Select majority samples close to minority
- Preserves important samples

**Pros**: Faster training, balanced dataset
**Cons**: Loses information

#### Combined

**SMOTE + Tomek**: Oversample then clean boundary

**SMOTE + ENN**: Oversample then remove noisy samples

### Algorithm-Level Techniques

#### Class Weights

**Idea**: Penalize misclassification of minority class more

**Implementation**:
- Inverse frequency: w_i = n_samples / (n_classes · n_samples_i)
- Custom weights based on cost

**Pros**: No data modification, works with any algorithm

**Cons**: Requires tuning

#### Threshold Moving

**Idea**: Adjust decision threshold (default 0.5)

**Process**:
1. Train model normally
2. Tune threshold on validation set
3. Optimize for desired metric (F1, recall)

**Pros**: Simple, no retraining

**Cons**: Only for probabilistic models

#### Focal Loss

**Formula**: -(1-p)^γ log(p)

**Idea**: Down-weight easy examples, focus on hard ones

**Pros**: Addresses imbalance during training

**Cons**: Requires custom implementation

### Evaluation Strategies

**Don't use**: Accuracy

**Use**:
- Precision, Recall, F1
- PR curve and Average Precision
- ROC-AUC (if not too imbalanced)
- Confusion matrix
- Cost-sensitive metrics

### Anomaly Detection Approach

**When**: Extreme imbalance (< 0.1% minority)

**Methods**:
- One-class SVM
- Isolation Forest
- Autoencoders (reconstruction error)

**Idea**: Model normal class, flag deviations

### Ensemble Approaches

**Balanced Random Forest**:
- Bootstrap with balanced sampling
- Each tree sees balanced data

**EasyEnsemble**:
- Multiple undersampled subsets
- Train model on each
- Combine predictions

**BalanceCascade**:
- Sequential undersampling
- Remove correctly classified majority samples

### Best Practices

1. **Understand cost**: What's worse - false positive or false negative?
2. **Try multiple approaches**: Combine techniques
3. **Use appropriate metrics**: Not accuracy
4. **Validate carefully**: Stratified CV
5. **Consider domain**: Some imbalance is natural
6. **Collect more data**: If possible, especially minority class
7. **Feature engineering**: Better features help more than resampling
8. **Start simple**: Class weights before complex resampling

### Real-World Strategy

1. **Baseline**: Train without handling imbalance
2. **Class weights**: Easy first step
3. **Resampling**: If weights insufficient
4. **Ensemble**: Combine multiple approaches
5. **Threshold tuning**: Final optimization
6. **Evaluate**: Use appropriate metrics throughout

---

## INTERVIEW TIPS

### Common Questions

1. **Explain bias-variance tradeoff with example**
2. **How do you handle overfitting?**
3. **Which loss function for [specific problem]?**
4. **Explain precision vs recall with real example**
5. **How do you handle class imbalance?**
6. **What's the difference between L1 and L2 regularization?**
7. **When would you use Random Forest vs Gradient Boosting?**
8. **Explain cross-validation and when to use different types**
9. **How do you handle missing data?**
10. **What feature engineering would you do for [specific problem]?**

### Key Takeaways

- **Understand trade-offs**: Every technique has pros/cons
- **Think about data**: Size, quality, distribution matter
- **Consider business context**: Costs of errors, latency requirements
- **Start simple**: Baseline before complex solutions
- **Validate properly**: Avoid data leakage, use appropriate CV
- **Iterate**: Feature engineering often beats complex models
- **Explain clearly**: Use examples, avoid jargon when possible
- **Ask questions**: Clarify requirements before solving


# ML Fundamentals

## Supervised vs Unsupervised Learning

**Q: Explain the difference between supervised and unsupervised learning with examples.**

A: Supervised learning uses labeled data where the target output is known. The model learns to map inputs to outputs. Examples: classification (spam detection), regression (house price prediction).

Unsupervised learning works with unlabeled data to find patterns or structure. Examples: clustering (customer segmentation), dimensionality reduction (PCA), anomaly detection.

## Bias-Variance Tradeoff

**Q: What is the bias-variance tradeoff?**

A: Bias is the error from overly simplistic assumptions in the model (underfitting). High bias means the model misses relevant patterns.

Variance is the error from sensitivity to small fluctuations in training data (overfitting). High variance means the model learns noise.

The tradeoff: reducing bias increases variance and vice versa. Goal is to find the sweet spot that minimizes total error.

## Overfitting and Regularization

**Q: How do you prevent overfitting?**

A: 
- Use more training data
- Regularization (L1/L2): adds penalty term to loss function
- Cross-validation: validate on held-out data
- Early stopping: stop training when validation error increases
- Dropout: randomly disable neurons during training
- Data augmentation: create synthetic training examples
- Reduce model complexity: fewer parameters/layers

## Loss Functions

**Q: What loss functions would you use for different tasks?**

A:
- Binary classification: Binary cross-entropy
- Multi-class classification: Categorical cross-entropy
- Regression: MSE (Mean Squared Error), MAE (Mean Absolute Error), Huber loss
- Ranking: Hinge loss, triplet loss
- Object detection: Combination of classification and localization losses

## Evaluation Metrics

**Q: When would you use precision vs recall?**

A: Precision = TP/(TP+FP) - of predicted positives, how many are correct
Recall = TP/(TP+FN) - of actual positives, how many did we find

Use precision when false positives are costly (spam filter - don't want to mark important emails as spam).

Use recall when false negatives are costly (cancer detection - don't want to miss actual cases).

F1-score balances both: 2 * (precision * recall)/(precision + recall)

## Feature Engineering

**Q: What are common feature engineering techniques?**

A:
- Normalization/Standardization: scale features to similar ranges
- One-hot encoding: convert categorical variables
- Binning: discretize continuous variables
- Polynomial features: create interaction terms
- Log transforms: handle skewed distributions
- Feature crosses: combine multiple features
- Domain-specific features: leverage domain knowledge

## Cross-Validation

**Q: Explain k-fold cross-validation and when to use it.**

A: Split data into k equal folds. Train on k-1 folds, validate on remaining fold. Repeat k times, each fold used once for validation. Average results across folds.

Benefits: better estimate of model performance, uses all data for both training and validation, reduces variance in performance estimate.

Use when: limited data, need robust performance estimate, hyperparameter tuning.

Time series: use time-based splits instead to avoid data leakage.
