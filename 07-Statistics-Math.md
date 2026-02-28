# Statistics & Math for ML

## Probability Basics

**Q: Explain Bayes' Theorem and its application in ML.**

A: P(A|B) = P(B|A) * P(A) / P(B)

- P(A|B): Posterior probability
- P(B|A): Likelihood
- P(A): Prior probability
- P(B): Evidence

Applications:
- Naive Bayes classifier
- Bayesian inference
- Spam detection: P(spam|words) = P(words|spam) * P(spam) / P(words)

## Maximum Likelihood Estimation

**Q: What is MLE and how is it used?**

A: MLE finds parameter values that maximize the likelihood of observing the data.

For parameter θ: θ_MLE = argmax P(data|θ)

In practice, maximize log-likelihood (easier to compute):
L(θ) = log P(data|θ)

Examples:
- Linear regression: MLE with Gaussian noise → minimize MSE
- Logistic regression: MLE → minimize cross-entropy
- Neural networks: MLE framework for loss functions

## Gradient Descent

**Q: Explain gradient descent variants.**

A:

**Batch GD**: Use entire dataset for each update
- Stable convergence
- Slow for large datasets
- Guaranteed to converge to minimum (convex) or local minimum (non-convex)

**Stochastic GD**: Use single sample for each update
- Fast, can escape local minima
- Noisy updates, unstable
- Requires learning rate decay

**Mini-batch GD**: Use batch of samples
- Balance between batch and stochastic
- Efficient on GPUs
- Most common in practice

Learning rate scheduling: decay over time for better convergence

## Dimensionality Reduction

**Q: Compare PCA and t-SNE.**

A:

**PCA** (Principal Component Analysis):
- Linear transformation
- Finds directions of maximum variance
- Preserves global structure
- Fast, deterministic
- Use for: feature reduction, visualization, noise reduction

**t-SNE** (t-Distributed Stochastic Neighbor Embedding):
- Non-linear transformation
- Preserves local structure (nearby points stay nearby)
- Stochastic, different runs give different results
- Slow for large datasets
- Use for: visualization only (not feature extraction)

Other methods: UMAP (faster than t-SNE), autoencoders (neural network approach)

## Hypothesis Testing

**Q: Explain p-value and statistical significance.**

A: P-value is the probability of observing results at least as extreme as the data, assuming the null hypothesis is true.

- p < 0.05: Reject null hypothesis (statistically significant)
- p ≥ 0.05: Fail to reject null hypothesis

**Type I error (α)**: False positive - reject true null hypothesis
**Type II error (β)**: False negative - fail to reject false null hypothesis
**Power = 1 - β**: Probability of detecting true effect

**Common tests**:
- t-test: Compare means of two groups
- Chi-square test: Test independence of categorical variables
- ANOVA: Compare means of multiple groups
- Mann-Whitney U: Non-parametric alternative to t-test

Applications in ML:
- A/B testing
- Feature selection
- Model comparison
- Validating improvements

## Confusion Matrix

**Q: Explain confusion matrix and derived metrics.**

A:
```
                Predicted
              Pos    Neg
Actual Pos    TP     FN
       Neg    FP     TN
```

**Metrics**:
- Accuracy = (TP + TN) / Total
- Precision = TP / (TP + FP) - of predicted positives, how many correct
- Recall (Sensitivity) = TP / (TP + FN) - of actual positives, how many found
- Specificity = TN / (TN + FP) - true negative rate
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- F-beta: weighted harmonic mean, β > 1 favors recall, β < 1 favors precision

**Use case dependent**:
- Medical diagnosis: high recall (don't miss diseases)
- Spam filter: high precision (don't block important emails)
- Fraud detection: balance based on cost of false positives vs false negatives

## Central Limit Theorem

**Q: What is CLT and why is it important?**

A: The sampling distribution of the mean approaches a normal distribution as sample size increases, regardless of the population distribution.

**Formally**: If X₁, X₂, ..., Xₙ are i.i.d. with mean μ and variance σ², then:
(X̄ - μ) / (σ/√n) → N(0, 1) as n → ∞

**Implications**:
- Can use normal distribution for inference even if data isn't normal
- Confidence intervals for means
- Justifies many statistical tests
- Foundation for hypothesis testing
- Explains why averages are more stable than individual observations

**Requirements**: 
- Independent samples
- Finite variance
- Usually n ≥ 30 is sufficient

## Regularization Math

**Q: Explain L1 vs L2 regularization mathematically.**

A:

**L2 (Ridge)**: Loss + λ Σ w²
- Penalizes large weights quadratically
- Weights shrink toward zero but don't become exactly zero
- Closed-form solution exists: w = (X^T X + λI)^(-1) X^T y
- Handles multicollinearity well
- Differentiable everywhere

**L1 (Lasso)**: Loss + λ Σ |w|
- Penalizes absolute weights
- Produces sparse solutions (some weights exactly zero)
- Feature selection built-in
- No closed-form solution (use coordinate descent, proximal gradient)
- Not differentiable at zero

**Elastic Net**: Loss + λ₁ Σ |w| + λ₂ Σ w²
- Combines L1 and L2
- Balance between sparsity and grouping
- Groups correlated features

**Why L1 gives sparsity**:
- L1 constraint is diamond-shaped (corners at axes)
- L2 constraint is circular (no corners)
- Optimization more likely to hit corners with L1

Choose L1 for feature selection, L2 for better prediction, Elastic Net for both.

## Covariance and Correlation

**Q: Difference between covariance and correlation?**

A:

**Covariance**: Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)]
- Measures linear relationship
- Units depend on X and Y
- Range: (-∞, +∞)
- Positive: variables increase together
- Negative: one increases, other decreases

**Correlation**: ρ = Cov(X,Y) / (σₓ σᵧ)
- Normalized covariance (Pearson correlation)
- Unitless
- Range: [-1, +1]
- -1: perfect negative, 0: no linear relationship, +1: perfect positive
- Measures strength and direction of linear relationship

**Important**: Correlation doesn't imply causation!

**Applications**:
- Feature selection (remove highly correlated features)
- Multicollinearity detection (VIF = 1/(1-R²))
- Exploratory data analysis
- Portfolio optimization

**Other correlation measures**:
- Spearman: rank correlation (monotonic relationships)
- Kendall's tau: rank correlation (ordinal data)

## Probability Distributions

**Q: Explain common probability distributions used in ML.**

A:

**Bernoulli**: Single binary trial
- P(X=1) = p, P(X=0) = 1-p
- Use: binary classification

**Binomial**: Number of successes in n trials
- P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
- Use: multiple binary trials

**Gaussian (Normal)**: Continuous, bell-shaped
- PDF: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
- Parameters: mean μ, variance σ²
- Use: continuous data, CLT, many natural phenomena

**Poisson**: Count of events in fixed interval
- P(X=k) = (λ^k * e^(-λ)) / k!
- Parameter: rate λ
- Use: rare events, count data

**Exponential**: Time between events
- PDF: f(x) = λe^(-λx)
- Memoryless property
- Use: survival analysis, waiting times

**Uniform**: All values equally likely
- PDF: f(x) = 1/(b-a) for x ∈ [a,b]
- Use: random initialization, sampling

**Beta**: Continuous on [0,1]
- Parameters: α, β
- Use: prior for probabilities, Bayesian inference

**Dirichlet**: Multivariate generalization of Beta
- Use: topic modeling, prior for categorical distributions

## Expectation and Variance

**Q: Explain expectation, variance, and their properties.**

A:

**Expectation (Mean)**: E[X] = Σ x * P(X=x) (discrete) or ∫ x * f(x) dx (continuous)

**Properties**:
- Linearity: E[aX + bY] = aE[X] + bE[Y]
- E[X + c] = E[X] + c
- E[cX] = c * E[X]
- E[XY] = E[X]E[Y] if X, Y independent

**Variance**: Var(X) = E[(X - μ)²] = E[X²] - (E[X])²

**Properties**:
- Var(X + c) = Var(X)
- Var(cX) = c² * Var(X)
- Var(X + Y) = Var(X) + Var(Y) if X, Y independent
- Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y) in general

**Standard deviation**: σ = √Var(X)

**Covariance**: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] = E[XY] - E[X]E[Y]

## Conditional Probability

**Q: Explain conditional probability and independence.**

A:

**Conditional probability**: P(A|B) = P(A ∩ B) / P(B)
- Probability of A given B occurred

**Chain rule**: P(A ∩ B) = P(A|B) * P(B) = P(B|A) * P(A)

**Law of total probability**: P(A) = Σ P(A|Bᵢ) * P(Bᵢ)

**Independence**: 
- Events A, B independent if P(A ∩ B) = P(A) * P(B)
- Equivalently: P(A|B) = P(A)
- Random variables X, Y independent if P(X,Y) = P(X) * P(Y)

**Conditional independence**:
- X ⊥ Y | Z means X, Y independent given Z
- P(X,Y|Z) = P(X|Z) * P(Y|Z)
- Important in Bayesian networks, graphical models

## Maximum A Posteriori (MAP)

**Q: Compare MLE and MAP estimation.**

A:

**MLE**: θ_MLE = argmax P(data|θ)
- Maximize likelihood
- No prior on parameters
- Can overfit with limited data

**MAP**: θ_MAP = argmax P(θ|data) = argmax P(data|θ) * P(θ)
- Maximize posterior
- Incorporates prior P(θ)
- Regularization through prior
- Reduces overfitting

**Relationship**:
- MAP with uniform prior = MLE
- MAP with Gaussian prior on weights = L2 regularization
- MAP with Laplace prior on weights = L1 regularization

**Example**:
- Linear regression with Gaussian prior → Ridge regression
- Logistic regression with Laplace prior → L1-regularized logistic regression

## Sampling Methods

**Q: Explain different sampling techniques.**

A:

**Simple random sampling**: Each sample equally likely
- Unbiased
- May not represent rare groups

**Stratified sampling**: Divide into strata, sample from each
- Ensures representation of subgroups
- Reduces variance
- Use when groups have different characteristics

**Importance sampling**: Sample from proposal distribution
- Estimate expectations under target distribution
- Weight samples by importance weights
- Use when target distribution hard to sample from

**Rejection sampling**: Sample from proposal, accept/reject
- Generate samples from complex distributions
- Inefficient if proposal doesn't match target well

**Markov Chain Monte Carlo (MCMC)**:
- Metropolis-Hastings: propose and accept/reject
- Gibbs sampling: sample each variable conditionally
- Use for Bayesian inference, complex posteriors

**Bootstrap**: Resample with replacement
- Estimate sampling distribution
- Confidence intervals
- Model evaluation

## Information Theory

**Q: Explain entropy, cross-entropy, and KL divergence.**

A:

**Entropy**: H(X) = -Σ P(x) log P(x)
- Measures uncertainty/information content
- Higher entropy = more uncertain
- Uniform distribution has maximum entropy
- Use: decision trees (information gain)

**Cross-entropy**: H(P,Q) = -Σ P(x) log Q(x)
- Measures difference between distributions P and Q
- Use: classification loss function
- Binary: -[y log(ŷ) + (1-y) log(1-ŷ)]

**KL Divergence**: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))
- Measures how Q differs from P
- Non-symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
- Always non-negative
- D_KL(P||Q) = H(P,Q) - H(P)
- Use: variational inference, model comparison

**Mutual Information**: I(X;Y) = D_KL(P(X,Y) || P(X)P(Y))
- Measures dependence between X and Y
- I(X;Y) = 0 if X, Y independent
- Use: feature selection

## Linear Algebra for ML

**Q: Explain key linear algebra concepts for ML.**

A:

**Matrix multiplication**: (AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
- Not commutative: AB ≠ BA
- Associative: (AB)C = A(BC)
- Use: neural network forward pass

**Transpose**: (A^T)ᵢⱼ = Aⱼᵢ
- (AB)^T = B^T A^T
- (A^T)^T = A

**Inverse**: AA^(-1) = I
- Only for square, non-singular matrices
- (AB)^(-1) = B^(-1) A^(-1)
- Use: solving linear systems

**Eigenvalues/Eigenvectors**: Av = λv
- λ: eigenvalue, v: eigenvector
- Use: PCA, spectral clustering, stability analysis

**Singular Value Decomposition**: A = UΣV^T
- U, V: orthogonal matrices
- Σ: diagonal matrix of singular values
- Use: dimensionality reduction, matrix approximation, recommender systems

**Norms**:
- L1 norm: ||x||₁ = Σ|xᵢ|
- L2 norm: ||x||₂ = √(Σxᵢ²)
- Frobenius norm: ||A||_F = √(Σᵢⱼ Aᵢⱼ²)

**Positive definite**: x^T A x > 0 for all x ≠ 0
- All eigenvalues positive
- Use: convex optimization, covariance matrices

## Convex Optimization

**Q: Why is convexity important in ML?**

A:

**Convex function**: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y) for λ ∈ [0,1]
- Bowl-shaped
- Any local minimum is global minimum
- Gradient descent guaranteed to converge

**Convex set**: λx + (1-λ)y ∈ S for all x,y ∈ S, λ ∈ [0,1]
- Line segment between any two points in set

**Examples of convex functions**:
- Linear: ax + b
- Quadratic: x^T A x (if A positive semidefinite)
- Exponential: e^x
- Negative entropy: x log x
- Norms: ||x||

**Non-convex in ML**:
- Neural networks (non-convex loss landscape)
- Matrix factorization
- K-means clustering

**Why convexity matters**:
- Guaranteed global optimum
- Efficient algorithms
- Theoretical guarantees
- Easier to analyze

**Convex ML problems**:
- Linear regression
- Logistic regression
- SVM
- Lasso, Ridge regression


## Calculus for ML

**Q: Explain key calculus concepts used in ML.**

A:

**Derivative**: Rate of change
- f'(x) = lim[h→0] (f(x+h) - f(x)) / h
- Gradient: vector of partial derivatives
- Use: optimization, backpropagation

**Chain rule**: (f ∘ g)'(x) = f'(g(x)) * g'(x)
- Essential for backpropagation
- Compute gradients through composed functions

**Partial derivatives**: ∂f/∂x holding other variables constant
- Gradient: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
- Points in direction of steepest ascent

**Taylor series**: f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...
- Local approximation of function
- Use: Newton's method, second-order optimization

**Jacobian**: Matrix of first-order partial derivatives
- J_ij = ∂f_i/∂x_j
- Use: neural network gradients

**Hessian**: Matrix of second-order partial derivatives
- H_ij = ∂²f/∂x_i∂x_j
- Describes curvature
- Use: Newton's method, analyzing critical points

**Gradient descent update**: x_{t+1} = x_t - α∇f(x_t)
- α: learning rate
- Moves in direction of steepest descent

## Statistical Tests

**Q: When to use different statistical tests?**

A:

**t-test**: Compare means of two groups
- Independent samples t-test: different groups
- Paired t-test: same group, before/after
- Assumptions: normality, equal variances
- Use: A/B testing with continuous metric

**Chi-square test**: Test independence of categorical variables
- χ² = Σ(Observed - Expected)² / Expected
- Use: feature selection, contingency tables

**ANOVA**: Compare means of 3+ groups
- F-statistic
- Post-hoc tests for pairwise comparisons
- Use: comparing multiple model variants

**Mann-Whitney U**: Non-parametric alternative to t-test
- Compares distributions
- No normality assumption
- Use: ordinal data, non-normal distributions

**Kolmogorov-Smirnov**: Test if sample from distribution
- Compares empirical CDF to theoretical CDF
- Use: data drift detection, distribution testing

**Shapiro-Wilk**: Test for normality
- Use: check assumptions before parametric tests

**Levene's test**: Test for equal variances
- Use: check homoscedasticity assumption

## Confidence Intervals

**Q: Explain confidence intervals and their interpretation.**

A:

**Definition**: Range that likely contains true parameter value

**For mean**: x̄ ± t_(α/2) * (s/√n)
- x̄: sample mean
- s: sample standard deviation
- n: sample size
- t_(α/2): t-value for confidence level

**Interpretation**:
- 95% CI: If we repeat experiment many times, 95% of intervals contain true parameter
- NOT: 95% probability true parameter in this interval (frequentist view)

**Factors affecting width**:
- Sample size: larger n → narrower CI
- Variability: larger s → wider CI
- Confidence level: higher confidence → wider CI

**Bootstrap CI**:
- Resample with replacement
- Compute statistic on each resample
- Percentile method: use quantiles of bootstrap distribution

**Use cases**:
- Report uncertainty in estimates
- Compare models (overlapping CIs)
- A/B testing (CI for difference)

## Multiple Testing Correction

**Q: Why and how to correct for multiple testing?**

A:

**Problem**: Testing multiple hypotheses increases false positive rate
- If α = 0.05 and test 20 hypotheses, expect 1 false positive
- Family-wise error rate (FWER) increases

**Bonferroni correction**: α_corrected = α / m
- m: number of tests
- Very conservative
- Use: when tests independent

**Holm-Bonferroni**: Step-down procedure
- Less conservative than Bonferroni
- Sort p-values, adjust thresholds

**False Discovery Rate (FDR)**: Expected proportion of false positives
- Benjamini-Hochberg procedure
- Less conservative than FWER control
- Use: feature selection, genomics

**Why it matters in ML**:
- Feature selection: testing many features
- Hyperparameter tuning: multiple configurations
- A/B testing: multiple metrics
- Model comparison: multiple models

## Effect Size

**Q: Why is effect size important beyond p-values?**

A:

**Problem with p-values**:
- Depends on sample size
- Large n → small p-value even for tiny effects
- Doesn't measure practical significance

**Cohen's d**: Standardized mean difference
- d = (μ₁ - μ₂) / σ_pooled
- Small: 0.2, Medium: 0.5, Large: 0.8
- Use: comparing group means

**Pearson's r**: Correlation coefficient
- Measures linear relationship strength
- Small: 0.1, Medium: 0.3, Large: 0.5

**R²**: Proportion of variance explained
- 0 to 1
- Use: regression models

**Odds ratio**: Ratio of odds
- OR > 1: increased odds
- OR < 1: decreased odds
- Use: logistic regression, case-control studies

**Practical significance**:
- Consider both statistical and practical significance
- Small p-value doesn't mean important effect
- Report effect sizes with confidence intervals

## Bias-Variance Decomposition

**Q: Derive the bias-variance decomposition.**

A:

**Expected prediction error**:
E[(y - ŷ)²] = Bias² + Variance + Irreducible Error

**Derivation**:
- y = f(x) + ε, where E[ε] = 0, Var(ε) = σ²
- ŷ = f̂(x) is our estimate

E[(y - ŷ)²] = E[(f(x) + ε - f̂(x))²]
            = E[(f(x) - f̂(x))²] + E[ε²] + 2E[(f(x) - f̂(x))ε]
            = E[(f(x) - f̂(x))²] + σ²  (since E[ε] = 0)

E[(f(x) - f̂(x))²] = E[(f(x) - E[f̂(x)] + E[f̂(x)] - f̂(x))²]
                   = (f(x) - E[f̂(x)])² + E[(E[f̂(x)] - f̂(x))²]
                   = Bias² + Variance

**Components**:
- **Bias²**: (E[f̂(x)] - f(x))² - error from wrong assumptions
- **Variance**: E[(f̂(x) - E[f̂(x)])²] - error from sensitivity to training data
- **Irreducible error**: σ² - noise in data

**Trade-off**:
- Simple models: high bias, low variance
- Complex models: low bias, high variance
- Goal: minimize total error

## Monte Carlo Methods

**Q: Explain Monte Carlo estimation.**

A:

**Principle**: Use random sampling to estimate quantities

**Monte Carlo integration**:
- Estimate E[f(X)] where X ~ p(x)
- Sample x₁, ..., xₙ ~ p(x)
- Estimate: (1/n) Σ f(xᵢ)
- Law of large numbers: converges to true expectation

**Variance reduction**:
- Importance sampling: sample from better distribution
- Control variates: use correlated variable with known expectation
- Antithetic variates: use negatively correlated samples

**Applications in ML**:
- Bayesian inference (MCMC)
- Reinforcement learning (Monte Carlo tree search)
- Dropout (Monte Carlo dropout for uncertainty)
- Estimating gradients (REINFORCE)

**Convergence**: Error decreases as O(1/√n)
- Need 4x samples to halve error
- Dimension-independent (unlike grid methods)

## Scenario-Based Statistics Problems

### Scenario 1: A/B Test Analysis

**Q: You run an A/B test with 10,000 users in each group. Control has 5% conversion, treatment has 5.5% conversion. Is this significant?**

A:

**Setup**:
- n_A = n_B = 10,000
- p_A = 0.05, p_B = 0.055
- Test: H₀: p_A = p_B vs H₁: p_A ≠ p_B

**Two-proportion z-test**:

1. **Pooled proportion**: p̂ = (500 + 550) / 20,000 = 0.0525

2. **Standard error**: SE = √[p̂(1-p̂)(1/n_A + 1/n_B)]
   = √[0.0525 × 0.9475 × (1/10000 + 1/10000)]
   = √[0.0525 × 0.9475 × 0.0002]
   = 0.00315

3. **Test statistic**: z = (p_B - p_A) / SE
   = (0.055 - 0.05) / 0.00315
   = 1.587

4. **P-value**: P(|Z| > 1.587) ≈ 0.112 (two-tailed)

**Conclusion**: Not statistically significant at α = 0.05

**Effect size**: Relative lift = (0.055 - 0.05) / 0.05 = 10%

**Practical considerations**:
- 10% lift may be practically significant
- Consider confidence interval: 0.005 ± 1.96 × 0.00315 = [-0.0012, 0.0112]
- Includes zero, consistent with non-significance
- May need larger sample size or longer test duration

### Scenario 2: Feature Selection with Correlation

**Q: You have 100 features. How do you select the most important ones while avoiding multicollinearity?**

A:

**Approach**:

1. **Correlation with target**:
   - Compute correlation of each feature with target
   - Rank by absolute correlation
   - Select top k features

2. **Remove multicollinearity**:
   - Compute correlation matrix among features
   - If |corr(X_i, X_j)| > threshold (e.g., 0.9):
     - Keep feature with higher correlation to target
     - Remove the other
   
3. **Variance Inflation Factor (VIF)**:
   - VIF_i = 1 / (1 - R²_i)
   - R²_i from regressing X_i on other features
   - Remove features with VIF > 10

4. **Regularization approach**:
   - Use Lasso (L1) for automatic feature selection
   - Non-zero coefficients indicate important features

**Code sketch**:
```python
# Remove highly correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_filtered = X.drop(columns=to_drop)

# Select by correlation with target
correlations = X_filtered.corrwith(y).abs()
top_features = correlations.nlargest(20).index
```

**Validation**:
- Cross-validation to ensure selected features generalize
- Check model performance with selected features

### Scenario 3: Detecting Data Drift

**Q: Your model's performance degraded in production. How do you detect if input distribution changed?**

A:

**Statistical tests**:

1. **Kolmogorov-Smirnov (KS) test**:
   - Compare training and production distributions
   - For each feature:
     - D = max|F_train(x) - F_prod(x)|
     - Reject H₀ if D > critical value
   - Works for continuous features

2. **Chi-square test**:
   - For categorical features
   - Compare frequency distributions
   - χ² = Σ(O - E)² / E

3. **Population Stability Index (PSI)**:
   - PSI = Σ(p_prod - p_train) × ln(p_prod / p_train)
   - PSI < 0.1: no significant change
   - 0.1 < PSI < 0.2: moderate change
   - PSI > 0.2: significant change

**Implementation**:
```python
def psi(expected, actual, bins=10):
    # Bin the data
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins+1))
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi_value = np.sum((actual_percents - expected_percents) * 
                       np.log(actual_percents / expected_percents))
    
    return psi_value
```

**Action**:
- If drift detected: retrain model on recent data
- Monitor continuously
- Set up alerts for significant drift

### Scenario 4: Sample Size Calculation

**Q: How many samples do you need for an A/B test to detect a 5% relative improvement with 80% power?**

A:

**Given**:
- Baseline conversion: p₁ = 0.10
- Relative improvement: 5%
- Treatment conversion: p₂ = 0.105
- Power: 1 - β = 0.80
- Significance level: α = 0.05 (two-tailed)

**Formula** (for proportions):
n = (z_(α/2) + z_β)² × [p₁(1-p₁) + p₂(1-p₂)] / (p₂ - p₁)²

**Calculation**:
- z_(α/2) = 1.96 (for α = 0.05, two-tailed)
- z_β = 0.84 (for power = 0.80)
- p₁(1-p₁) = 0.10 × 0.90 = 0.09
- p₂(1-p₂) = 0.105 × 0.895 = 0.094
- (p₂ - p₁)² = (0.005)² = 0.000025

n = (1.96 + 0.84)² × (0.09 + 0.094) / 0.000025
  = 7.84 × 0.184 / 0.000025
  = 57,715 per group

**Interpretation**:
- Need ~58,000 users per group
- Total: ~116,000 users
- Small effect sizes require large samples

**Factors affecting sample size**:
- Smaller effect → larger n
- Higher power → larger n
- Lower baseline rate → larger n
- More stringent α → larger n

### Scenario 5: Handling Imbalanced Data Statistically

**Q: You have 99% negative, 1% positive samples. How do you evaluate model performance?**

A:

**Problem with accuracy**:
- Model predicting all negative: 99% accuracy
- Misleading metric

**Better metrics**:

1. **Precision-Recall**:
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1 = 2PR / (P + R)
   - Focus on positive class performance

2. **ROC-AUC**:
   - Area under ROC curve
   - Threshold-independent
   - Measures ranking quality

3. **PR-AUC**:
   - Area under Precision-Recall curve
   - Better for imbalanced data than ROC-AUC
   - Focuses on positive class

4. **Matthews Correlation Coefficient**:
   - MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
   - Range: [-1, 1]
   - Balanced measure even with imbalance

**Statistical approach**:
- Stratified sampling: maintain class ratio in train/test
- Weighted loss: weight positive class higher
- Resampling: oversample minority or undersample majority

**Evaluation strategy**:
```python
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Comprehensive evaluation
print(classification_report(y_true, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_true, y_scores)}")
print(f"PR-AUC: {average_precision_score(y_true, y_scores)}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Analyze FP and FN rates
```

### Scenario 6: Bayesian A/B Testing

**Q: Explain Bayesian approach to A/B testing and when to use it.**

A:

**Frequentist vs Bayesian**:

**Frequentist**:
- Fixed but unknown parameter
- P-value: probability of data given H₀
- Binary decision: reject or not

**Bayesian**:
- Parameter is random variable with distribution
- Posterior: probability of parameter given data
- Probability statements about parameters

**Bayesian A/B test**:

1. **Prior**: Beta(α, β) for conversion rate
   - Uniform prior: Beta(1, 1)
   - Informative prior: Beta(α, β) based on historical data

2. **Likelihood**: Binomial(n, p)
   - n trials, k successes

3. **Posterior**: Beta(α + k, β + n - k)
   - Conjugate prior makes this easy

4. **Decision**:
   - P(p_B > p_A | data) = ?
   - Monte Carlo: sample from posteriors, compute proportion

**Example**:
```python
import numpy as np
from scipy.stats import beta

# Data
n_A, k_A = 10000, 500  # Control
n_B, k_B = 10000, 550  # Treatment

# Posteriors (uniform prior)
posterior_A = beta(1 + k_A, 1 + n_A - k_A)
posterior_B = beta(1 + k_B, 1 + n_B - k_B)

# Sample from posteriors
samples_A = posterior_A.rvs(100000)
samples_B = posterior_B.rvs(100000)

# Probability B > A
prob_B_better = np.mean(samples_B > samples_A)
print(f"P(B > A) = {prob_B_better:.3f}")

# Expected lift
expected_lift = np.mean((samples_B - samples_A) / samples_A)
print(f"Expected lift: {expected_lift:.1%}")
```

**Advantages**:
- Direct probability statements
- Incorporates prior knowledge
- Can stop early with confidence
- Handles multiple comparisons naturally

**When to use**:
- Have prior information
- Want probability of improvement
- Sequential testing (early stopping)
- Multiple variants

**Disadvantages**:
- Requires choosing prior
- More complex to implement
- May be harder to explain to stakeholders
