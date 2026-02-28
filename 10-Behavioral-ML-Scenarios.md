# Behavioral & ML Scenarios

## Model Performance Issues

**Q: Your model performs well on training data but poorly on test data. What do you do?**

A:

**Diagnosis**: Overfitting

**Solutions**:
1. Get more training data
2. Add regularization (L1/L2, dropout)
3. Reduce model complexity
4. Use data augmentation
5. Early stopping
6. Cross-validation to tune hyperparameters
7. Check for data leakage

**Investigation steps**:
- Plot learning curves (train vs validation loss)
- Analyze error patterns on validation set
- Check feature importance
- Look for data quality issues

## Data Imbalance

**Q: You have a dataset with 99% negative and 1% positive examples. How do you handle it?**

A:

**Approaches**:

1. **Resampling**:
   - Oversample minority class (SMOTE)
   - Undersample majority class
   - Combination of both

2. **Algorithm level**:
   - Use class weights in loss function
   - Focal loss (focuses on hard examples)
   - Anomaly detection approach

3. **Evaluation**:
   - Don't use accuracy
   - Use precision, recall, F1, AUC-ROC
   - Precision-recall curve more informative

4. **Ensemble**:
   - Train multiple models on balanced subsets
   - Combine predictions

**Choose based on**:
- Cost of false positives vs false negatives
- Available compute
- Amount of data

## Feature Selection

**Q: You have 10,000 features. How do you select the most important ones?**

A:

**Methods**:

1. **Filter methods** (fast, model-agnostic):
   - Correlation with target
   - Chi-square test
   - Mutual information
   - Variance threshold

2. **Wrapper methods** (slow, model-specific):
   - Forward selection
   - Backward elimination
   - Recursive feature elimination (RFE)

3. **Embedded methods**:
   - L1 regularization (Lasso)
   - Tree-based feature importance
   - Neural network attention weights

4. **Dimensionality reduction**:
   - PCA
   - Autoencoders

**Strategy**:
- Start with filter methods to remove obviously bad features
- Use embedded methods during training
- Validate with cross-validation
- Consider domain knowledge

## Cold Start Problem

**Q: How do you handle cold start in a recommendation system?**

A:

**For new users**:
- Ask for preferences during onboarding
- Use demographic information
- Show popular items
- Use content-based recommendations
- Explore-exploit strategy

**For new items**:
- Use item features (content-based)
- Show to diverse user sample
- Promote to early adopters
- Use item metadata for similarity

**Hybrid approach**:
- Combine collaborative and content-based
- Multi-armed bandit for exploration
- Transfer learning from similar domains

## Model Debugging

**Q: Your model's accuracy suddenly dropped in production. How do you debug?**

A:

**Investigation steps**:

1. **Check data pipeline**:
   - Data quality issues
   - Schema changes
   - Missing values
   - Feature distribution shift

2. **Compare distributions**:
   - Training vs production data
   - Feature drift detection
   - Label distribution changes

3. **Model issues**:
   - Model version correct?
   - Preprocessing consistent?
   - Serving infrastructure problems?

4. **External factors**:
   - Seasonality
   - User behavior changes
   - Competitor actions
   - System changes

**Actions**:
- Roll back to previous version
- Retrain with recent data
- Update features
- Adjust thresholds
- A/B test fixes

## Handling Missing Data

**Q: How do you handle missing values in your dataset?**

A:

**Strategies**:

1. **Remove**:
   - Drop rows (if few missing)
   - Drop columns (if mostly missing)
   - Only if data is MCAR (Missing Completely At Random)

2. **Imputation**:
   - Mean/median/mode (simple, fast)
   - Forward/backward fill (time series)
   - KNN imputation (use similar samples)
   - Model-based (predict missing values)
   - Multiple imputation (account for uncertainty)

3. **Indicator**:
   - Add binary feature indicating missingness
   - Useful if missingness is informative

4. **Model-based**:
   - Use algorithms that handle missing values (XGBoost)
   - Treat as separate category (for categorical)

**Choose based on**:
- Why data is missing (MCAR, MAR, MNAR)
- Amount of missing data
- Feature importance
- Computational constraints


## Model Not Learning

**Q: Your model's loss is not decreasing during training. What could be wrong?**

A:

**Potential issues**:

1. **Learning rate problems**:
   - Too high: loss oscillates or diverges
   - Too low: learning too slow
   - Solution: learning rate finder, adaptive optimizers (Adam)

2. **Vanishing/exploding gradients**:
   - Very deep networks
   - Poor initialization
   - Solution: batch normalization, residual connections, gradient clipping

3. **Data issues**:
   - Labels incorrect or noisy
   - Features not normalized
   - Data leakage in wrong direction
   - Solution: verify data quality, normalize features

4. **Architecture issues**:
   - Model too simple (underfitting)
   - Wrong activation functions
   - Solution: increase capacity, try different architectures

5. **Implementation bugs**:
   - Loss function incorrect
   - Gradient computation wrong
   - Data loading issues
   - Solution: gradient checking, unit tests, visualize data

**Debugging steps**:
- Start with small dataset (should overfit)
- Check gradients (gradient checking)
- Visualize activations and gradients
- Try simpler model first
- Verify data preprocessing

## Conflicting Metrics

**Q: Your model has high accuracy but low precision. How do you handle this?**

A:

**Understanding the conflict**:
- High accuracy: overall correct predictions
- Low precision: many false positives
- Likely: imbalanced dataset, model predicts majority class

**Analysis**:
1. Check class distribution
2. Look at confusion matrix
3. Understand business impact of FP vs FN

**Solutions**:

1. **Adjust threshold**:
   - Increase threshold to reduce FP
   - Trade recall for precision
   - Find optimal threshold for business metric

2. **Rebalance training**:
   - Class weights
   - Resampling
   - Focal loss

3. **Choose right metric**:
   - If FP costly: optimize precision
   - If FN costly: optimize recall
   - Balance: F1 or F-beta score

4. **Ensemble**:
   - Combine models with different precision-recall trade-offs

**Example**: Spam detection
- High accuracy (99%) but low precision (50%)
- Many legitimate emails marked as spam (FP)
- Solution: increase threshold, optimize for precision

## Feature Engineering Challenges

**Q: You have raw text, images, and tabular data. How do you combine them in one model?**

A:

**Multi-modal learning approach**:

1. **Feature extraction per modality**:
   - Text: BERT embeddings, TF-IDF
   - Images: CNN features (ResNet, EfficientNet)
   - Tabular: normalize, encode categoricals

2. **Fusion strategies**:
   
   **Early fusion**:
   - Concatenate all features
   - Single model processes combined features
   - Simple but may not capture modality-specific patterns
   
   **Late fusion**:
   - Separate models per modality
   - Combine predictions (average, weighted, stacking)
   - Better for modality-specific learning
   
   **Intermediate fusion**:
   - Process each modality separately
   - Combine intermediate representations
   - Cross-modal attention

3. **Architecture**:
```
Text → BERT → [768-dim]
                          → Concatenate → Dense layers → Output
Image → ResNet → [2048-dim]
          