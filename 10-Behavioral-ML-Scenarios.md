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
          


## Real-World ML Scenarios

### Scenario 1: Model Performance Degradation

**Q: Your recommendation model's CTR dropped from 5% to 3% over the past week. Walk through your debugging process.**

A:

**Step 1: Verify the issue**
- Check if it's a measurement error
- Verify logging/tracking is working
- Compare across different segments (mobile/desktop, regions)
- Check if it's gradual or sudden

**Step 2: Data pipeline investigation**
- Schema changes in input data
- Missing features or null values increased
- Data source changes
- ETL pipeline failures
- Feature computation bugs

**Step 3: Distribution analysis**
- Compare feature distributions (training vs current)
- Check for data drift (KS test, PSI)
- Identify which features shifted most
- Analyze user behavior changes

**Step 4: Model investigation**
- Verify correct model version deployed
- Check model serving infrastructure
- Latency issues causing timeouts
- Preprocessing consistency
- Model staleness (when last trained)

**Step 5: External factors**
- Seasonality (holidays, events)
- Competitor actions
- Product changes (UI/UX updates)
- Marketing campaigns
- User base composition changes

**Step 6: Segment analysis**
- Break down by user segments
- Identify which segments affected most
- New users vs returning users
- Geographic differences

**Actions taken**:
1. Immediate: Rollback to previous model version if infrastructure issue
2. Short-term: Retrain on recent data if data drift
3. Long-term: Improve monitoring, add drift detection, update features

**Communication**:
- Alert stakeholders immediately
- Provide daily updates
- Share root cause analysis
- Document learnings

### Scenario 2: Conflicting Metrics

**Q: Your model has 95% accuracy but stakeholders are unhappy. What do you investigate?**

A:

**Understanding the disconnect**:

1. **Is accuracy the right metric?**
   - Imbalanced classes? (95% accuracy by predicting majority class)
   - Check precision, recall, F1
   - Business cares about different metric (revenue, user satisfaction)

2. **Segment performance**:
   - Model may perform poorly on important segments
   - VIP users, high-value transactions
   - Specific product categories
   - Geographic regions

3. **Error analysis**:
   - What types of errors is model making?
   - Are false positives or false negatives more costly?
   - Qualitative review of misclassifications
   - Pattern in errors (systematic bias)

4. **User experience**:
   - Model latency too high
   - Predictions not actionable
   - UI/UX issues in presenting predictions
   - Lack of explainability

5. **Business context**:
   - Model solves wrong problem
   - Misalignment on success criteria
   - Expectations not set properly
   - Comparison to previous solution

**Example scenario**: Fraud detection
- 95% accuracy sounds good
- But if fraud is 1%, predicting "no fraud" always gives 99% accuracy
- Stakeholders care about catching fraud (recall)
- Need to optimize for F1 or recall at specific precision

**Resolution**:
- Align on correct success metrics upfront
- Regular stakeholder communication
- Show multiple metrics (accuracy, precision, recall, business KPIs)
- A/B test to measure real impact

### Scenario 3: Limited Training Data

**Q: You have only 1,000 labeled examples but need to build a production model. What's your approach?**

A:

**Strategies**:

1. **Transfer learning**:
   - Use pre-trained model (BERT for NLP, ResNet for vision)
   - Fine-tune on your small dataset
   - Often achieves good performance with limited data

2. **Data augmentation**:
   - Synthetic data generation
   - Image: rotation, flip, crop, color jitter
   - Text: back-translation, paraphrasing, synonym replacement
   - Careful not to change labels

3. **Semi-supervised learning**:
   - Use unlabeled data (usually abundant)
   - Self-training: predict on unlabeled, add confident predictions
   - Consistency regularization
   - Pseudo-labeling

4. **Active learning**:
   - Strategically select which samples to label
   - Label most informative examples
   - Uncertainty sampling, query-by-committee
   - Maximize learning per labeled example

5. **Few-shot learning**:
   - Meta-learning approaches
   - Siamese networks, prototypical networks
   - Learn to learn from few examples

6. **Simpler models**:
   - Start with logistic regression, random forest
   - Fewer parameters = less overfitting
   - Strong regularization (L1, L2)
   - Ensemble of simple models

7. **Feature engineering**:
   - Domain knowledge to create informative features
   - Reduces need for model to learn from scratch
   - Feature selection to avoid overfitting

8. **Get more labels**:
   - Crowdsourcing (Amazon MTurk)
   - Weak supervision (Snorkel)
   - Programmatic labeling with rules
   - Expert labeling for critical examples

**Validation strategy**:
- Careful cross-validation (stratified k-fold)
- Hold out test set (don't touch until final evaluation)
- Monitor for overfitting closely

**Example approach**:
1. Start with pre-trained model + fine-tuning
2. Apply data augmentation
3. Use active learning to label 500 more examples strategically
4. Ensemble multiple models
5. Achieve reasonable performance with 1,500 total labels

### Scenario 4: Model Fairness Issue

**Q: Your hiring model is flagged for gender bias. How do you address this?**

A:

**Step 1: Verify the bias**
- Measure performance across demographic groups
- Compute fairness metrics:
  - Demographic parity: equal positive rate
  - Equal opportunity: equal TPR
  - Equalized odds: equal TPR and FPR
- Statistical significance testing
- Intersectional analysis (gender × race, etc.)

**Step 2: Identify source**
- Historical bias in training data
- Proxy features (zip code → race)
- Measurement bias (labels biased)
- Representation bias (underrepresented groups)
- Aggregation bias (one model for all groups)

**Step 3: Mitigation strategies**

**Pre-processing**:
- Reweighting: oversample underrepresented groups
- Resampling: balance training data
- Data augmentation for minority groups
- Remove biased features (but may not solve proxy issue)

**In-processing**:
- Fairness constraints during training
- Adversarial debiasing
- Multi-task learning (predict outcome + demographic)
- Regularization for fairness

**Post-processing**:
- Adjust decision thresholds per group
- Calibration per group
- Reject option (defer uncertain cases)

**Step 4: Validation**
- Test on held-out data
- Measure fairness metrics
- Check for fairness-accuracy tradeoff
- Ensure no new biases introduced

**Step 5: Organizational response**
- Transparency: document bias and mitigation
- Human oversight: human-in-the-loop for decisions
- Regular audits: monitor fairness over time
- Diverse team: include diverse perspectives
- Ethical review: ethics committee approval

**Legal/Compliance**:
- Consult legal team
- Ensure compliance with regulations (GDPR, EEOC)
- Document decision-making process
- Prepare for audits

**Communication**:
- Transparent with stakeholders
- Explain limitations
- Set appropriate expectations
- Ongoing monitoring and reporting

**Example resolution**:
- Remove gender feature and proxies
- Retrain with balanced data
- Apply fairness constraints
- Achieve demographic parity within 5%
- Implement ongoing monitoring
- Human review for all positive predictions

### Scenario 5: Real-Time Prediction Latency

**Q: Your model needs to make predictions in <50ms but currently takes 200ms. How do you optimize?**

A:

**Step 1: Profile and identify bottlenecks**
- Measure each component:
  - Feature fetching: ?ms
  - Preprocessing: ?ms
  - Model inference: ?ms
  - Post-processing: ?ms
- Identify the slowest parts

**Step 2: Model optimization**

**Model compression**:
- Quantization: FP32 → INT8 (4x smaller, 2-4x faster)
- Pruning: remove unimportant weights
- Knowledge distillation: train smaller student model
- Architecture search: find efficient architecture

**Inference optimization**:
- Batch predictions (if applicable)
- TensorRT, ONNX Runtime
- Hardware acceleration (GPU, TPU)
- Model compilation (TorchScript, TF Lite)

**Simpler model**:
- Replace deep model with shallow model
- Ensemble → single model
- Neural network → gradient boosting
- Trade accuracy for speed (if acceptable)

**Step 3: Feature optimization**

**Precomputation**:
- Compute features offline
- Store in fast key-value store (Redis)
- Update periodically

**Feature selection**:
- Remove slow-to-compute features
- Keep only most important features
- Feature importance analysis

**Parallel fetching**:
- Fetch features in parallel
- Async I/O
- Connection pooling

**Step 4: Infrastructure optimization**

**Caching**:
- Cache predictions for common inputs
- Cache intermediate computations
- Cache features
- TTL based on staleness tolerance

**Load balancing**:
- Distribute load across servers
- Auto-scaling
- Request queuing

**Serving optimization**:
- Reduce network latency
- Co-locate services
- Use faster serialization (Protocol Buffers)

**Step 5: Algorithmic changes**

**Two-stage approach**:
- Fast model for initial filtering
- Slow model for top candidates
- Example: retrieve 1000 → rank top 100

**Approximate methods**:
- Approximate nearest neighbors (FAISS)
- Approximate inference
- Early stopping

**Example optimization path**:
1. Profile: 50ms features, 150ms inference
2. Precompute features: 50ms → 5ms
3. Quantize model: 150ms → 40ms
4. Total: 45ms ✓

**Validation**:
- Measure latency at p50, p95, p99
- Ensure accuracy not degraded significantly
- A/B test in production
- Monitor under load

### Scenario 6: Stakeholder Wants Explainability

**Q: Business stakeholders want to understand why your model makes certain predictions. How do you provide explainability?**

A:

**Understanding the need**:
- Regulatory requirement (GDPR, fair lending)
- Build trust with users
- Debug model behavior
- Identify biases
- Business insights

**Approaches by model type**:

**Interpretable models** (inherently explainable):
- Linear/Logistic regression: coefficient values
- Decision trees: tree visualization, rules
- Rule-based systems: explicit rules
- GAMs (Generalized Additive Models): feature contributions

**Complex models** (need explanation methods):

**Global explanations** (overall model behavior):

1. **Feature importance**:
   - Permutation importance
   - SHAP feature importance
   - Gain/split importance (trees)
   - Shows which features matter most

2. **Partial dependence plots**:
   - Show effect of feature on prediction
   - Marginalize over other features
   - Visualize relationships

3. **Surrogate models**:
   - Train interpretable model to mimic complex model
   - Decision tree approximation
   - Linear approximation

**Local explanations** (individual predictions):

1. **SHAP (SHapley Additive exPlanations)**:
   - Feature contribution to specific prediction
   - Based on game theory
   - Consistent and accurate
   - Works for any model

2. **LIME (Local Interpretable Model-agnostic Explanations)**:
   - Approximate model locally
   - Perturb input, observe output
   - Fit linear model locally
   - Fast, intuitive

3. **Attention weights** (neural networks):
   - Show which inputs model focused on
   - Visualize attention maps
   - Transformers, attention-based models

4. **Counterfactual explanations**:
   - "If X changed to Y, prediction would change"
   - Actionable insights
   - What-if analysis

**Implementation example**:

```python
import shap

# Train model
model.fit(X_train, y_train)

# Create explainer
explainer = shap.TreeExplainer(model)  # For tree models
# explainer = shap.KernelExplainer(model.predict, X_train)  # Model-agnostic

# Explain prediction
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)  # Global
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])  # Local
```

**Presentation to stakeholders**:

1. **Feature importance dashboard**:
   - Top 10 most important features
   - Bar chart visualization
   - Update regularly

2. **Individual prediction explanations**:
   - Show top 5 contributing features
   - Positive/negative contributions
   - Comparison to average

3. **Natural language explanations**:
   - "Prediction: High risk"
   - "Because: High transaction amount ($5000), new merchant, unusual location"
   - "Similar to: 85% of high-risk transactions"

4. **Interactive tools**:
   - What-if analysis
   - Feature adjustment sliders
   - See prediction change in real-time

**Validation**:
- Explanations should be consistent
- Align with domain knowledge
- User studies for comprehension
- Compare multiple explanation methods

**Trade-offs**:
- Accuracy vs interpretability
- Global vs local explanations
- Computational cost
- Explanation complexity

### Scenario 7: Handling Concept Drift

**Q: Your model was trained on 2022 data. It's now 2024 and performance is degrading. How do you handle concept drift?**

A:

**Types of drift**:

1. **Covariate shift**: P(X) changes, P(Y|X) stays same
   - Example: User demographics change
   - Solution: Reweight training data

2. **Concept drift**: P(Y|X) changes
   - Example: What constitutes spam evolves
   - Solution: Retrain model

3. **Prior probability shift**: P(Y) changes
   - Example: Fraud rate increases
   - Solution: Adjust thresholds

**Detection methods**:

1. **Performance monitoring**:
   - Track accuracy, precision, recall over time
   - Alert when drops below threshold
   - Compare to baseline

2. **Statistical tests**:
   - Kolmogorov-Smirnov test (feature distributions)
   - Population Stability Index (PSI)
   - Chi-square test (categorical features)

3. **Model-based detection**:
   - Train drift detector
   - Predict if sample is from training or current distribution
   - High accuracy → drift detected

**Mitigation strategies**:

1. **Periodic retraining**:
   - Retrain weekly/monthly
   - Use recent data
   - Automated pipeline

2. **Online learning**:
   - Update model incrementally
   - Learn from new data continuously
   - Careful with catastrophic forgetting

3. **Ensemble with time decay**:
   - Combine multiple models
   - Weight recent models higher
   - Gradually phase out old models

4. **Adaptive models**:
   - Meta-learning
   - Transfer learning
   - Domain adaptation

5. **Feature engineering**:
   - Add time-aware features
   - Trend features
   - Seasonality features

**Implementation**:

```python
# Drift detection
from scipy.stats import ks_2samp

def detect_drift(train_data, prod_data, threshold=0.05):
    drifted_features = []
    for feature in train_data.columns:
        statistic, p_value = ks_2samp(train_data[feature], prod_data[feature])
        if p_value < threshold:
            drifted_features.append(feature)
    return drifted_features

# Retraining strategy
def should_retrain(current_performance, baseline, threshold=0.05):
    return current_performance < baseline * (1 - threshold)

# Automated retraining
if should_retrain(current_f1, baseline_f1):
    # Fetch recent data
    recent_data = get_data(last_n_days=90)
    
    # Retrain model
    model.fit(recent_data)
    
    # Validate
    val_score = evaluate(model, validation_data)
    
    # Deploy if better
    if val_score > current_performance:
        deploy(model)
```

**Best practices**:
- Monitor continuously
- Automate detection and retraining
- Keep multiple model versions
- Gradual rollout of new models
- Maintain training data pipeline
- Document drift incidents

### Scenario 8: Cross-Functional Collaboration

**Q: You're working with product managers, engineers, and business stakeholders on an ML project. How do you ensure effective collaboration?**

A:

**Communication strategies**:

1. **Speak their language**:
   - Product: user impact, features, roadmap
   - Engineering: latency, scalability, infrastructure
   - Business: ROI, KPIs, revenue impact
   - Avoid jargon, use analogies

2. **Set expectations**:
   - ML is iterative, not deterministic
   - Timelines for experimentation
   - Accuracy limitations
   - Data requirements

3. **Regular updates**:
   - Weekly syncs
   - Demo progress
   - Share metrics
   - Blockers and asks

**Alignment**:

1. **Define success metrics upfront**:
   - Business metric (revenue, engagement)
   - ML metric (accuracy, AUC)
   - Ensure alignment
   - Get buy-in from all stakeholders

2. **Prioritization**:
   - Impact vs effort matrix
   - Focus on high-impact projects
   - Quick wins for momentum
   - Long-term strategic bets

3. **Trade-offs**:
   - Accuracy vs latency
   - Complexity vs interpretability
   - Cost vs performance
   - Make trade-offs explicit

**Collaboration practices**:

1. **Shared documentation**:
   - Project goals and requirements
   - Data dictionary
   - Model documentation
   - Decision log

2. **Cross-functional reviews**:
   - Design reviews
   - Code reviews
   - Model reviews
   - Include diverse perspectives

3. **Experimentation culture**:
   - A/B testing framework
   - Fail fast, learn quickly
   - Share learnings
   - Celebrate experiments, not just successes

**Handling disagreements**:

1. **Data-driven decisions**:
   - Run experiments
   - Let data decide
   - A/B test competing approaches

2. **Escalation path**:
   - Clear decision-makers
   - Escalate when stuck
   - Document decisions

3. **Compromise**:
   - Find middle ground
   - Phased approach
   - MVP first, iterate

**Example scenario**:

**Situation**: Product wants 99% accuracy, you estimate 90% is realistic

**Approach**:
1. Explain accuracy limitations with current data
2. Show accuracy vs cost curve
3. Propose 90% accuracy MVP
4. Plan for improvement (more data, better features)
5. Demonstrate business impact of 90% vs 99%
6. Get agreement on 90% target with improvement roadmap

**Building trust**:
- Deliver on commitments
- Be transparent about challenges
- Admit when you don't know
- Give credit to others
- Seek feedback

## Behavioral Interview Questions

### Tell Me About a Time...

**Q: Tell me about a time when your model failed in production.**

**STAR format** (Situation, Task, Action, Result):

**Situation**: Deployed fraud detection model that had 95% accuracy in offline testing.

**Task**: Model started flagging 30% of legitimate transactions as fraud in production, causing customer complaints.

**Action**:
1. Immediately rolled back to previous model
2. Investigated root cause: training data was from 6 months ago, user behavior had changed
3. Analyzed production data distribution vs training data
4. Retrained model on recent 3 months of data
5. Added data drift monitoring
6. Implemented gradual rollout (5% → 25% → 100%)

**Result**: 
- New model reduced false positives by 80%
- Implemented automated retraining pipeline
- Set up alerts for data drift
- Learned importance of fresh training data and monitoring

**Q: Describe a time you had to explain a complex ML concept to non-technical stakeholders.**

**Situation**: Needed to explain why our recommendation model couldn't achieve 100% accuracy.

**Task**: Product manager expected perfect recommendations, didn't understand ML limitations.

**Action**:
1. Used analogy: "Like predicting weather, we can be accurate but not perfect"
2. Showed examples of ambiguous cases where even humans disagree
3. Demonstrated accuracy vs cost curve
4. Explained diminishing returns (90% → 95% requires 10x more data)
5. Focused on business impact: 85% accuracy still drives 20% more engagement

**Result**:
- Stakeholder understood limitations
- Agreed on 85% accuracy target
- Focused on business metrics instead of accuracy
- Built trust through transparency

**Q: Tell me about a time you disagreed with your team about a technical approach.**

**Situation**: Team wanted to use complex deep learning model, I advocated for simpler gradient boosting.

**Task**: Decide on model architecture for fraud detection system.

**Action**:
1. Proposed running experiment: both approaches in parallel
2. Defined success criteria: accuracy, latency, interpretability
3. Implemented both models
4. Compared on held-out test set
5. Measured inference latency
6. Evaluated explainability for compliance

**Result**:
- Gradient boosting achieved similar accuracy (94% vs 95%)
- 10x faster inference (10ms vs 100ms)
- Better explainability for regulatory compliance
- Team agreed on gradient boosting
- Learned value of experimentation over debate

**Q: Describe a challenging ML problem you solved.**

**Situation**: E-commerce company had 99% negative, 1% positive class imbalance for fraud detection.

**Task**: Build model that catches fraud without too many false positives.

**Action**:
1. Analyzed cost of false positives vs false negatives
2. Used SMOTE for oversampling minority class
3. Implemented focal loss to focus on hard examples
4. Ensemble of models trained on different balanced subsets
5. Tuned threshold based on business cost function
6. Added manual review queue for borderline cases

**Result**:
- Achieved 85% recall at 95% precision
- Reduced fraud losses by 60%
- Kept false positive rate under 2%
- Saved company $5M annually

### Strengths and Weaknesses

**Q: What's your biggest strength as an ML engineer?**

**Answer**: Problem-solving and end-to-end thinking.

**Example**: 
- Don't just build models, understand business problem
- Consider entire pipeline: data → model → deployment → monitoring
- Recent project: recommendation system
  - Understood business goal (increase engagement)
  - Designed data pipeline for real-time features
  - Built model with latency constraints
  - Implemented A/B testing framework
  - Set up monitoring and retraining
- Result: 15% increase in user engagement

**Q: What's an area you're working to improve?**

**Answer**: Communication with non-technical stakeholders.

**Example**:
- Early in career, used too much technical jargon
- Stakeholders didn't understand trade-offs
- Led to misaligned expectations
- Now working on:
  - Using analogies and visualizations
  - Focusing on business impact
  - Regular check-ins for alignment
- Recent improvement: Successfully explained model limitations to product team, aligned on realistic goals

### Motivation and Goals

**Q: Why do you want to work at [Company]?**

**Structure**:
1. Company's mission/products resonate
2. Technical challenges align with interests
3. Team/culture fit
4. Growth opportunities

**Example** (for Meta):
"I'm excited about Meta's scale and impact. Working on recommendation systems that serve billions of users is incredibly challenging and impactful. I'm particularly interested in [specific product/team], and I admire Meta's culture of moving fast and data-driven decision making. I see opportunities to grow in [specific area] and contribute to [specific initiative]."

**Q: Where do you see yourself in 5 years?**

**Answer**: 
- Technical growth: Deep expertise in [specific area]
- Leadership: Mentoring junior engineers, leading projects
- Impact: Working on high-impact problems at scale
- Continuous learning: Staying current with ML advances

**Example**:
"In 5 years, I see myself as a senior/staff ML engineer, leading complex ML projects from conception to production. I want to mentor junior engineers and contribute to technical strategy. I'm particularly interested in [specific area like recommendation systems, NLP, etc.] and want to become a domain expert. I also want to contribute to the ML community through open source and publications."

## Key Takeaways

**For technical scenarios**:
- Use structured approach (identify, analyze, solve, validate)
- Consider multiple solutions
- Discuss trade-offs
- Think about production implications
- Mention monitoring and iteration

**For behavioral questions**:
- Use STAR format
- Be specific with examples
- Show impact with metrics
- Demonstrate learning
- Be honest about failures

**General tips**:
- Think out loud
- Ask clarifying questions
- Consider edge cases
- Discuss scalability
- Show business awareness
- Demonstrate collaboration skills
