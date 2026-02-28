# ML Engineering & Production

## Model Deployment

**Q: Describe strategies for deploying ML models to production.**

A:

**Deployment patterns**:

1. **Batch prediction**:
   - Pre-compute predictions offline
   - Store in database/cache
   - Low latency, no real-time compute
   - Use when: recommendations, rankings

2. **Online prediction**:
   - Real-time inference on request
   - Model served via API
   - Higher latency, fresh predictions
   - Use when: fraud detection, personalization

3. **Edge deployment**:
   - Model runs on device
   - No network latency, privacy benefits
   - Limited compute, need model compression
   - Use when: mobile apps, IoT

**Serving infrastructure**:
- TensorFlow Serving, TorchServe, ONNX Runtime
- REST API or gRPC
- Load balancing and auto-scaling
- Model versioning and rollback
- A/B testing framework

## Model Optimization

**Q: How do you optimize models for production?**

A:

**Quantization**:
- Reduce precision (FP32 → INT8)
- 4x smaller, 2-4x faster
- Minimal accuracy loss with post-training quantization
- Quantization-aware training for better accuracy

**Pruning**:
- Remove unimportant weights/neurons
- Structured pruning: remove entire channels
- Unstructured pruning: remove individual weights
- Iterative pruning + fine-tuning

**Knowledge distillation**:
- Train small student model to mimic large teacher
- Student learns from teacher's soft predictions
- Maintains accuracy with fewer parameters
- DistilBERT, MobileNet examples

**Model compression**:
- Low-rank factorization
- Weight sharing
- Huffman encoding

**Hardware optimization**:
- Use GPU/TPU for inference
- Batch requests together
- ONNX for cross-platform optimization
- TensorRT for NVIDIA GPUs

## Feature Store

**Q: What is a feature store and why use it?**

A: Centralized platform for storing, managing, and serving features

**Benefits**:
- Feature reuse across teams/models
- Consistent features for training and serving
- Avoid training-serving skew
- Version control for features
- Monitoring and lineage tracking

**Components**:
- Offline store: historical features for training (data warehouse)
- Online store: low-latency features for serving (key-value store)
- Feature registry: metadata, schemas, documentation
- Transformation engine: compute features from raw data

**Examples**: Feast, Tecton, AWS SageMaker Feature Store

**Use cases**:
- Point-in-time correct features for training
- Real-time feature computation
- Feature sharing across models
- Backfilling historical features

## Model Monitoring

**Q: What should you monitor in production ML systems?**

A:

**Model performance**:
- Prediction accuracy/error metrics
- Latency (p50, p95, p99)
- Throughput (QPS)
- Error rates and exceptions

**Data quality**:
- Feature distributions (detect drift)
- Missing values
- Out-of-range values
- Schema violations

**Data drift**:
- Covariate shift: input distribution changes
- Concept drift: relationship between X and Y changes
- Use statistical tests: KS test, PSI (Population Stability Index)

**Model drift**:
- Performance degradation over time
- Compare online metrics to baseline
- Trigger retraining when drift detected

**Business metrics**:
- User engagement
- Revenue impact
- A/B test results

**Alerting**:
- Set thresholds for critical metrics
- Automated alerts for anomalies
- Dashboards for visualization

## A/B Testing

**Q: How do you A/B test ML models?**

A:

**Setup**:
- Control: existing model
- Treatment: new model
- Random assignment of users
- Sufficient sample size for statistical power

**Metrics**:
- Primary: business metric (CTR, conversion, revenue)
- Secondary: model metrics (accuracy, latency)
- Guardrail: ensure no harm (error rate, user satisfaction)

**Analysis**:
- Statistical significance: t-test, chi-square
- Confidence intervals
- Multiple testing correction (Bonferroni)
- Segment analysis: check for heterogeneous effects

**Challenges**:
- Network effects: users influence each other
- Novelty effect: temporary behavior change
- Long-term vs short-term impact
- Interaction effects between experiments

**Best practices**:
- Run for sufficient duration (1-2 weeks minimum)
- Check for sample ratio mismatch
- Monitor throughout experiment
- Document results and learnings

## ML Pipeline

**Q: Design an end-to-end ML pipeline.**

A:

**Components**:

1. **Data ingestion**:
   - Batch: scheduled jobs (Airflow, Prefect)
   - Streaming: Kafka, Kinesis
   - Data validation and quality checks

2. **Feature engineering**:
   - Transform raw data to features
   - Feature store for storage
   - Versioning and lineage

3. **Training**:
   - Experiment tracking (MLflow, Weights & Biases)
   - Hyperparameter tuning
   - Distributed training for large models
   - Model versioning and registry

4. **Evaluation**:
   - Offline metrics on validation set
   - Compare to baseline
   - Error analysis
   - Model approval gate

5. **Deployment**:
   - Containerization (Docker)
   - Orchestration (Kubernetes)
   - Gradual rollout (canary, blue-green)
   - A/B testing

6. **Monitoring**:
   - Model performance
   - Data drift
   - System health
   - Automated retraining triggers

**Tools**: Kubeflow, SageMaker, Vertex AI, MLflow

**Best practices**:
- Automate everything
- Version control (code, data, models)
- Reproducibility
- Testing at each stage
- Documentation


## Training-Serving Skew

**Q: What is training-serving skew and how do you prevent it?**

A:

**Definition**: Discrepancy between training and serving environments causing performance degradation

**Types**:

1. **Data skew**:
   - Different feature distributions
   - Missing features in production
   - Different data sources

2. **Code skew**:
   - Different preprocessing logic
   - Library version mismatches
   - Numerical precision differences

3. **Schema skew**:
   - Feature type mismatches
   - Different feature names
   - Schema evolution

**Prevention**:

1. **Unified preprocessing**:
   - Same code for training and serving
   - Package preprocessing in model artifact
   - TensorFlow Transform, Scikit-learn pipelines

2. **Feature store**:
   - Consistent feature computation
   - Same features for training and serving
   - Point-in-time correctness

3. **Validation**:
   - Schema validation (TensorFlow Data Validation)
   - Feature distribution monitoring
   - Shadow mode testing

4. **Testing**:
   - Integration tests with production data
   - Compare training and serving outputs
   - Canary deployments

## Model Versioning and Registry

**Q: How do you manage model versions in production?**

A:

**Model registry components**:
- Model artifacts (weights, architecture)
- Metadata (metrics, hyperparameters, training data)
- Lineage (code version, data version, parent models)
- Stage (development, staging, production)
- Approval workflow

**Versioning strategies**:

1. **Semantic versioning**: major.minor.patch
   - Major: breaking changes
   - Minor: new features, backward compatible
   - Patch: bug fixes

2. **Timestamp-based**: YYYYMMDD-HHMMSS

3. **Git commit hash**: tie to code version

**Tools**: MLflow Model Registry, SageMaker Model Registry, Vertex AI Model Registry

**Best practices**:
- Immutable model artifacts
- Tag production models
- Rollback capability
- Audit trail
- Automated testing before promotion

## Continuous Training

**Q: Design a continuous training system.**

A:

**Triggers for retraining**:

1. **Time-based**: Daily, weekly, monthly
2. **Performance-based**: When metrics degrade
3. **Data-based**: When sufficient new data accumulated
4. **Drift-based**: When data distribution changes

**Pipeline**:

1. **Data collection**:
   - Aggregate new labeled data
   - Include feedback loop (user corrections)
   - Validate data quality

2. **Training**:
   - Automated training job
   - Hyperparameter tuning
   - Compare with current production model

3. **Evaluation**:
   - Offline metrics on holdout set
   - Compare to baseline
   - Statistical significance test
   - Error analysis

4. **Deployment**:
   - Automated if metrics improve
   - Manual approval for critical systems
   - Gradual rollout
   - Monitor closely

5. **Feedback**:
   - Collect performance metrics
   - Update training data
   - Iterate

**Challenges**:
- Catastrophic forgetting (model forgets old patterns)
- Data quality issues
- Concept drift
- Computational cost
- Coordination with other systems

**Solutions**:
- Incremental learning
- Ensemble old and new models
- Curriculum learning
- Active learning for labeling

## Distributed Training

**Q: How do you scale training to multiple GPUs/machines?**

A:

**Data parallelism**:
- Replicate model on each device
- Split data across devices
- Synchronize gradients (all-reduce)
- Most common approach

**Model parallelism**:
- Split model across devices
- Each device computes part of model
- Use when model doesn't fit on single device
- Pipeline parallelism for efficiency

**Strategies**:

1. **Synchronous training**:
   - Wait for all workers before updating
   - Consistent but slower
   - Better convergence

2. **Asynchronous training**:
   - Workers update independently
   - Faster but can be unstable
   - Stale gradients problem

**Frameworks**:
- PyTorch Distributed (DDP, FSDP)
- TensorFlow Distributed (MirroredStrategy, MultiWorkerMirroredStrategy)
- Horovod
- DeepSpeed, Megatron for large models

**Optimizations**:
- Gradient accumulation
- Mixed precision training (FP16)
- Gradient checkpointing (trade compute for memory)
- ZeRO optimizer (sharded optimizer states)

**Challenges**:
- Communication overhead
- Load balancing
- Fault tolerance
- Debugging

## Model Serving Architecture

**Q: Design a scalable model serving system.**

A:

**Components**:

1. **Load balancer**:
   - Distribute requests across servers
   - Health checks
   - Sticky sessions if needed

2. **Inference servers**:
   - Model loaded in memory
   - Batch requests for efficiency
   - GPU utilization
   - Auto-scaling based on load

3. **Model cache**:
   - Cache frequent predictions
   - Redis, Memcached
   - TTL for freshness

4. **Feature service**:
   - Fetch features for prediction
   - Low-latency key-value store
   - Feature store integration

5. **Monitoring**:
   - Latency, throughput, errors
   - Model performance metrics
   - Resource utilization

**Optimization techniques**:

1. **Batching**:
   - Dynamic batching: accumulate requests
   - Trade latency for throughput
   - Batch size tuning

2. **Model optimization**:
   - Quantization, pruning
   - TensorRT, ONNX Runtime
   - Hardware-specific optimizations

3. **Caching**:
   - Prediction cache
   - Feature cache
   - Embedding cache

4. **Multi-model serving**:
   - Serve multiple models on same infrastructure
   - Model multiplexing
   - Resource sharing

**Deployment strategies**:

1. **Blue-green**: Two environments, switch traffic
2. **Canary**: Gradual rollout to subset of users
3. **Shadow**: Run new model alongside old, don't serve predictions
4. **A/B testing**: Split traffic between models

**Latency optimization**:
- Reduce model size
- Optimize preprocessing
- Use faster hardware (GPU, TPU)
- Parallel feature fetching
- Async processing where possible

## Data Pipelines

**Q: Design a data pipeline for ML training.**

A:

**Requirements**:
- Scalable (TB-PB of data)
- Reliable (fault tolerance)
- Reproducible (versioning)
- Efficient (minimize cost)

**Architecture**:

1. **Data ingestion**:
   - Batch: Airflow, Prefect, Luigi
   - Streaming: Kafka, Kinesis, Pub/Sub
   - Change data capture (CDC) for databases

2. **Data storage**:
   - Raw data: S3, GCS, HDFS
   - Processed data: Data warehouse (BigQuery, Snowflake)
   - Feature store: Online + offline stores

3. **Data processing**:
   - Batch: Spark, Beam, Dask
   - Streaming: Flink, Spark Streaming
   - SQL for transformations

4. **Data validation**:
   - Schema validation
   - Data quality checks
   - Anomaly detection
   - Great Expectations, TensorFlow Data Validation

5. **Data versioning**:
   - DVC, LakeFS
   - Track data lineage
   - Reproducibility

**Best practices**:
- Idempotent operations
- Incremental processing
- Partitioning (by date, user, etc.)
- Monitoring and alerting
- Data retention policies
- Cost optimization (compression, partitioning)

## Experiment Tracking

**Q: How do you track and manage ML experiments?**

A:

**What to track**:
- Hyperparameters
- Metrics (train, validation, test)
- Model artifacts
- Code version (git commit)
- Data version
- Environment (dependencies, hardware)
- Training time and cost

**Tools**: MLflow, Weights & Biases, Neptune, TensorBoard

**Best practices**:

1. **Naming conventions**: Descriptive experiment names
2. **Tagging**: Tag experiments by project, team, goal
3. **Comparison**: Compare experiments side-by-side
4. **Reproducibility**: Log everything needed to reproduce
5. **Collaboration**: Share experiments with team
6. **Visualization**: Plot metrics, compare runs

**Workflow**:
1. Start experiment run
2. Log parameters
3. Train model
4. Log metrics at each epoch
5. Log final model artifact
6. Tag and annotate
7. Compare with baseline

**Advanced features**:
- Hyperparameter sweeps
- Parallel experiments
- Early stopping based on metrics
- Automatic model selection

## CI/CD for ML

**Q: How do you implement CI/CD for ML systems?**

A:

**Continuous Integration**:

1. **Code testing**:
   - Unit tests for preprocessing, model code
   - Integration tests
   - Linting and formatting

2. **Data validation**:
   - Schema checks
   - Data quality tests
   - Distribution checks

3. **Model testing**:
   - Training pipeline test (small dataset)
   - Model performance tests
   - Inference tests

**Continuous Deployment**:

1. **Automated training**:
   - Trigger on code/data changes
   - Run training pipeline
   - Evaluate model

2. **Model validation**:
   - Compare to baseline
   - Check for regressions
   - Bias and fairness tests

3. **Deployment**:
   - Package model
   - Deploy to staging
   - Run integration tests
   - Deploy to production (canary/blue-green)

4. **Monitoring**:
   - Track deployment
   - Monitor metrics
   - Rollback if issues

**Tools**: Jenkins, GitLab CI, GitHub Actions, CircleCI

**Challenges**:
- Long training times
- Non-deterministic results
- Data dependencies
- Model size

**Best practices**:
- Fast feedback loop (test on small data)
- Separate model training from deployment
- Automated rollback
- Feature flags for gradual rollout

## Model Explainability

**Q: How do you make ML models interpretable?**

A:

**Model-agnostic methods**:

1. **SHAP** (SHapley Additive exPlanations):
   - Feature importance for each prediction
   - Based on game theory
   - Works for any model

2. **LIME** (Local Interpretable Model-agnostic Explanations):
   - Approximate model locally with interpretable model
   - Perturb input, observe output
   - Explain individual predictions

3. **Partial Dependence Plots**:
   - Show effect of feature on prediction
   - Marginalize over other features

4. **Feature importance**:
   - Permutation importance
   - Drop-column importance

**Model-specific methods**:

1. **Linear models**: Coefficient values
2. **Tree models**: Feature importance, tree visualization
3. **Neural networks**: 
   - Attention weights
   - Grad-CAM for CNNs
   - Integrated gradients

**Use cases**:
- Debugging models
- Building trust
- Regulatory compliance
- Fairness auditing
- Feature engineering insights

**Trade-offs**:
- Accuracy vs interpretability
- Global vs local explanations
- Computational cost

## Fairness and Bias

**Q: How do you ensure ML models are fair?**

A:

**Types of bias**:

1. **Data bias**:
   - Historical bias in training data
   - Sampling bias
   - Label bias

2. **Algorithmic bias**:
   - Model amplifies existing bias
   - Proxy features (zip code → race)

3. **Deployment bias**:
   - Different performance across groups
   - Feedback loops

**Fairness metrics**:

1. **Demographic parity**: Equal positive rate across groups
2. **Equal opportunity**: Equal true positive rate
3. **Equalized odds**: Equal TPR and FPR
4. **Calibration**: Predicted probabilities match actual rates

**Mitigation strategies**:

1. **Pre-processing**:
   - Reweighting samples
   - Resampling
   - Data augmentation

2. **In-processing**:
   - Fairness constraints in training
   - Adversarial debiasing
   - Multi-task learning

3. **Post-processing**:
   - Adjust thresholds per group
   - Calibration

**Tools**: Fairlearn, AI Fairness 360, What-If Tool

**Best practices**:
- Define fairness for your use case
- Measure across demographic groups
- Regular audits
- Diverse training data
- Human oversight
- Transparency

## Cost Optimization

**Q: How do you optimize ML infrastructure costs?**

A:

**Training costs**:

1. **Compute**:
   - Spot/preemptible instances (70% cheaper)
   - Right-size instances (don't over-provision)
   - Auto-scaling
   - Checkpointing for fault tolerance

2. **Efficiency**:
   - Mixed precision training (2x faster)
   - Gradient accumulation (larger effective batch size)
   - Early stopping
   - Hyperparameter optimization (fewer experiments)

3. **Scheduling**:
   - Run during off-peak hours
   - Batch jobs efficiently
   - Share resources across teams

**Inference costs**:

1. **Model optimization**:
   - Quantization, pruning, distillation
   - Reduce model size and latency
   - Fewer resources needed

2. **Serving**:
   - Batch predictions where possible
   - Cache frequent predictions
   - Auto-scaling (scale down when idle)
   - CPU for small models, GPU for large

3. **Architecture**:
   - Edge deployment (no cloud costs)
   - Serverless for variable load
   - Reserved instances for steady load

**Storage costs**:
- Data lifecycle policies (archive old data)
- Compression
- Deduplication
- Delete unused artifacts

**Monitoring**:
- Track costs per model, team, project
- Set budgets and alerts
- Cost attribution
- Regular audits

**Trade-offs**:
- Cost vs latency
- Cost vs accuracy
- Development speed vs efficiency


## ML Engineering Scenarios

### Scenario 1: Real-Time Fraud Detection System

**Q: Design a real-time fraud detection system for payment transactions.**

A:

**Requirements**:
- Latency < 100ms (block transaction if fraud)
- Handle 10K+ transactions/second
- High precision (minimize false positives)
- Adapt to new fraud patterns
- 99.99% uptime

**Architecture**:

1. **Feature engineering**:
   - Transaction features: amount, merchant, location, time
   - User features: account age, history, velocity (transactions/hour)
   - Aggregations: rolling windows (1h, 24h, 7d)
   - Device fingerprinting
   - Network features (IP, geolocation)

2. **Feature store**:
   - Online store: Redis/DynamoDB for low-latency lookup
   - Precompute aggregations
   - Update in real-time (streaming)
   - Offline store: historical data for training

3. **Model serving**:
   - Ensemble: Gradient boosting + neural network + rules
   - Model hosted on GPU servers
   - Load balancing across replicas
   - Fallback to simpler model if timeout

4. **Decision logic**:
   - Risk score (0-1)
   - Threshold-based decision
   - Different thresholds by transaction type
   - Manual review queue for borderline cases

5. **Streaming pipeline**:
   - Kafka for transaction stream
   - Flink/Spark Streaming for feature computation
   - Update feature store in real-time
   - Async model updates

**Training pipeline**:
- Daily retraining on recent data
- Handle class imbalance (SMOTE, class weights)
- Incorporate feedback (confirmed fraud, false positives)
- A/B test new models

**Monitoring**:
- Precision, recall, F1
- False positive rate (critical)
- Latency (p50, p95, p99)
- Feature drift
- Model performance by segment

**Challenges**:
- Adversarial users (fraudsters adapt)
- Concept drift (fraud patterns evolve)
- Cold start (new users/merchants)
- Imbalanced data (fraud is rare)
- Real-time feature computation

**Fallback strategies**:
- Rule-based system if model fails
- Cached predictions for repeat transactions
- Degrade gracefully (allow transaction, flag for review)

**Cost optimization**:
- Cache frequent feature lookups
- Batch feature computation where possible
- Use CPU for simple models
- Auto-scale based on traffic

### Scenario 2: Recommendation System at Scale

**Q: Design a recommendation system for a video streaming platform (Netflix-like).**

A:

**Requirements**:
- Personalized recommendations for 100M+ users
- Billions of user-item interactions
- Real-time updates (reflect recent watches)
- Diverse recommendations
- Cold start for new users/content

**Architecture**:

**Stage 1: Candidate generation** (retrieve ~1000 candidates)

1. **Collaborative filtering**:
   - Matrix factorization (ALS)
   - User and item embeddings
   - Approximate nearest neighbors (FAISS)

2. **Content-based**:
   - Item features (genre, actors, director)
   - User preferences
   - Cosine similarity

3. **Trending/popular**:
   - Time-decayed popularity
   - Geographic trending

4. **Contextual**:
   - Time of day, device
   - Recently watched genres

**Stage 2: Ranking** (rank top 1000 → top 50)

1. **Features**:
   - User: watch history, ratings, demographics
   - Item: metadata, popularity, recency
   - Context: time, device, session
   - User-item: similarity, past interactions

2. **Model**:
   - Deep neural network (Wide & Deep, DeepFM)
   - Predict watch probability or watch time
   - Multi-task: predict engagement + completion rate

3. **Training**:
   - Positive: watched videos
   - Negative: impressions without clicks
   - Sample negatives (can't use all)
   - Importance weighting

**Stage 3: Re-ranking** (apply business logic)

1. **Diversity**: Avoid too similar items
2. **Freshness**: Promote new content
3. **Exploration**: Show some random items
4. **Business rules**: Promote originals, contracts

**Serving**:

1. **Batch predictions**:
   - Precompute recommendations daily
   - Store in user profile database
   - Fast lookup, no real-time compute

2. **Real-time updates**:
   - Update after each watch
   - Recompute top candidates
   - Incremental updates to embeddings

3. **Caching**:
   - Cache user embeddings
   - Cache item embeddings
   - Cache recommendations

**Feature store**:
- User features: watch history, preferences
- Item features: metadata, embeddings
- Aggregations: watch counts, ratings
- Real-time: recent watches (last hour)

**Training pipeline**:
- Daily training on recent interactions
- Distributed training (billions of examples)
- Hyperparameter tuning
- A/B test new models

**Evaluation**:
- Offline: AUC, NDCG, Recall@k
- Online: CTR, watch time, retention
- User surveys

**Cold start**:
- New users: popular items, onboarding survey
- New items: content-based, show to diverse users
- Explore-exploit (multi-armed bandit)

**Challenges**:
- Scale (billions of interactions)
- Real-time updates
- Diversity vs relevance
- Filter bubbles
- Popularity bias

**Infrastructure**:
- Spark for batch processing
- Flink for streaming
- FAISS for ANN search
- Redis for caching
- Kubernetes for serving

### Scenario 3: Search Ranking System

**Q: Design a search ranking system for an e-commerce platform.**

A:

**Requirements**:
- Rank products for search queries
- Sub-second latency
- Personalized results
- Handle typos and synonyms
- Billions of products

**Architecture**:

**Stage 1: Query understanding**

1. **Query processing**:
   - Spell correction (edit distance, language model)
   - Tokenization and normalization
   - Query expansion (synonyms)
   - Intent classification (navigational, informational, transactional)

2. **Query rewriting**:
   - "iphone 13" → "iphone 13 pro max"
   - Learn from click data

**Stage 2: Retrieval** (get ~1000 candidates)

1. **Inverted index**:
   - BM25 scoring
   - Fast keyword matching
   - Elasticsearch, Solr

2. **Semantic search**:
   - Encode query and products to embeddings
   - FAISS for ANN search
   - Handles semantic similarity

3. **Filters**:
   - Apply user filters (price, brand, rating)
   - Availability check

**Stage 3: Ranking** (rank top 1000)

1. **Features**:
   - Query-product: BM25, semantic similarity, exact match
   - Product: popularity, rating, price, reviews
   - User: history, preferences, location
   - Context: time, device, session

2. **Model**:
   - Learning to rank (LambdaMART, LambdaRank)
   - Neural ranking (BERT-based cross-encoder)
   - Predict relevance score

3. **Training**:
   - Labels: clicks, purchases, dwell time
   - Pairwise or listwise loss
   - Position bias correction

**Stage 4: Re-ranking**

1. **Business logic**:
   - Promote sponsored products
   - Diversity (different brands)
   - Freshness (new products)

2. **Personalization**:
   - Boost based on user preferences
   - Recently viewed items

**Serving**:

1. **Caching**:
   - Cache popular queries
   - Cache product embeddings
   - TTL for freshness

2. **Latency optimization**:
   - Parallel retrieval and feature fetching
   - Early termination
   - Approximate scoring

**Feature store**:
- Product features: metadata, embeddings, stats
- User features: history, preferences
- Real-time: recent searches, clicks

**Training pipeline**:
- Daily training on click logs
- Negative sampling (impressions without clicks)
- A/B test ranking changes

**Evaluation**:
- Offline: NDCG, MRR, MAP
- Online: CTR, conversion rate, revenue
- User satisfaction surveys

**Challenges**:
- Long-tail queries (rare queries)
- Cold start (new products)
- Seasonality
- Query ambiguity
- Balancing relevance and business goals

**Monitoring**:
- Query latency
- Null result rate (no results)
- CTR by query type
- Revenue impact

### Scenario 4: Model Retraining Pipeline

**Q: Design an automated model retraining pipeline.**

A:

**Requirements**:
- Detect when retraining is needed
- Automated training and deployment
- Validation before deployment
- Rollback capability
- Cost-efficient

**Architecture**:

**1. Monitoring and triggers**:

Triggers for retraining:
- Scheduled (weekly, monthly)
- Performance degradation (accuracy drops)
- Data drift detected
- Sufficient new data accumulated

Monitoring:
- Model performance metrics
- Data distribution (KS test, PSI)
- Feature importance changes
- Business metrics

**2. Data pipeline**:

```
Raw data → Validation → Feature engineering → Feature store
```

- Collect new labeled data
- Validate schema and quality
- Compute features
- Store in feature store
- Version data

**3. Training pipeline**:

```
Data → Train → Evaluate → Register → Deploy
```

Steps:
1. Fetch training data from feature store
2. Split train/validation/test
3. Train model with hyperparameter tuning
4. Evaluate on test set
5. Compare with current production model
6. Register in model registry if better
7. Tag for deployment

**4. Validation**:

Before deployment:
- Offline metrics (accuracy, AUC, etc.)
- Statistical significance test
- Bias and fairness checks
- Inference latency test
- Integration tests

**5. Deployment**:

Strategy:
- Shadow mode: run alongside production, don't serve
- Canary: 5% traffic → 25% → 50% → 100%
- Monitor closely during rollout
- Automated rollback if metrics degrade

**6. Feedback loop**:

- Collect predictions and outcomes
- Label new data (user feedback, ground truth)
- Add to training data
- Continuous improvement

**Tools**:
- Orchestration: Airflow, Kubeflow Pipelines
- Training: SageMaker, Vertex AI
- Monitoring: Prometheus, Grafana
- Model registry: MLflow

**Automation**:
- Trigger retraining automatically
- Automated validation
- Automated deployment (with approval gate)
- Automated rollback

**Cost optimization**:
- Train only when needed (not on schedule if performance good)
- Use spot instances
- Incremental training where possible
- Cache intermediate results

**Challenges**:
- Catastrophic forgetting
- Data quality issues
- Coordination with other systems
- Downtime during deployment

**Best practices**:
- Version everything (data, code, models)
- Reproducibility
- Comprehensive testing
- Gradual rollout
- Monitor closely
- Document changes

### Scenario 5: Multi-Model Serving Platform

**Q: Design a platform to serve hundreds of ML models.**

A:

**Requirements**:
- Serve 100+ models
- Different frameworks (TensorFlow, PyTorch, XGBoost)
- Varying latency requirements
- Resource efficiency
- Easy onboarding for new models

**Architecture**:

**1. Model repository**:
- Centralized storage (S3, GCS)
- Model registry (MLflow, SageMaker)
- Versioning and metadata
- Access control

**2. Serving infrastructure**:

**Option A: Shared infrastructure**
- Multi-model server (TorchServe, TensorFlow Serving)
- Load multiple models on same instance
- Resource sharing
- Cost-efficient

**Option B: Dedicated infrastructure**
- Each model on separate instances
- Better isolation
- Independent scaling
- Higher cost

**Hybrid approach**:
- Critical models: dedicated
- Low-traffic models: shared
- Auto-scaling per model

**3. Model loading**:
- Lazy loading: load on first request
- Preloading: load at startup
- LRU cache: evict least recently used
- Model warming: preload popular models

**4. Request routing**:
- API gateway
- Route by model name/version
- Load balancing
- Circuit breaker for failing models

**5. Resource management**:
- CPU/GPU allocation per model
- Memory limits
- Request queuing
- Priority queues (critical models first)

**6. Monitoring**:
- Per-model metrics:
  - Latency (p50, p95, p99)
  - Throughput (QPS)
  - Error rate
  - Resource usage
- Aggregated metrics
- Alerting

**7. Deployment**:
- CI/CD pipeline
- Automated testing
- Canary deployments
- Blue-green for critical models
- Rollback capability

**API design**:

```
POST /predict
{
  "model": "fraud-detection",
  "version": "v2.1",
  "inputs": {...}
}
```

Response:
```
{
  "predictions": [...],
  "model_version": "v2.1",
  "latency_ms": 45
}
```

**Features**:
- Model versioning in API
- Batch prediction endpoint
- Async prediction for long-running
- Streaming predictions

**Optimization**:
- Batching: accumulate requests
- Caching: cache frequent predictions
- Model optimization: quantization, pruning
- Hardware acceleration: GPU, TPU

**Onboarding**:
- Self-service portal
- Model validation
- Performance testing
- Documentation and examples

**Cost management**:
- Track cost per model
- Auto-scale based on traffic
- Spot instances for non-critical
- Shared resources for low-traffic

**Challenges**:
- Resource contention
- Version management
- Dependency conflicts
- Monitoring at scale
- Cost allocation

**Tools**:
- KServe (Kubernetes-based)
- Seldon Core
- Ray Serve
- TensorFlow Serving
- TorchServe

### Scenario 6: Feature Store Implementation

**Q: Design and implement a feature store for an ML platform.**

A:

**Requirements**:
- Store features for training and serving
- Low-latency serving (<10ms)
- Point-in-time correctness for training
- Feature versioning
- Feature sharing across teams

**Architecture**:

**1. Offline store** (for training):
- Data warehouse: BigQuery, Snowflake, Redshift
- Stores historical features
- Supports time-travel queries
- Batch feature computation

**2. Online store** (for serving):
- Key-value store: Redis, DynamoDB, Cassandra
- Low-latency lookups
- Stores latest feature values
- Real-time feature updates

**3. Feature registry**:
- Metadata: feature definitions, schemas, owners
- Lineage: data sources, transformations
- Documentation
- Discovery (search features)

**4. Transformation engine**:
- Compute features from raw data
- Batch: Spark, Beam
- Streaming: Flink, Spark Streaming
- Consistent transformations for training and serving

**5. Ingestion pipeline**:

Batch:
```
Raw data → Transform → Offline store
                    → Materialize → Online store
```

Streaming:
```
Event stream → Transform → Online store
                        → Archive → Offline store
```

**Feature definitions**:

```python
@feature
def user_transaction_count_7d(user_id, timestamp):
    return count_transactions(user_id, timestamp - 7d, timestamp)
```

**Point-in-time correctness**:
- Training: use features as they existed at prediction time
- Avoid data leakage
- Time-travel queries in offline store

**Feature serving**:

```python
# Training
features = feature_store.get_historical_features(
    entity_df=training_data,
    features=["user_transaction_count_7d", "user_avg_amount"]
)

# Serving
features = feature_store.get_online_features(
    entity_keys={"user_id": "123"},
    features=["user_transaction_count_7d", "user_avg_amount"]
)
```

**Feature versioning**:
- Version feature definitions
- Backward compatibility
- Migration path for breaking changes

**Monitoring**:
- Feature freshness (staleness)
- Feature distribution (drift detection)
- Serving latency
- Data quality

**Benefits**:
- Feature reuse (DRY principle)
- Consistency (training-serving)
- Faster development
- Collaboration
- Governance

**Challenges**:
- Complexity
- Operational overhead
- Cost (storage, compute)
- Learning curve

**Tools**:
- Feast (open-source)
- Tecton
- AWS SageMaker Feature Store
- Vertex AI Feature Store
- Databricks Feature Store

**Best practices**:
- Start simple (don't over-engineer)
- Document features well
- Monitor feature quality
- Version features
- Test transformations
- Optimize for common access patterns

## Advanced ML Engineering Topics

**Q: Explain shadow mode deployment.**

A:

**Definition**: Run new model alongside production model without serving predictions to users

**Process**:
1. Deploy new model
2. Send same requests to both models
3. Compare predictions
4. Log differences
5. Analyze performance

**Benefits**:
- Test in production without risk
- Detect issues before full deployment
- Compare performance on real traffic
- Build confidence

**Metrics to compare**:
- Prediction agreement rate
- Latency
- Error rates
- Resource usage

**When to use**:
- High-stakes applications
- Major model changes
- New model architecture
- After significant retraining

**Q: What is model calibration and why is it important?**

A:

**Definition**: Predicted probabilities match actual frequencies

**Example**: If model predicts 70% probability, 70% of those predictions should be correct

**Importance**:
- Decision-making (thresholds)
- Cost-sensitive applications
- User trust
- Regulatory requirements

**Calibration methods**:

1. **Platt scaling**: Logistic regression on predictions
2. **Isotonic regression**: Non-parametric, monotonic
3. **Temperature scaling**: Scale logits before softmax

**Evaluation**:
- Calibration plot (reliability diagram)
- Expected Calibration Error (ECE)
- Brier score

**When needed**:
- Neural networks (often overconfident)
- After model compression
- Domain shift

**Q: How do you handle model dependencies in production?**

A:

**Challenges**:
- Library version conflicts
- OS dependencies
- Hardware requirements
- Reproducibility

**Solutions**:

1. **Containerization** (Docker):
   - Package model with dependencies
   - Consistent environment
   - Portable across platforms

2. **Virtual environments**:
   - Python: venv, conda
   - Isolate dependencies
   - Requirements.txt or environment.yml

3. **Model serialization**:
   - Save model in framework-agnostic format
   - ONNX, PMML
   - Reduces framework dependency

4. **Dependency pinning**:
   - Lock exact versions
   - Avoid breaking changes
   - requirements.txt with ==

5. **Infrastructure as code**:
   - Terraform, CloudFormation
   - Version infrastructure
   - Reproducible deployments

**Best practices**:
- Minimal dependencies
- Regular updates (security)
- Test with exact production versions
- Document dependencies
- Automated dependency scanning
