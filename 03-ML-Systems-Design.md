# ML Systems Design

## Design Framework

**Q: Walk through designing an ML system (e.g., recommendation system, search ranking).**

A: Use this framework:

1. **Clarify requirements**
   - Business objective and success metrics
   - Scale (users, items, QPS)
   - Latency requirements
   - Constraints (compute, data, privacy)

2. **Define ML problem**
   - Formulate as classification/ranking/regression
   - Define labels and features
   - Online vs offline evaluation metrics

3. **Data pipeline**
   - Data sources and collection
   - Feature engineering
   - Training data generation
   - Data quality and validation

4. **Model development**
   - Baseline model
   - Advanced models
   - Training infrastructure
   - Experimentation framework

5. **Serving architecture**
   - Online vs batch predictions
   - Caching strategy
   - Fallback mechanisms
   - A/B testing framework

6. **Monitoring and iteration**
   - Model performance metrics
   - Data drift detection
   - Retraining strategy
   - Feedback loops

## Recommendation System

**Q: Design a recommendation system for an e-commerce platform.**

A:

**Objective**: Increase user engagement and purchases

**Approach**:
- Collaborative filtering: user-item interactions
- Content-based: item features
- Hybrid: combine both

**Architecture**:
1. Candidate generation: retrieve ~1000 candidates
   - User history similarity
   - Popular items
   - Content-based matching
2. Ranking: score and rank top candidates
   - Deep neural network with user/item features
   - Predict click probability or purchase probability
3. Re-ranking: apply business rules, diversity

**Features**:
- User: demographics, browsing history, purchase history
- Item: category, price, ratings, popularity
- Context: time, device, location
- Interaction: click-through rate, conversion rate

**Training**:
- Positive: clicks, purchases
- Negative: impressions without clicks
- Handle imbalanced data with sampling

**Serving**:
- Pre-compute embeddings
- Cache popular recommendations
- Real-time feature computation
- <100ms latency target

## Search Ranking

**Q: Design a search ranking system.**

A:

**Two-stage approach**:

1. **Retrieval**: Get candidate documents (~1000)
   - Inverted index for keyword matching
   - BM25 scoring
   - Semantic search with embeddings

2. **Ranking**: ML model to rank candidates
   - Learning to rank (LTR) approach
   - Pointwise, pairwise, or listwise loss

**Features**:
- Query: length, terms, intent classification
- Document: relevance score, PageRank, freshness
- Query-document: TF-IDF, BM25, semantic similarity
- User: location, history, click patterns
- Context: time, device

**Training data**:
- Explicit: human ratings
- Implicit: clicks, dwell time, skip rate
- Position bias correction

**Metrics**:
- Offline: NDCG, MRR, MAP
- Online: CTR, time to success, user satisfaction

**Challenges**:
- Cold start for new documents
- Query understanding and expansion
- Personalization vs privacy
- Handling typos and synonyms

## Fraud Detection

**Q: Design a fraud detection system for transactions.**

A:

**Problem**: Binary classification (fraud vs legitimate)

**Challenges**:
- Highly imbalanced (fraud is rare)
- Adversarial (fraudsters adapt)
- Real-time requirements
- False positives are costly

**Features**:
- Transaction: amount, merchant, location, time
- User: account age, history, velocity features
- Device: fingerprint, IP address
- Behavioral: deviation from normal patterns

**Model**:
- Ensemble of models (random forest, gradient boosting, neural network)
- Anomaly detection for novel fraud patterns
- Rule-based system for known patterns

**Training**:
- Oversample minority class or use class weights
- Use precision-recall curve (not accuracy)
- Optimize for business metric (cost of fraud vs false positives)

**Serving**:
- Real-time scoring (<100ms)
- Threshold tuning based on risk tolerance
- Manual review queue for borderline cases
- Feedback loop for model updates

**Monitoring**:
- Track precision/recall over time
- Detect new fraud patterns
- A/B test model changes carefully
