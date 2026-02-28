# Natural Language Processing

## Word Embeddings

**Q: Explain Word2Vec and its training objectives.**

A: Word2Vec learns dense vector representations where semantically similar words have similar vectors.

Two architectures:
- CBOW (Continuous Bag of Words): Predict target word from context words
- Skip-gram: Predict context words from target word

Training objectives:
- Maximize probability of context words given target
- Use negative sampling: sample negative examples to make training efficient
- Hierarchical softmax: alternative to negative sampling

Properties:
- Captures semantic relationships (king - man + woman ≈ queen)
- Fixed vocabulary, no handling of OOV words
- Context-independent (same vector regardless of context)

## Transformers

**Q: Explain the Transformer architecture and self-attention mechanism.**

A: Transformer uses self-attention to process sequences in parallel (unlike RNNs).

**Self-attention**:
- Compute Query, Key, Value matrices from input
- Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Each position attends to all positions
- Multi-head attention: multiple attention mechanisms in parallel

**Architecture**:
- Encoder: stack of self-attention + feed-forward layers
- Decoder: self-attention + encoder-decoder attention + feed-forward
- Positional encoding: inject position information
- Layer normalization and residual connections

**Advantages**:
- Parallelizable (faster training)
- Long-range dependencies
- Interpretable attention weights

**Disadvantages**:
- Quadratic complexity in sequence length
- Requires more data than RNNs

## BERT vs GPT

**Q: Compare BERT and GPT architectures and training.**

A:

**BERT** (Bidirectional Encoder Representations from Transformers):
- Encoder-only architecture
- Bidirectional context (sees both left and right)
- Pre-training: Masked Language Modeling (MLM) + Next Sentence Prediction
- Fine-tuning for downstream tasks
- Best for: classification, NER, question answering

**GPT** (Generative Pre-trained Transformer):
- Decoder-only architecture
- Unidirectional (left-to-right)
- Pre-training: Causal language modeling (predict next token)
- Few-shot learning with prompting
- Best for: text generation, completion

**Key difference**: BERT sees full context (better for understanding), GPT is autoregressive (better for generation).

## Text Classification

**Q: Design a text classification system.**

A:

**Approaches**:

1. **Traditional ML**:
   - Features: TF-IDF, n-grams
   - Models: Logistic regression, SVM, Naive Bayes
   - Fast, interpretable, good baseline

2. **Deep Learning**:
   - CNN: 1D convolutions over word embeddings
   - RNN/LSTM: Sequential processing
   - Attention-based: focus on important words

3. **Transfer Learning**:
   - Pre-trained models: BERT, RoBERTa, DistilBERT
   - Fine-tune on task-specific data
   - Best performance, requires more compute

**Pipeline**:
- Preprocessing: lowercase, remove special chars, tokenization
- Handle class imbalance: oversampling, class weights
- Evaluation: accuracy, F1, confusion matrix
- Error analysis: identify failure patterns

## Named Entity Recognition

**Q: How would you build an NER system?**

A: NER identifies and classifies named entities (person, organization, location, etc.)

**Approaches**:

1. **Rule-based**: Regex patterns, gazetteers. Fast but limited coverage.

2. **Traditional ML**: CRF (Conditional Random Fields)
   - Features: word, POS tags, capitalization, context
   - Models sequential dependencies
   - Requires feature engineering

3. **Deep Learning**: BiLSTM-CRF
   - BiLSTM: captures context from both directions
   - CRF layer: enforces valid tag sequences
   - Word + character embeddings

4. **Transformers**: BERT fine-tuning
   - Token classification task
   - Best performance
   - Handle subword tokenization carefully

**Evaluation**: Precision, recall, F1 at entity level (not token level)

**Challenges**:
- Ambiguous entities (Apple company vs fruit)
- Nested entities
- Domain-specific entities
- Limited training data

## Machine Translation

**Q: Explain sequence-to-sequence models for translation.**

A:

**Seq2Seq with Attention**:
- Encoder: RNN/LSTM encodes source sentence
- Decoder: RNN/LSTM generates target sentence
- Attention: decoder focuses on relevant encoder states
- Solves fixed-length bottleneck problem

**Transformer-based**:
- Encoder-decoder Transformer architecture
- Self-attention in encoder and decoder
- Cross-attention from decoder to encoder
- Parallel training, better long-range dependencies

**Training**:
- Parallel corpus (source-target pairs)
- Teacher forcing: use ground truth as decoder input
- Loss: Cross-entropy on target tokens
- Beam search for inference

**Evaluation**:
- BLEU score: n-gram overlap with references
- Human evaluation for quality
- Back-translation for data augmentation

**Challenges**:
- Low-resource languages
- Domain adaptation
- Rare words and OOV
- Maintaining context and style

## Attention Mechanisms Deep Dive

**Q: Explain different types of attention mechanisms.**

A:

**Scaled Dot-Product Attention**:
- Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Scaling by √d_k prevents softmax saturation
- O(n²) complexity in sequence length

**Multi-Head Attention**:
- Run h attention mechanisms in parallel
- Different heads learn different relationships
- Concatenate outputs and project
- MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O

**Cross-Attention**:
- Query from one sequence, Key/Value from another
- Used in encoder-decoder models
- Decoder attends to encoder outputs

**Self-Attention**:
- Q, K, V all from same sequence
- Each token attends to all tokens
- Captures dependencies within sequence

**Sparse Attention** (for long sequences):
- Longformer: local + global attention
- BigBird: random + window + global
- Reduces O(n²) to O(n)

## Large Language Models (LLMs)

**Q: Explain the evolution and architecture of modern LLMs.**

A:

**GPT Series**:
- GPT-2: 1.5B parameters, decoder-only
- GPT-3: 175B parameters, few-shot learning
- GPT-4: Multimodal, improved reasoning
- Training: next token prediction on massive text corpus

**Key innovations**:
- Scale: more parameters, more data, more compute
- In-context learning: learn from examples in prompt
- Instruction tuning: fine-tune on instruction-following tasks
- RLHF (Reinforcement Learning from Human Feedback): align with human preferences

**Architecture choices**:
- Decoder-only (GPT) vs Encoder-only (BERT) vs Encoder-Decoder (T5)
- Positional encodings: absolute vs relative (RoPE, ALiBi)
- Normalization: LayerNorm placement (pre-norm vs post-norm)
- Activation functions: GELU, SwiGLU

**Challenges**:
- Hallucinations: generating false information
- Context length limitations
- Computational cost
- Bias and safety concerns

## Prompt Engineering

**Q: What are best practices for prompt engineering?**

A:

**Techniques**:

1. **Zero-shot**: Direct instruction without examples
   - "Classify sentiment: [text]"

2. **Few-shot**: Provide examples in prompt
   - "Positive: I love this! Negative: This is terrible. [text]"

3. **Chain-of-Thought (CoT)**: Ask model to reason step-by-step
   - "Let's think step by step..."
   - Improves reasoning tasks

4. **Self-consistency**: Generate multiple reasoning paths, take majority vote

5. **ReAct**: Combine reasoning and actions
   - Thought → Action → Observation loop

**Best practices**:
- Be specific and clear
- Provide context and constraints
- Use delimiters to separate sections
- Specify output format
- Iterate and refine based on results

## Tokenization

**Q: Compare different tokenization strategies.**

A:

**Word-level**:
- Split on whitespace/punctuation
- Large vocabulary, OOV problem
- Simple but inflexible

**Character-level**:
- No OOV issues
- Very long sequences
- Loses word-level semantics

**Subword tokenization**:

1. **BPE (Byte Pair Encoding)**:
   - Start with characters, merge frequent pairs
   - Used in GPT
   - Balances vocabulary size and sequence length

2. **WordPiece**:
   - Similar to BPE, used in BERT
   - Merges based on likelihood

3. **SentencePiece**:
   - Language-agnostic, treats text as raw bytes
   - Used in T5, XLNet
   - No pre-tokenization needed

**Trade-offs**:
- Vocabulary size vs sequence length
- OOV handling vs semantic preservation
- Language-specific vs universal

## Question Answering Systems

**Q: Design a question answering system.**

A:

**Types**:

1. **Extractive QA**: Extract answer span from context
   - SQuAD dataset format
   - BERT fine-tuning: predict start/end positions
   - Fast, interpretable

2. **Abstractive QA**: Generate answer
   - Seq2seq models
   - More flexible but can hallucinate

3. **Open-domain QA**: No context provided
   - Retrieval + reading comprehension
   - Dense passage retrieval (DPR)
   - RAG (Retrieval-Augmented Generation)

**Architecture** (Extractive):
- Encoder: BERT/RoBERTa
- Output: two classifiers for start/end positions
- Loss: cross-entropy on position labels

**Retrieval-Augmented**:
1. Retriever: find relevant documents
   - BM25 (sparse) or dense embeddings
2. Reader: extract/generate answer from retrieved docs
3. End-to-end training possible

**Evaluation**:
- Exact Match (EM): exact string match
- F1: token-level overlap
- Human evaluation for quality

## Sentiment Analysis

**Q: Build a production sentiment analysis system.**

A:

**Approaches**:

1. **Lexicon-based**:
   - VADER, TextBlob
   - Fast, no training needed
   - Limited accuracy, can't handle context

2. **Traditional ML**:
   - TF-IDF features + Logistic Regression/SVM
   - Good baseline, interpretable
   - Requires feature engineering

3. **Deep Learning**:
   - LSTM with word embeddings
   - CNN for text
   - Attention mechanisms

4. **Transfer Learning**:
   - Fine-tune BERT/RoBERTa
   - Best accuracy
   - DistilBERT for faster inference

**Challenges**:
- Sarcasm and irony
- Negation handling ("not good")
- Aspect-based sentiment (multiple aspects in text)
- Domain-specific language
- Multilingual support

**Production considerations**:
- Latency requirements (real-time vs batch)
- Model size vs accuracy trade-off
- Handling emojis, slang, typos
- Confidence scores for uncertain predictions
- A/B testing different models

## Text Summarization

**Q: Compare extractive and abstractive summarization.**

A:

**Extractive**:
- Select important sentences from source
- TextRank, LexRank algorithms
- Fast, factually accurate
- Can be choppy, lacks coherence

**Abstractive**:
- Generate new sentences
- Seq2seq with attention
- Transformer models (BART, T5, Pegasus)
- More fluent but can hallucinate

**Evaluation metrics**:
- ROUGE: n-gram overlap with reference
- BLEU: originally for translation
- BERTScore: semantic similarity
- Human evaluation: fluency, coherence, factuality

**Challenges**:
- Long documents (context length limits)
- Factual consistency
- Handling multiple documents
- Domain adaptation

**Production approach**:
- Hybrid: extractive for candidate selection, abstractive for refinement
- Post-processing for factuality checks
- Length control mechanisms


## Embedding Models

**Q: Compare different embedding approaches.**

A:

**Static embeddings** (Word2Vec, GloVe, FastText):
- One vector per word
- Context-independent
- Fast, small memory footprint
- Can't handle polysemy (bank: river vs financial)

**Contextualized embeddings** (ELMo, BERT):
- Different vectors based on context
- Captures word sense disambiguation
- Larger models, slower
- Better performance on downstream tasks

**Sentence embeddings**:
- Sentence-BERT: siamese BERT for sentence similarity
- Universal Sentence Encoder
- Use cases: semantic search, clustering, duplicate detection

**Dense retrieval**:
- Encode queries and documents as dense vectors
- Similarity via dot product or cosine
- Faster than cross-encoder, less accurate
- Used in search and QA systems

## Language Model Evaluation

**Q: How do you evaluate language models?**

A:

**Intrinsic metrics**:
- Perplexity: exp(average negative log-likelihood)
- Lower is better
- Measures how well model predicts test data
- Doesn't correlate perfectly with downstream performance

**Extrinsic metrics**:
- Performance on downstream tasks (GLUE, SuperGLUE)
- Task-specific metrics (F1, accuracy, BLEU)
- Better indicator of real-world usefulness

**Human evaluation**:
- Fluency: grammatical and natural
- Coherence: logical flow
- Factuality: accurate information
- Relevance: on-topic

**Benchmark suites**:
- GLUE: 9 NLU tasks
- SuperGLUE: harder version
- MMLU: multitask language understanding
- BIG-bench: diverse challenging tasks

## Handling Long Documents

**Q: How do you process documents longer than model's context window?**

A:

**Strategies**:

1. **Truncation**:
   - Take first N tokens
   - Simple but loses information
   - Works if important info is at start

2. **Sliding window**:
   - Process overlapping chunks
   - Aggregate predictions (max, average, voting)
   - Maintains local context

3. **Hierarchical models**:
   - Encode sentences/paragraphs separately
   - Second-level model over encodings
   - Captures document structure

4. **Sparse attention**:
   - Longformer, BigBird
   - Efficient attention for long sequences
   - O(n) instead of O(n²)

5. **Retrieval-based**:
   - Retrieve relevant chunks
   - Process only relevant parts
   - Used in open-domain QA

**Trade-offs**:
- Accuracy vs computational cost
- Context preservation vs efficiency
- Task-dependent optimal strategy

## NLP System Design Scenarios

### Scenario 1: Search Query Understanding

**Q: Design a system to understand and improve search queries.**

A:

**Components**:

1. **Query classification**:
   - Intent detection (navigational, informational, transactional)
   - Entity recognition
   - Language detection

2. **Query rewriting**:
   - Spell correction (edit distance, language model)
   - Query expansion (synonyms, related terms)
   - Query relaxation for zero results

3. **Query segmentation**:
   - Break into meaningful units
   - "new york pizza" → [new york] [pizza]

4. **Personalization**:
   - User history and preferences
   - Location-based adjustments
   - Click-through data

**ML models**:
- BERT for query understanding
- Seq2seq for query rewriting
- Click models for relevance

**Evaluation**:
- Click-through rate
- Time to success
- User satisfaction surveys
- A/B testing

### Scenario 2: Content Moderation

**Q: Design a content moderation system for social media.**

A:

**Requirements**:
- Detect toxic, hateful, spam content
- Multiple languages
- Real-time processing
- High precision (avoid false positives)

**Architecture**:

1. **Rule-based filters** (first pass):
   - Keyword blacklists
   - Regex patterns
   - Fast, catches obvious cases

2. **ML classifiers**:
   - Multi-label classification (toxic, hate, spam, etc.)
   - BERT-based models
   - Ensemble for robustness

3. **Contextual analysis**:
   - Consider conversation thread
   - User history
   - Sarcasm detection

4. **Human review**:
   - Queue borderline cases
   - Active learning: retrain on reviewed examples
   - Feedback loop

**Challenges**:
- Adversarial users (obfuscation, misspellings)
- Cultural context and slang
- Evolving language
- Bias and fairness
- Appeal process

**Metrics**:
- Precision/recall on labeled data
- False positive rate (critical)
- Processing latency
- Human review queue size

### Scenario 3: Chatbot for Customer Support

**Q: Design a customer support chatbot.**

A:

**Architecture**:

1. **Intent classification**:
   - Identify user's goal (refund, track order, etc.)
   - Multi-class classifier
   - Confidence threshold for escalation

2. **Entity extraction**:
   - Order ID, product name, dates
   - NER model
   - Validation against database

3. **Dialogue management**:
   - Track conversation state
   - Multi-turn context
   - Slot filling for required information

4. **Response generation**:
   - Template-based for structured responses
   - Retrieval-based for FAQ
   - Generative (GPT) for complex queries
   - Hybrid approach

5. **Knowledge base**:
   - FAQ database
   - Product documentation
   - Retrieval system (semantic search)

**Fallback strategy**:
- Confidence thresholds
- Escalate to human agent
- Clarifying questions

**Evaluation**:
- Task completion rate
- User satisfaction (CSAT)
- Average handling time
- Escalation rate
- Intent classification accuracy

**Production considerations**:
- Latency < 1 second
- Multilingual support
- Personalization (user history)
- A/B testing responses
- Continuous learning from interactions

### Scenario 4: Document Search and Ranking

**Q: Build a semantic search system for internal documents.**

A:

**Requirements**:
- Search across millions of documents
- Understand semantic meaning (not just keywords)
- Fast retrieval (<100ms)
- Relevance ranking

**Architecture**:

1. **Indexing pipeline**:
   - Document preprocessing (OCR, parsing)
   - Chunk documents (paragraphs/sections)
   - Generate embeddings (Sentence-BERT)
   - Store in vector database (Pinecone, Weaviate, FAISS)

2. **Query processing**:
   - Query expansion (synonyms)
   - Encode query to embedding
   - Retrieve top-k candidates (ANN search)

3. **Ranking**:
   - Re-rank candidates with cross-encoder
   - Consider recency, authority, user preferences
   - Learning to rank (LambdaMART)

4. **Hybrid approach**:
   - Combine semantic (dense) and keyword (BM25) search
   - Reciprocal rank fusion

**Features**:
- Query-document similarity
- Document metadata (date, author, type)
- User engagement signals (clicks, dwell time)
- Document quality scores

**Evaluation**:
- Offline: NDCG, MRR, MAP
- Online: CTR, time to success
- User feedback

**Challenges**:
- Cold start for new documents
- Handling domain-specific terminology
- Multilingual documents
- Access control and permissions
- Keeping index updated

### Scenario 5: Email Auto-Reply Suggestions

**Q: Design a system that suggests email replies.**

A:

**Problem**: Generate 3 short reply suggestions for incoming emails

**Approach**:

1. **Email understanding**:
   - Intent classification (question, request, FYI)
   - Sentiment analysis
   - Extract key entities and topics
   - Urgency detection

2. **Reply generation**:
   - Template-based for common patterns
   - Seq2seq model for custom replies
   - Fine-tuned GPT on email corpus
   - Ensure diversity in suggestions

3. **Personalization**:
   - User's writing style
   - Common phrases and sign-offs
   - Relationship with sender
   - Previous email thread context

4. **Ranking**:
   - Relevance to email content
   - Appropriateness (tone, formality)
   - Diversity among suggestions
   - User click history

**Training data**:
- Email-reply pairs from users (with consent)
- Filter for quality and privacy
- Augmentation for rare cases

**Evaluation**:
- Click-through rate on suggestions
- Edit distance (how much user modifies)
- Acceptance rate
- User satisfaction

**Privacy and safety**:
- On-device processing where possible
- Anonymize training data
- Content filtering
- User control and opt-out

### Scenario 6: Multilingual Product Reviews Analysis

**Q: Analyze product reviews in multiple languages at scale.**

A:

**Requirements**:
- Process reviews in 20+ languages
- Extract sentiment, aspects, key issues
- Real-time dashboard for product teams
- Handle millions of reviews

**Architecture**:

1. **Language detection**:
   - FastText language identifier
   - Route to appropriate pipeline

2. **Translation** (optional):
   - Translate to English for unified processing
   - Or use multilingual models (mBERT, XLM-R)
   - Trade-off: translation cost vs model complexity

3. **Sentiment analysis**:
   - Multilingual BERT fine-tuned on reviews
   - Aspect-based sentiment (price, quality, shipping)
   - Rating prediction

4. **Topic modeling**:
   - Extract common themes
   - LDA or neural topic models
   - Cluster similar reviews

5. **Key phrase extraction**:
   - Identify frequently mentioned issues
   - RAKE, YAKE, or transformer-based
   - Trend detection over time

**Aggregation**:
- Product-level metrics
- Time-series analysis
- Comparison across products/categories
- Alert on sudden sentiment drops

**Challenges**:
- Language-specific nuances
- Sarcasm and cultural context
- Fake reviews detection
- Handling code-mixed text
- Scalability (batch processing)

**Metrics**:
- Sentiment accuracy per language
- Aspect extraction F1
- Processing throughput
- Dashboard load time
- Actionable insights generated

## Advanced NLP Concepts

**Q: Explain few-shot and zero-shot learning in NLP.**

A:

**Zero-shot**:
- Model performs task without task-specific training
- Relies on pre-training and prompting
- Example: "Translate to French: Hello" (without translation training)
- GPT-3 demonstrates strong zero-shot capabilities

**Few-shot**:
- Provide few examples in prompt (in-context learning)
- No gradient updates
- Example: Show 3 sentiment examples, classify 4th
- Performance improves with more examples

**Meta-learning**:
- Train model to learn from few examples
- MAML (Model-Agnostic Meta-Learning)
- Learn good initialization for fast adaptation

**Applications**:
- Low-resource languages
- Rare entity types
- New domains without labeled data
- Rapid prototyping

**Q: What is transfer learning in NLP and why is it effective?**

A:

**Paradigm**: Pre-train on large corpus, fine-tune on task

**Pre-training objectives**:
- Masked language modeling (BERT)
- Causal language modeling (GPT)
- Denoising autoencoding (T5)
- Contrastive learning (SimCSE)

**Why it works**:
- Learns general language understanding
- Captures syntax, semantics, world knowledge
- Reduces need for task-specific data
- Better than training from scratch

**Fine-tuning strategies**:
- Full fine-tuning: update all parameters
- Adapter layers: add small trainable modules
- LoRA: low-rank adaptation of weights
- Prompt tuning: only tune prompt embeddings

**Domain adaptation**:
- Continue pre-training on domain data
- Then fine-tune on task
- Improves performance on specialized domains (medical, legal)

**Q: How do you handle class imbalance in text classification?**

A:

**Data-level**:
- Oversample minority class
- Undersample majority class
- SMOTE for text (careful with discrete data)
- Data augmentation (back-translation, paraphrasing)

**Algorithm-level**:
- Class weights in loss function
- Focal loss: focus on hard examples
- Cost-sensitive learning

**Evaluation**:
- Don't use accuracy
- Precision, recall, F1 per class
- Macro vs micro averaging
- Confusion matrix analysis

**Ensemble**:
- Train multiple models on balanced subsets
- Combine predictions

**Threshold tuning**:
- Adjust classification threshold
- Optimize for business metric
- Different thresholds per class

**Active learning**:
- Iteratively label uncertain examples
- Focus on minority class
- Reduces labeling cost


# Natural Language Processing

## Word Embeddings

**Q: Explain Word2Vec and its training objectives.**

A: Word2Vec learns dense vector representations where semantically similar words have similar vectors.

Two architectures:
- CBOW (Continuous Bag of Words): Predict target word from context words
- Skip-gram: Predict context words from target word

Training objectives:
- Maximize probability of context words given target
- Use negative sampling: sample negative examples to make training efficient
- Hierarchical softmax: alternative to negative sampling

Properties:
- Captures semantic relationships (king - man + woman ≈ queen)
- Fixed vocabulary, no handling of OOV words
- Context-independent (same vector regardless of context)

## Transformers

**Q: Explain the Transformer architecture and self-attention mechanism.**

A: Transformer uses self-attention to process sequences in parallel (unlike RNNs).

**Self-attention**:
- Compute Query, Key, Value matrices from input
- Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Each position attends to all positions
- Multi-head attention: multiple attention mechanisms in parallel

**Architecture**:
- Encoder: stack of self-attention + feed-forward layers
- Decoder: self-attention + encoder-decoder attention + feed-forward
- Positional encoding: inject position information
- Layer normalization and residual connections

**Advantages**:
- Parallelizable (faster training)
- Long-range dependencies
- Interpretable attention weights

**Disadvantages**:
- Quadratic complexity in sequence length
- Requires more data than RNNs

## BERT vs GPT

**Q: Compare BERT and GPT architectures and training.**

A:

**BERT** (Bidirectional Encoder Representations from Transformers):
- Encoder-only architecture
- Bidirectional context (sees both left and right)
- Pre-training: Masked Language Modeling (MLM) + Next Sentence Prediction
- Fine-tuning for downstream tasks
- Best for: classification, NER, question answering

**GPT** (Generative Pre-trained Transformer):
- Decoder-only architecture
- Unidirectional (left-to-right)
- Pre-training: Causal language modeling (predict next token)
- Few-shot learning with prompting
- Best for: text generation, completion

**Key difference**: BERT sees full context (better for understanding), GPT is autoregressive (better for generation).

## Text Classification

**Q: Design a text classification system.**

A:

**Approaches**:

1. **Traditional ML**:
   - Features: TF-IDF, n-grams
   - Models: Logistic regression, SVM, Naive Bayes
   - Fast, interpretable, good baseline

2. **Deep Learning**:
   - CNN: 1D convolutions over word embeddings
   - RNN/LSTM: Sequential processing
   - Attention-based: focus on important words

3. **Transfer Learning**:
   - Pre-trained models: BERT, RoBERTa, DistilBERT
   - Fine-tune on task-specific data
   - Best performance, requires more compute

**Pipeline**:
- Preprocessing: lowercase, remove special chars, tokenization
- Handle class imbalance: oversampling, class weights
- Evaluation: accuracy, F1, confusion matrix
- Error analysis: identify failure patterns

## Named Entity Recognition

**Q: How would you build an NER system?**

A: NER identifies and classifies named entities (person, organization, location, etc.)

**Approaches**:

1. **Rule-based**: Regex patterns, gazetteers. Fast but limited coverage.

2. **Traditional ML**: CRF (Conditional Random Fields)
   - Features: word, POS tags, capitalization, context
   - Models sequential dependencies
   - Requires feature engineering

3. **Deep Learning**: BiLSTM-CRF
   - BiLSTM: captures context from both directions
   - CRF layer: enforces valid tag sequences
   - Word + character embeddings

4. **Transformers**: BERT fine-tuning
   - Token classification task
   - Best performance
   - Handle subword tokenization carefully

**Evaluation**: Precision, recall, F1 at entity level (not token level)

**Challenges**:
- Ambiguous entities (Apple company vs fruit)
- Nested entities
- Domain-specific entities
- Limited training data

## Machine Translation

**Q: Explain sequence-to-sequence models for translation.**

A:

**Seq2Seq with Attention**:
- Encoder: RNN/LSTM encodes source sentence
- Decoder: RNN/LSTM generates target sentence
- Attention: decoder focuses on relevant encoder states
- Solves fixed-length bottleneck problem

**Transformer-based**:
- Encoder-decoder Transformer architecture
- Self-attention in encoder and decoder
- Cross-attention from decoder to encoder
- Parallel training, better long-range dependencies

**Training**:
- Parallel corpus (source-target pairs)
- Teacher forcing: use ground truth as decoder input
- Loss: Cross-entropy on target tokens
- Beam search for inference

**Evaluation**:
- BLEU score: n-gram overlap with references
- Human evaluation for quality
- Back-translation for data augmentation

**Challenges**:
- Low-resource languages
- Domain adaptation
- Rare words and OOV
- Maintaining context and style
