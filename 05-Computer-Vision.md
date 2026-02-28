# Computer Vision

## Image Classification

**Q: Explain modern CNN architectures for image classification.**

A:

**ResNet** (Residual Networks):
- Skip connections: x + F(x) instead of F(x)
- Solves vanishing gradient problem
- Enables training very deep networks (50-152 layers)
- Identity mapping allows gradient flow
- Bottleneck design: 1x1 → 3x3 → 1x1 convolutions

**Inception**:
- Multi-scale feature extraction
- Parallel convolutions with different kernel sizes (1x1, 3x3, 5x5)
- 1x1 convolutions for dimensionality reduction
- Efficient computation

**EfficientNet**:
- Compound scaling: balance depth, width, resolution
- Neural architecture search for optimal structure
- State-of-the-art accuracy with fewer parameters

**MobileNet**:
- Depthwise separable convolutions
- Reduces computation and parameters
- Designed for mobile/edge devices

**Vision Transformer (ViT)**:
- Apply transformer architecture to images
- Split image into patches, treat as sequence
- Self-attention across patches
- Requires large datasets or pre-training

## Object Detection

**Q: Compare two-stage and one-stage object detectors.**

A:

**Two-stage detectors** (R-CNN family):

1. **Faster R-CNN**: 
   - RPN generates proposals
   - ROI pooling and classification
   - Anchor boxes at multiple scales
   - Higher accuracy, slower (~5-10 FPS)

**One-stage detectors**:

1. **YOLO** (You Only Look Once):
   - Single forward pass
   - Divides image into grid
   - Each cell predicts bounding boxes and classes
   - Very fast, real-time capable (30-60 FPS)
   - Lower accuracy on small objects

2. **SSD** (Single Shot Detector):
   - Multi-scale feature maps
   - Default boxes at different scales
   - Balance between speed and accuracy

3. **RetinaNet**:
   - Focal loss to handle class imbalance
   - Feature Pyramid Network (FPN)

**Trade-off**: Two-stage more accurate, one-stage faster

## Semantic Segmentation

**Q: Explain semantic segmentation architectures.**

A: Pixel-level classification (assign class to each pixel)

**U-Net**:
- Encoder-decoder architecture
- Symmetric skip connections
- Concatenate encoder features with decoder
- Excellent for medical imaging

**DeepLab**:
- Atrous (dilated) convolutions: expand receptive field
- Atrous Spatial Pyramid Pooling (ASPP): multi-scale context
- DeepLabv3+ with encoder-decoder

**Evaluation**: IoU (Intersection over Union), pixel accuracy, mean IoU

## Instance Segmentation

**Q: How does Mask R-CNN work?**

A: Extends Faster R-CNN for instance segmentation

**Architecture**:
1. Backbone CNN: feature extraction
2. RPN: generate region proposals
3. ROI Align: extract features (better than ROI pooling)
4. Parallel branches:
   - Classification: object class
   - Bounding box regression
   - Mask prediction: binary mask

**Training**:
- Multi-task loss: L = L_cls + L_box + L_mask
- Binary cross-entropy per pixel

## Transfer Learning

**Q: Best practices for transfer learning in computer vision.**

A:

**Strategy based on data size**:

1. **Small data, similar domain**:
   - Freeze all layers except final classifier
   - Train only the new classification head

2. **Medium data, similar domain**:
   - Freeze early layers (generic features)
   - Fine-tune later layers
   - Use smaller learning rate

3. **Large data**:
   - Fine-tune entire network
   - Use small learning rate

**Tips**:
- Use ImageNet pre-trained weights
- Learning rate: 10-100x smaller for pre-trained layers
- Data augmentation crucial with small datasets



## CNN Deep Dive

**Q: Explain key CNN concepts and design choices.**

A:

**Receptive field**:
- Region of input that affects a neuron's output
- Grows with depth
- Calculate: RF = 1 + Σ(kernel_size - 1) * Π(previous_strides)

**Pooling**:
- Max pooling: take maximum in window
- Average pooling: take average
- Reduces spatial dimensions
- Provides translation invariance

**1x1 convolutions**:
- Dimensionality reduction/expansion
- Add non-linearity
- Cross-channel information mixing

**Depthwise separable convolutions**:
- Depthwise: convolve each channel separately
- Pointwise: 1x1 convolution across channels
- Reduces parameters significantly
- Used in MobileNet, EfficientNet

## Image Augmentation

**Q: What augmentation techniques improve model robustness?**

A:

**Geometric transformations**:
- Rotation, flipping, scaling
- Random crops and resizing
- Affine transformations
- Elastic deformations

**Color transformations**:
- Brightness, contrast, saturation
- Hue shifting, color jittering
- Histogram equalization

**Advanced techniques**:
- Cutout: mask random patches
- Mixup: blend two images and labels
- CutMix: paste patches from other images
- AutoAugment: learned augmentation policies
- RandAugment: simplified random augmentation

**Test-time augmentation**: Average predictions on augmented versions

## Object Tracking

**Q: Explain approaches to object tracking in videos.**

A:

**Single Object Tracking**:
- Siamese networks (SiamFC, SiamRPN)
- Learn similarity between template and search region
- One-shot learning

**Multi-Object Tracking**:
- Tracking-by-detection approach
- Detect objects in each frame
- Associate detections across frames (SORT, DeepSORT)
- IoU matching + appearance features
- Kalman filter for motion prediction

**Challenges**:
- Occlusions
- Scale changes
- Fast motion
- ID switches

## Face Recognition

**Q: Design a face recognition system.**

A:

**Pipeline**:

1. **Face detection**: MTCNN, RetinaFace
2. **Face alignment**: Normalize pose using landmarks
3. **Face embedding**: Deep CNN (FaceNet, ArcFace)
4. **Face matching**: Cosine similarity or Euclidean distance

**Training**:
- Triplet loss: anchor, positive, negative
- ArcFace loss: additive angular margin
- Large-scale datasets

**Challenges**:
- Pose variation
- Lighting conditions
- Occlusions (masks, glasses)
- Bias and fairness

**Production**:
- 1:1 verification vs 1:N identification
- Liveness detection (anti-spoofing)
- Privacy considerations

## Image Retrieval

**Q: Build a large-scale image retrieval system.**

A:

**Architecture**:

1. **Feature extraction**:
   - CNN backbone (ResNet, EfficientNet)
   - Global pooling
   - Embedding dimension (128-2048)

2. **Indexing**:
   - Store embeddings in vector database
   - Approximate Nearest Neighbor (ANN) search
   - FAISS, Annoy, ScaNN

3. **Retrieval**:
   - Encode query image
   - ANN search for top-k similar embeddings
   - Optional re-ranking

**Training**:
- Metric learning: triplet loss, contrastive loss
- Self-supervised learning (SimCLR, MoCo)

**Evaluation**: mAP, Recall@k

**Applications**: Visual search, duplicate detection, reverse image search

## Video Understanding

**Q: How do you process videos for action recognition?**

A:

**Approaches**:

1. **2D CNN + Temporal pooling**:
   - Extract features per frame
   - Aggregate over time (average, max)
   - Simple but loses temporal info

2. **3D CNN**:
   - Convolve over space and time
   - C3D, I3D models
   - Captures motion patterns
   - Computationally expensive

3. **Two-stream networks**:
   - Spatial stream: RGB frames
   - Temporal stream: optical flow
   - Fuse predictions
   - Captures appearance and motion

4. **Transformer-based**:
   - Video Vision Transformer (ViViT)
   - TimeSformer
   - Attention over space and time

**Challenges**:
- Long videos (memory constraints)
- Temporal modeling
- Computational cost
- Limited labeled data

## Optical Flow

**Q: What is optical flow and its applications?**

A:

**Definition**: Motion of pixels between consecutive frames

**Methods**:
- Classical: Lucas-Kanade, Farneback
- Deep learning: FlowNet, PWC-Net, RAFT

**Applications**:
- Action recognition
- Video stabilization
- Object tracking
- Video compression
- Motion analysis

**Challenges**:
- Occlusions
- Large displacements
- Illumination changes
- Real-time processing

## Image Generation

**Q: Explain different approaches to image generation.**

A:

**GANs** (Generative Adversarial Networks):
- Generator creates images
- Discriminator distinguishes real vs fake
- Adversarial training
- StyleGAN, BigGAN for high-quality images
- Challenges: mode collapse, training instability

**VAE** (Variational Autoencoders):
- Encoder: image → latent code
- Decoder: latent code → image
- Probabilistic framework
- Smoother latent space than GANs
- Images often blurry

**Diffusion Models**:
- Add noise gradually (forward process)
- Learn to denoise (reverse process)
- DALL-E 2, Stable Diffusion, Imagen
- State-of-the-art quality
- Slower generation than GANs

**Autoregressive**:
- Generate pixel by pixel
- PixelCNN, PixelRNN
- Slow but high quality

## Multi-Modal Learning

**Q: Explain vision-language models.**

A:

**CLIP** (Contrastive Language-Image Pre-training):
- Joint training on image-text pairs
- Contrastive loss: match correct pairs
- Zero-shot image classification via text prompts
- Applications: image search, classification

**Vision-Language Transformers**:
- ViLT, ALBEF, BLIP
- Cross-modal attention
- Tasks: VQA, image captioning, visual reasoning

**Image Captioning**:
- Encoder: CNN extracts image features
- Decoder: LSTM/Transformer generates caption
- Attention mechanism
- Evaluation: BLEU, CIDEr, SPICE

**Visual Question Answering**:
- Input: image + question
- Output: answer
- Requires visual reasoning
- Attention over image regions

## Data Efficiency

**Q: How do you train CV models with limited data?**

A:

**Strategies**:

1. **Transfer learning**: Pre-trained models
2. **Data augmentation**: Increase effective dataset size
3. **Self-supervised learning**: Learn from unlabeled data
4. **Few-shot learning**: Learn from few examples
5. **Synthetic data**: Generate training data
6. **Active learning**: Select most informative samples
7. **Semi-supervised learning**: Use unlabeled data

**Self-supervised methods**:
- Contrastive learning (SimCLR, MoCo)
- Masked image modeling (MAE)
- Rotation prediction
- Jigsaw puzzles

## Model Compression

**Q: How do you deploy large CV models on edge devices?**

A:

**Techniques**:

1. **Quantization**:
   - FP32 → INT8
   - 4x smaller, 2-4x faster
   - Post-training or quantization-aware training

2. **Pruning**:
   - Remove unimportant weights
   - Structured (channels) or unstructured
   - Iterative pruning + fine-tuning

3. **Knowledge distillation**:
   - Train small student from large teacher
   - Student mimics teacher's outputs

4. **Neural Architecture Search**:
   - Find efficient architectures
   - MobileNet, EfficientNet

5. **Efficient architectures**:
   - Depthwise separable convolutions
   - Inverted residuals
   - Squeeze-and-excitation blocks

**Hardware optimization**:
- TensorRT for NVIDIA GPUs
- CoreML for iOS
- TFLite for mobile
- ONNX for cross-platform



## CV System Design Scenarios

### Scenario 1: Photo Tagging System

**Q: Design an automatic photo tagging system for a social media platform.**

A:

**Requirements**:
- Tag people, objects, scenes, activities
- Process millions of photos daily
- Real-time for new uploads
- Privacy-preserving

**Architecture**:

1. **Multi-task model**:
   - Shared backbone (EfficientNet, ResNet)
   - Multiple heads:
     - Face detection + recognition
     - Object detection (YOLO)
     - Scene classification
     - Activity recognition

2. **Face recognition**:
   - Detect faces in photo
   - Generate embeddings
   - Match against user's face database
   - Suggest tags with confidence scores

3. **Object/scene tagging**:
   - Multi-label classification
   - Threshold tuning per tag
   - Hierarchical tags (animal → dog → golden retriever)

4. **Privacy**:
   - User consent for face recognition
   - Opt-out mechanisms
   - On-device processing where possible
   - Encrypted embeddings

**Serving**:
- Batch processing for uploaded photos
- GPU inference servers
- Caching for popular tags
- Fallback to CPU for overflow

**Evaluation**:
- Precision/recall per tag type
- User acceptance rate of suggestions
- Tagging latency
- User engagement with tagged photos

**Challenges**:
- Occlusions and poor lighting
- Similar-looking people
- Rare objects/scenes
- Bias in training data
- Evolving fashion/trends

### Scenario 2: Visual Search for E-commerce

**Q: Build a visual search system where users can search products using images.**

A:

**Requirements**:
- Search by uploading photo or screenshot
- Find similar products in catalog
- Handle different angles, lighting, backgrounds
- Sub-second latency
- Millions of products

**Architecture**:

1. **Image preprocessing**:
   - Object detection to isolate product
   - Background removal
   - Image normalization

2. **Feature extraction**:
   - CNN backbone (EfficientNet)
   - Global pooling (GeM pooling)
   - 512-dim embedding vector
   - Trained with metric learning

3. **Indexing**:
   - Store product embeddings in FAISS
   - Product quantization for compression
   - Inverted file index for speed
   - Sharding by category

4. **Retrieval**:
   - Encode query image
   - ANN search for top-100 candidates
   - Re-rank with cross-encoder
   - Apply filters (price, brand, availability)

5. **Ranking**:
   - Visual similarity score
   - Product popularity
   - User preferences
   - Business rules (margins, inventory)

**Training**:
- Triplet loss on product images
- Hard negative mining
- Data augmentation (backgrounds, lighting)
- Multi-task: category classification + embedding

**Evaluation**:
- Recall@k (k=10, 50, 100)
- Click-through rate
- Conversion rate
- Search-to-purchase time

**Optimizations**:
- GPU batch inference
- Caching popular queries
- Progressive loading of results
- Mobile-optimized models

**Challenges**:
- Cross-domain: user photos vs catalog images
- Partial views and occlusions
- Similar products (variants)
- Cold start for new products
- Seasonal trends

### Scenario 3: Autonomous Vehicle Perception

**Q: Design the perception system for a self-driving car.**

A:

**Requirements**:
- Detect vehicles, pedestrians, cyclists, traffic signs
- Lane detection and drivable area
- Depth estimation
- Real-time (30+ FPS)
- High accuracy (safety-critical)

**Sensors**:
- Multiple cameras (front, sides, rear)
- LiDAR for 3D information
- Radar for velocity
- Sensor fusion

**Architecture**:

1. **Object detection**:
   - Multi-class detector (YOLOv8, EfficientDet)
   - 3D bounding boxes
   - Velocity estimation
   - Track objects across frames

2. **Semantic segmentation**:
   - Drivable area
   - Lane markings
   - Sidewalks, crosswalks
   - Real-time model (BiSeNet, DDRNet)

3. **Depth estimation**:
   - Monocular depth (if no LiDAR)
   - Stereo matching (if stereo cameras)
   - Fusion with LiDAR

4. **Traffic sign recognition**:
   - Detection + classification
   - OCR for speed limits
   - Temporal consistency

5. **Sensor fusion**:
   - Combine camera, LiDAR, radar
   - Kalman filter for tracking
   - Handle sensor failures

**Processing pipeline**:
- Parallel processing of camera streams
- Multi-task network (detection + segmentation)
- Temporal information (optical flow)
- Post-processing and filtering

**Hardware**:
- NVIDIA Drive platform
- Multiple GPUs/TPUs
- Redundancy for safety

**Evaluation**:
- Detection: mAP, recall at high precision
- Segmentation: IoU
- Latency: end-to-end processing time
- Real-world testing miles

**Challenges**:
- Adverse weather (rain, fog, snow)
- Night-time and low light
- Rare events (accidents, unusual objects)
- Occlusions
- Real-time constraints
- Safety validation

### Scenario 4: Medical Image Analysis

**Q: Build a system to detect diseases from medical images (X-rays, CT scans).**

A:

**Requirements**:
- High accuracy (diagnostic quality)
- Interpretability (explain predictions)
- Handle class imbalance (rare diseases)
- Regulatory compliance (FDA approval)
- Integration with hospital systems

**Architecture**:

1. **Preprocessing**:
   - DICOM format handling
   - Windowing and normalization
   - Artifact removal
   - Image quality checks

2. **Model**:
   - CNN backbone (DenseNet, EfficientNet)
   - Pre-trained on ImageNet, fine-tuned on medical data
   - Multi-label classification (multiple findings)
   - Attention mechanisms for interpretability

3. **Localization**:
   - Grad-CAM for heatmaps
   - Show which regions influenced prediction
   - Bounding boxes for abnormalities

4. **Ensemble**:
   - Multiple models for robustness
   - Different architectures
   - Voting or averaging

5. **Uncertainty estimation**:
   - Monte Carlo dropout
   - Ensemble disagreement
   - Flag uncertain cases for radiologist review

**Training**:
- Large datasets (ChestX-ray14, MIMIC-CXR)
- Class balancing (oversampling, focal loss)
- Data augmentation (careful with medical images)
- Cross-validation
- External validation on different hospitals

**Deployment**:
- PACS integration
- Radiologist workflow integration
- Prioritize urgent cases
- Second opinion mode (assist, not replace)

**Evaluation**:
- AUC-ROC per disease
- Sensitivity/specificity
- Comparison with radiologists
- Clinical validation studies

**Regulatory**:
- FDA 510(k) clearance
- CE marking (Europe)
- Clinical trials
- Continuous monitoring

**Challenges**:
- Limited labeled data
- Label noise (inter-rater disagreement)
- Domain shift (different scanners, hospitals)
- Rare diseases
- Liability and trust
- Bias (demographic representation)

### Scenario 5: Content Moderation

**Q: Design a system to detect inappropriate images (violence, nudity, hate symbols).**

A:

**Requirements**:
- Real-time moderation of user uploads
- Multiple violation types
- High precision (avoid false positives)
- Handle adversarial attacks
- Scale to millions of images/day

**Architecture**:

1. **Multi-stage pipeline**:
   
   **Stage 1: Hash-based filtering**
   - Perceptual hashing (pHash, dHash)
   - Match against known bad content database
   - Instant blocking, no ML needed
   - Handles exact and near-duplicates

   **Stage 2: ML classification**
   - Multi-label classifier
   - Categories: nudity, violence, hate symbols, gore, drugs
   - EfficientNet or ResNet backbone
   - Confidence scores per category

   **Stage 3: Human review**
   - Queue borderline cases (0.4-0.6 confidence)
   - Active learning: retrain on reviewed examples
   - Escalation for complex cases

2. **Adversarial robustness**:
   - Detect image manipulations (blur, crop, rotate)
   - Adversarial training
   - Ensemble models
   - OCR for text in images

3. **Context awareness**:
   - Consider user history
   - Geographic/cultural context
   - Reported content prioritization

**Training**:
- Large datasets (NSFW, violence datasets)
- Synthetic data generation
- Data augmentation (adversarial examples)
- Class balancing
- Regular retraining (new attack patterns)

**Serving**:
- GPU inference clusters
- Load balancing
- Caching for repeat uploads
- Graceful degradation (queue if overloaded)

**Evaluation**:
- Precision/recall per category
- False positive rate (critical)
- Processing latency
- Human review queue size
- User appeals and accuracy

**Challenges**:
- Cultural differences (what's inappropriate varies)
- Context-dependent (art vs pornography)
- Adversarial users (obfuscation techniques)
- Evolving content (new memes, symbols)
- Psychological impact on reviewers
- Bias and fairness

**Privacy**:
- No storage of violating content (hashes only)
- Encrypted processing
- Audit logs
- Compliance with regulations

### Scenario 6: Document OCR and Understanding

**Q: Build a system to extract information from documents (invoices, receipts, forms).**

A:

**Requirements**:
- Extract text and structure
- Handle various layouts and formats
- Multiple languages
- Accuracy > 95%
- Process scanned and mobile photos

**Architecture**:

1. **Image preprocessing**:
   - Deskewing and rotation correction
   - Binarization
   - Noise removal
   - Super-resolution for low-quality images

2. **Text detection**:
   - Detect text regions (EAST, CRAFT)
   - Handle multi-oriented text
   - Table detection

3. **Text recognition (OCR)**:
   - CRNN or Transformer-based
   - Attention mechanism
   - Language model for correction
   - Tesseract as baseline

4. **Layout analysis**:
   - Segment into regions (header, body, footer)
   - Reading order detection
   - Table structure recognition

5. **Information extraction**:
   - Named entity recognition
   - Key-value pair extraction (invoice number, date, total)
   - Template matching for structured documents
   - BERT for understanding

6. **Post-processing**:
   - Spell checking
   - Format validation (dates, amounts)
   - Business logic (totals should match)

**Training**:
- Synthetic data generation (render documents)
- Real scanned documents
- Data augmentation (blur, noise, perspective)
- Multi-task learning (detection + recognition)

**Serving**:
- Batch processing for bulk uploads
- Real-time API for mobile apps
- Confidence scores for manual review
- Human-in-the-loop for corrections

**Evaluation**:
- Character error rate (CER)
- Word error rate (WER)
- Field extraction accuracy
- End-to-end accuracy

**Challenges**:
- Handwritten text
- Poor image quality
- Complex layouts
- Multi-column documents
- Tables with merged cells
- Multiple languages in same document

**Applications**:
- Invoice processing
- Receipt scanning
- Form digitization
- ID card extraction
- License plate recognition

## Advanced CV Topics

**Q: Explain attention mechanisms in computer vision.**

A:

**Spatial attention**:
- Learn where to focus in image
- Attention map over spatial locations
- Multiply features by attention weights

**Channel attention**:
- Learn which feature channels are important
- Squeeze-and-Excitation (SE) blocks
- Global pooling → FC layers → sigmoid → scale

**Self-attention**:
- Non-local neural networks
- Each position attends to all positions
- Captures long-range dependencies
- Used in Vision Transformers

**Cross-attention**:
- Attend from one modality to another
- Vision-language models
- Query from text, key/value from image

**Q: What is domain adaptation in computer vision?**

A:

**Problem**: Model trained on source domain performs poorly on target domain

**Approaches**:

1. **Fine-tuning**: Train on small target domain data

2. **Domain adversarial training**:
   - Feature extractor + domain classifier
   - Learn domain-invariant features
   - Adversarial loss

3. **Style transfer**:
   - Transform source images to target style
   - CycleGAN, AdaIN
   - Train on transformed images

4. **Self-training**:
   - Pseudo-labeling on target domain
   - Iterative refinement

**Applications**:
- Synthetic to real (sim-to-real)
- Different cameras/lighting
- Geographic adaptation

**Q: How do you handle class imbalance in object detection?**

A:

**Problem**: Background class dominates, rare objects underrepresented

**Solutions**:

1. **Focal loss**:
   - Down-weight easy examples
   - Focus on hard examples
   - FL = -(1-p)^γ * log(p)

2. **Hard negative mining**:
   - Select hard background examples
   - Balance positive/negative ratio

3. **Data augmentation**:
   - Oversample rare classes
   - Copy-paste rare objects

4. **Class-balanced loss**:
   - Weight loss by inverse class frequency

5. **Two-stage training**:
   - Train on balanced subset
   - Fine-tune on full data

**Q: Explain test-time augmentation and its benefits.**

A:

**Technique**: Apply augmentations at inference time, average predictions

**Process**:
1. Generate multiple augmented versions (flip, crop, rotate)
2. Run model on each version
3. Average predictions (or vote for classification)

**Benefits**:
- Improved accuracy (1-2% typically)
- More robust predictions
- Uncertainty estimation

**Trade-off**: Slower inference (N times slower for N augmentations)

**Use cases**: Competitions, high-stakes applications, when accuracy > speed
