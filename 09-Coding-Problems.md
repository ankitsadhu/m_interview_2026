# ML Coding Problems

## Implement Gradient Descent

**Q: Implement gradient descent for linear regression.**

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(epochs):
        # Predictions
        predictions = X @ theta
        
        # Compute gradient
        gradient = (1/m) * X.T @ (predictions - y)
        
        # Update parameters
        theta -= learning_rate * gradient
    
    return theta
```

## K-Means Clustering

**Q: Implement K-means from scratch.**

```python
import numpy as np

def kmeans(X, k, max_iters=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(len(X), k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) 
                                   for i in range(k)])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
    
    return labels, centroids
```

## Decision Tree

**Q: Implement a simple decision tree classifier.**

```python
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
    
    def gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        return X[left_mask], y[left_mask], X[~left_mask], y[~left_mask]
    
    def find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self.split(X, y, feature, threshold)
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                gini = (len(y_left) * self.gini_impurity(y_left) + 
                        len(y_right) * self.gini_impurity(y_right)) / len(y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
```

## Softmax and Cross-Entropy

**Q: Implement softmax and cross-entropy loss.**

```python
import numpy as np

def softmax(logits):
    # Subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    # targets: one-hot encoded
    # Clip to avoid log(0)
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
    return -np.mean(np.sum(targets * np.log(predictions), axis=1))
```

## Batch Normalization

**Q: Implement batch normalization forward pass.**

```python
import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    # x: (batch_size, features)
    # gamma, beta: (features,)
    
    # Compute mean and variance
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    
    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # Scale and shift
    out = gamma * x_norm + beta
    
    # Cache for backward pass
    cache = (x, x_norm, mean, var, gamma, beta, eps)
    
    return out, cache
```


## Logistic Regression

**Q: Implement logistic regression from scratch.**

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            # Forward pass
            linear_pred = X @ self.weights + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * X.T @ (predictions - y)
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_pred = X @ self.weights + self.bias
        y_pred = self.sigmoid(linear_pred)
        return (y_pred >= 0.5).astype(int)
```

## Principal Component Analysis (PCA)

**Q: Implement PCA from scratch.**

```python
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]
    
    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
```

## Naive Bayes Classifier

**Q: Implement Gaussian Naive Bayes.**

```python
import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = len(X_c) / len(X)
    
    def _gaussian_pdf(self, x, mean, var):
        eps = 1e-6  # Avoid division by zero
        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        return coeff * exponent
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self._gaussian_pdf(
                    x, self.mean[c], self.var[c])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
```

## Neural Network from Scratch

**Q: Implement a simple neural network with one hidden layer.**

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        # Hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dz1 = (dz2 @ self.W2.T) * self.relu_derivative(self.z1)
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
```

## Convolutional Layer

**Q: Implement a 2D convolution operation.**

```python
import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """
    image: (H, W, C_in)
    kernel: (K, K, C_in, C_out)
    """
    H, W, C_in = image.shape
    K, _, _, C_out = kernel.shape
    
    # Add padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), 
                      mode='constant')
    
    H_padded, W_padded, _ = image.shape
    
    # Output dimensions
    H_out = (H_padded - K) // stride + 1
    W_out = (W_padded - K) // stride + 1
    
    output = np.zeros((H_out, W_out, C_out))
    
    # Perform convolution
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            
            # Extract patch
            patch = image[h_start:h_start+K, w_start:w_start+K, :]
            
            # Convolve with each filter
            for c in range(C_out):
                output[i, j, c] = np.sum(patch * kernel[:, :, :, c])
    
    return output

def max_pool2d(image, pool_size=2, stride=2):
    """
    image: (H, W, C)
    """
    H, W, C = image.shape
    
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    output = np.zeros((H_out, W_out, C))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            
            patch = image[h_start:h_start+pool_size, 
                         w_start:w_start+pool_size, :]
            output[i, j, :] = np.max(patch, axis=(0, 1))
    
    return output
```

## Attention Mechanism

**Q: Implement scaled dot-product attention.**

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: Query matrix (batch_size, seq_len, d_k)
    K: Key matrix (batch_size, seq_len, d_k)
    V: Value matrix (batch_size, seq_len, d_v)
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores + (mask * -1e9)
    
    # Apply softmax
    attention_weights = softmax(scores, axis=-1)
    
    # Compute weighted sum of values
    output = attention_weights @ V
    
    return output, attention_weights

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

## LSTM Cell

**Q: Implement an LSTM cell forward pass.**

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def lstm_cell_forward(x_t, h_prev, c_prev, Wf, Uf, bf, Wi, Ui, bi, 
                      Wc, Uc, bc, Wo, Uo, bo):
    """
    x_t: input at time t (batch_size, input_size)
    h_prev: previous hidden state (batch_size, hidden_size)
    c_prev: previous cell state (batch_size, hidden_size)
    W*, U*, b*: weight matrices and biases
    """
    
    # Forget gate
    f_t = sigmoid(x_t @ Wf + h_prev @ Uf + bf)
    
    # Input gate
    i_t = sigmoid(x_t @ Wi + h_prev @ Ui + bi)
    
    # Candidate cell state
    c_tilde = tanh(x_t @ Wc + h_prev @ Uc + bc)
    
    # Update cell state
    c_t = f_t * c_prev + i_t * c_tilde
    
    # Output gate
    o_t = sigmoid(x_t @ Wo + h_prev @ Uo + bo)
    
    # Hidden state
    h_t = o_t * tanh(c_t)
    
    cache = (x_t, h_prev, c_prev, f_t, i_t, c_tilde, o_t, c_t)
    
    return h_t, c_t, cache
```

## Triplet Loss

**Q: Implement triplet loss for metric learning.**

```python
import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    anchor: embeddings of anchor samples (batch_size, embedding_dim)
    positive: embeddings of positive samples (batch_size, embedding_dim)
    negative: embeddings of negative samples (batch_size, embedding_dim)
    margin: margin for triplet loss
    """
    # Compute distances
    pos_dist = np.sum((anchor - positive) ** 2, axis=1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=1)
    
    # Triplet loss
    loss = np.maximum(pos_dist - neg_dist + margin, 0)
    
    return np.mean(loss)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

def cosine_similarity(x1, x2):
    dot_product = np.sum(x1 * x2, axis=1)
    norm1 = np.linalg.norm(x1, axis=1)
    norm2 = np.linalg.norm(x2, axis=1)
    return dot_product / (norm1 * norm2 + 1e-8)
```

## Non-Maximum Suppression (NMS)

**Q: Implement NMS for object detection.**

```python
import numpy as np

def compute_iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    boxes: (N, 4) array of [x1, y1, x2, y2]
    scores: (N,) array of confidence scores
    """
    # Sort by scores (descending)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        # Keep box with highest score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = np.array([compute_iou(boxes[current], boxes[i]) 
                        for i in indices[1:]])
        
        # Keep boxes with IoU below threshold
        indices = indices[1:][ious < iou_threshold]
    
    return keep
```

## Beam Search

**Q: Implement beam search for sequence generation.**

```python
import numpy as np
from heapq import heappush, heappop

def beam_search(model, start_token, end_token, beam_width=3, max_length=20):
    """
    model: function that takes sequence and returns next token probabilities
    start_token: token to start generation
    end_token: token to end generation
    """
    # Initialize beam with start token
    beams = [(0.0, [start_token])]  # (score, sequence)
    completed = []
    
    for _ in range(max_length):
        candidates = []
        
        for score, sequence in beams:
            # Skip if sequence is complete
            if sequence[-1] == end_token:
                completed.append((score, sequence))
                continue
            
            # Get next token probabilities
            probs = model(sequence)
            
            # Get top-k tokens
            top_k_indices = np.argsort(probs)[-beam_width:]
            
            for idx in top_k_indices:
                new_score = score + np.log(probs[idx] + 1e-10)
                new_sequence = sequence + [idx]
                candidates.append((new_score, new_sequence))
        
        # Select top beam_width candidates
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        
        # Check if all beams are complete
        if all(seq[-1] == end_token for _, seq in beams):
            completed.extend(beams)
            break
    
    # Return best sequence
    if completed:
        return max(completed, key=lambda x: x[0])[1]
    else:
        return max(beams, key=lambda x: x[0])[1]
```

## Data Augmentation

**Q: Implement common image augmentation techniques.**

```python
import numpy as np

def random_flip(image, horizontal=True):
    """Randomly flip image horizontally or vertically."""
    if np.random.rand() > 0.5:
        if horizontal:
            return np.fliplr(image)
        else:
            return np.flipud(image)
    return image

def random_crop(image, crop_size):
    """Randomly crop image to crop_size."""
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    if h < crop_h or w < crop_w:
        return image
    
    top = np.random.randint(0, h - crop_h)
    left = np.random.randint(0, w - crop_w)
    
    return image[top:top+crop_h, left:left+crop_w]

def random_brightness(image, max_delta=0.2):
    """Randomly adjust brightness."""
    delta = np.random.uniform(-max_delta, max_delta)
    image = image + delta
    return np.clip(image, 0, 1)

def random_rotation(image, max_angle=15):
    """Rotate image by random angle."""
    angle = np.random.uniform(-max_angle, max_angle)
    # Simplified rotation (use cv2.getRotationMatrix2D in practice)
    return image  # Placeholder

def cutout(image, mask_size=16):
    """Apply cutout augmentation."""
    h, w = image.shape[:2]
    
    # Random position
    y = np.random.randint(0, h)
    x = np.random.randint(0, w)
    
    # Calculate mask boundaries
    y1 = max(0, y - mask_size // 2)
    y2 = min(h, y + mask_size // 2)
    x1 = max(0, x - mask_size // 2)
    x2 = min(w, x + mask_size // 2)
    
    # Apply mask
    image_aug = image.copy()
    image_aug[y1:y2, x1:x2] = 0
    
    return image_aug

def mixup(image1, image2, label1, label2, alpha=0.2):
    """Apply mixup augmentation."""
    lam = np.random.beta(alpha, alpha)
    mixed_image = lam * image1 + (1 - lam) * image2
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_image, mixed_label
```

## Precision, Recall, F1

**Q: Implement evaluation metrics from scratch.**

```python
import numpy as np

def confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm

def precision_recall_f1(y_true, y_pred, average='binary'):
    """
    Compute precision, recall, and F1 score.
    average: 'binary', 'macro', 'micro'
    """
    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return precision, recall, f1
    
    elif average == 'macro':
        classes = np.unique(y_true)
        precisions, recalls, f1s = [], [], []
        
        for c in classes:
            y_true_c = (y_true == c).astype(int)
            y_pred_c = (y_pred == c).astype(int)
            p, r, f = precision_recall_f1(y_true_c, y_pred_c, average='binary')
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        
        return np.mean(precisions), np.mean(recalls), np.mean(f1s)
    
    elif average == 'micro':
        tp = np.sum(y_true == y_pred)
        total = len(y_true)
        accuracy = tp / total
        return accuracy, accuracy, accuracy

def roc_auc_score(y_true, y_scores):
    """Compute ROC AUC score."""
    # Sort by scores
    indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[indices]
    
    # Compute TPR and FPR at different thresholds
    tpr_list, fpr_list = [0], [0]
    
    tp = 0
    fp = 0
    total_pos = np.sum(y_true)
    total_neg = len(y_true) - total_pos
    
    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / total_pos
        fpr = fp / total_neg
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Compute AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    
    return auc
```

## Learning Rate Schedulers

**Q: Implement common learning rate schedules.**

```python
import numpy as np

class StepLR:
    def __init__(self, initial_lr, step_size, gamma=0.1):
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self, epoch):
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))

class ExponentialLR:
    def __init__(self, initial_lr, gamma=0.95):
        self.initial_lr = initial_lr
        self.gamma = gamma
    
    def get_lr(self, epoch):
        return self.initial_lr * (self.gamma ** epoch)

class CosineAnnealingLR:
    def __init__(self, initial_lr, T_max, eta_min=0):
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self, epoch):
        return self.eta_min + (self.initial_lr - self.eta_min) * \
               (1 + np.cos(np.pi * epoch / self.T_max)) / 2

class WarmupLR:
    def __init__(self, initial_lr, warmup_epochs, target_lr):
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
    
    def get_lr(self, epoch):
        if epoch < self.warmup_epochs:
            return self.initial_lr + (self.target_lr - self.initial_lr) * \
                   epoch / self.warmup_epochs
        return self.target_lr
```

## Mini-Batch Gradient Descent

**Q: Implement mini-batch gradient descent with momentum.**

```python
import numpy as np

class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params, grads):
        if not self.velocity:
            for key in params:
                self.velocity[key] = np.zeros_like(params[key])
        
        for key in params:
            self.velocity[key] = self.momentum * self.velocity[key] - \
                                self.lr * grads[key]
            params[key] += self.velocity[key]

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def update(self, params, grads):
        if not self.m:
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
        
        self.t += 1
        
        for key in params:
            # Update biased first moment
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second moment
            self.v[key] = self.beta2 * self.v[key] + \
                         (1 - self.beta2) * (grads[key] ** 2)
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

## Cross-Validation

**Q: Implement k-fold cross-validation.**

```python
import numpy as np

def k_fold_split(X, y, k=5, shuffle=True):
    """
    Split data into k folds for cross-validation.
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    fold_size = n_samples // k
    folds = []
    
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples
        
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        folds.append((X_train, X_test, y_train, y_test))
    
    return folds

def cross_validate(model, X, y, k=5):
    """
    Perform k-fold cross-validation.
    """
    folds = k_fold_split(X, y, k)
    scores = []
    
    for X_train, X_test, y_train, y_test in folds:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        scores.append(accuracy)
    
    return np.mean(scores), np.std(scores)
```


## Scenario-Based Coding Problems

### Scenario 1: Build a Mini Recommendation Engine

**Q: Implement a collaborative filtering recommendation system.**

```python
import numpy as np

class CollaborativeFiltering:
    def __init__(self, n_factors=20, learning_rate=0.01, regularization=0.01):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, ratings_matrix, epochs=100):
        """
        ratings_matrix: (n_users, n_items) with 0 for missing ratings
        """
        n_users, n_items = ratings_matrix.shape
        
        # Initialize factor matrices
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Get indices of known ratings
        known_ratings = ratings_matrix > 0
        
        for epoch in range(epochs):
            for i in range(n_users):
                for j in range(n_items):
                    if known_ratings[i, j]:
                        # Compute prediction
                        pred = self.user_factors[i] @ self.item_factors[j]
                        error = ratings_matrix[i, j] - pred
                        
                        # Update factors with gradient descent
                        user_grad = -2 * error * self.item_factors[j] + \
                                   2 * self.reg * self.user_factors[i]
                        item_grad = -2 * error * self.user_factors[i] + \
                                   2 * self.reg * self.item_factors[j]
                        
                        self.user_factors[i] -= self.lr * user_grad
                        self.item_factors[j] -= self.lr * item_grad
            
            if epoch % 10 == 0:
                rmse = self.compute_rmse(ratings_matrix, known_ratings)
                print(f"Epoch {epoch}, RMSE: {rmse:.4f}")
    
    def compute_rmse(self, ratings_matrix, known_ratings):
        predictions = self.user_factors @ self.item_factors.T
        errors = (ratings_matrix - predictions)[known_ratings]
        return np.sqrt(np.mean(errors ** 2))
    
    def predict(self, user_id, item_id):
        return self.user_factors[user_id] @ self.item_factors[item_id]
    
    def recommend(self, user_id, n=10, exclude_rated=True):
        """Recommend top-n items for user."""
        scores = self.user_factors[user_id] @ self.item_factors.T
        
        if exclude_rated:
            # Exclude already rated items
            rated_items = np.where(ratings_matrix[user_id] > 0)[0]
            scores[rated_items] = -np.inf
        
        top_items = np.argsort(scores)[::-1][:n]
        return top_items, scores[top_items]

# Usage example
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4],
])

cf = CollaborativeFiltering(n_factors=2)
cf.fit(ratings, epochs=100)
recommendations, scores = cf.recommend(user_id=0, n=2)
print(f"Recommended items: {recommendations}")
```

### Scenario 2: Implement Text Preprocessing Pipeline

**Q: Build a text preprocessing pipeline for NLP.**

```python
import re
from collections import Counter
import numpy as np

class TextPreprocessor:
    def __init__(self, max_vocab_size=10000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def clean_text(self, text):
        """Basic text cleaning."""
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """Simple whitespace tokenization."""
        return text.split()
    
    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = self.tokenize(cleaned)
            word_counts.update(tokens)
        
        # Filter by minimum frequency
        word_counts = {word: count for word, count in word_counts.items() 
                      if count >= self.min_freq}
        
        # Sort by frequency and take top max_vocab_size
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], 
                            reverse=True)[:self.max_vocab_size]
        
        # Build vocabulary mappings
        self.vocab = dict(sorted_words)
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        
        for idx, (word, _) in enumerate(sorted_words, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.idx_to_word[0] = '<PAD>'
        self.idx_to_word[1] = '<UNK>'
    
    def text_to_sequence(self, text, max_length=None):
        """Convert text to sequence of indices."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        
        sequence = [self.word_to_idx.get(token, 1) for token in tokens]
        
        if max_length:
            if len(sequence) < max_length:
                # Pad
                sequence = sequence + [0] * (max_length - len(sequence))
            else:
                # Truncate
                sequence = sequence[:max_length]
        
        return sequence
    
    def sequences_to_matrix(self, sequences):
        """Convert sequences to matrix."""
        return np.array(sequences)

# Usage
texts = [
    "I love machine learning!",
    "Deep learning is amazing.",
    "Natural language processing is fun."
]

preprocessor = TextPreprocessor(max_vocab_size=100)
preprocessor.build_vocab(texts)

sequences = [preprocessor.text_to_sequence(text, max_length=10) 
            for text in texts]
print("Sequences:", sequences)
print("Vocabulary size:", len(preprocessor.word_to_idx))
```

### Scenario 3: Implement Feature Scaling

**Q: Implement different feature scaling methods.**

```python
import numpy as np

class FeatureScaler:
    def __init__(self, method='standard'):
        """
        method: 'standard', 'minmax', 'robust', 'maxabs'
        """
        self.method = method
        self.params = {}
    
    def fit(self, X):
        """Compute scaling parameters."""
        if self.method == 'standard':
            # Standardization: (x - mean) / std
            self.params['mean'] = np.mean(X, axis=0)
            self.params['std'] = np.std(X, axis=0)
        
        elif self.method == 'minmax':
            # Min-Max scaling: (x - min) / (max - min)
            self.params['min'] = np.min(X, axis=0)
            self.params['max'] = np.max(X, axis=0)
        
        elif self.method == 'robust':
            # Robust scaling: (x - median) / IQR
            self.params['median'] = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            self.params['iqr'] = q75 - q25
        
        elif self.method == 'maxabs':
            # Max absolute scaling: x / max(abs(x))
            self.params['max_abs'] = np.max(np.abs(X), axis=0)
    
    def transform(self, X):
        """Apply scaling transformation."""
        if self.method == 'standard':
            return (X - self.params['mean']) / (self.params['std'] + 1e-8)
        
        elif self.method == 'minmax':
            return (X - self.params['min']) / \
                   (self.params['max'] - self.params['min'] + 1e-8)
        
        elif self.method == 'robust':
            return (X - self.params['median']) / (self.params['iqr'] + 1e-8)
        
        elif self.method == 'maxabs':
            return X / (self.params['max_abs'] + 1e-8)
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled):
        """Reverse the scaling."""
        if self.method == 'standard':
            return X_scaled * self.params['std'] + self.params['mean']
        
        elif self.method == 'minmax':
            return X_scaled * (self.params['max'] - self.params['min']) + \
                   self.params['min']
        
        elif self.method == 'robust':
            return X_scaled * self.params['iqr'] + self.params['median']
        
        elif self.method == 'maxabs':
            return X_scaled * self.params['max_abs']

# Usage
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

scaler = FeatureScaler(method='standard')
X_scaled = scaler.fit_transform(X)
print("Scaled:", X_scaled)
X_original = scaler.inverse_transform(X_scaled)
print("Original:", X_original)
```

### Scenario 4: Implement Train-Test Split with Stratification

**Q: Implement stratified train-test split.**

```python
import numpy as np

def train_test_split(X, y, test_size=0.2, stratify=True, random_state=None):
    """
    Split data into train and test sets.
    If stratify=True, maintain class distribution.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if not stratify:
        # Simple random split
        np.random.shuffle(indices)
        split_idx = int(n_samples * (1 - test_size))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
    
    else:
        # Stratified split
        classes = np.unique(y)
        train_indices = []
        test_indices = []
        
        for c in classes:
            # Get indices for this class
            class_indices = indices[y == c]
            n_class = len(class_indices)
            
            # Shuffle class indices
            np.random.shuffle(class_indices)
            
            # Split this class
            split_idx = int(n_class * (1 - test_size))
            train_indices.extend(class_indices[:split_idx])
            test_indices.extend(class_indices[split_idx:])
        
        # Shuffle the final indices
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Usage
X = np.random.randn(100, 5)
y = np.random.randint(0, 3, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=True, random_state=42
)

print("Train class distribution:", np.bincount(y_train))
print("Test class distribution:", np.bincount(y_test))
```

### Scenario 5: Implement Ensemble Methods

**Q: Implement bagging and boosting from scratch.**

```python
import numpy as np

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
    
    def fit(self, X, y):
        n_samples = len(X)
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Train estimator
            estimator = self.base_estimator()
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)
    
    def predict(self, X):
        # Get predictions from all estimators
        predictions = np.array([est.predict(X) for est in self.estimators])
        
        # Majority vote
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=0, 
            arr=predictions
        )

class AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []
    
    def fit(self, X, y):
        n_samples = len(X)
        
        # Initialize weights
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # Train weak learner (decision stump)
            estimator = self._train_weak_learner(X, y, weights)
            
            # Get predictions
            predictions = estimator.predict(X)
            
            # Compute error
            incorrect = predictions != y
            error = np.sum(weights[incorrect]) / np.sum(weights)
            
            # Compute estimator weight
            estimator_weight = 0.5 * np.log((1 - error) / (error + 1e-10))
            
            # Update sample weights
            weights *= np.exp(-estimator_weight * y * predictions)
            weights /= np.sum(weights)
            
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
    
    def _train_weak_learner(self, X, y, weights):
        # Simple decision stump (one-level decision tree)
        # This is a placeholder - implement actual decision stump
        class DecisionStump:
            def __init__(self):
                self.feature = 0
                self.threshold = 0
                self.polarity = 1
            
            def fit(self, X, y, weights):
                # Find best split
                best_error = float('inf')
                
                for feature in range(X.shape[1]):
                    thresholds = np.unique(X[:, feature])
                    
                    for threshold in thresholds:
                        for polarity in [1, -1]:
                            predictions = np.ones(len(X))
                            predictions[polarity * X[:, feature] < 
                                      polarity * threshold] = -1
                            
                            error = np.sum(weights[predictions != y])
                            
                            if error < best_error:
                                best_error = error
                                self.feature = feature
                                self.threshold = threshold
                                self.polarity = polarity
            
            def predict(self, X):
                predictions = np.ones(len(X))
                predictions[self.polarity * X[:, self.feature] < 
                          self.polarity * self.threshold] = -1
                return predictions
        
        stump = DecisionStump()
        stump.fit(X, y, weights)
        return stump
    
    def predict(self, X):
        # Weighted vote
        predictions = np.zeros(len(X))
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            predictions += weight * estimator.predict(X)
        
        return np.sign(predictions)
```

### Scenario 6: Implement Data Loader with Batching

**Q: Implement a data loader for mini-batch training.**

```python
import numpy as np

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            
            batch_indices = indices[start_idx:end_idx]
            
            yield self.X[batch_indices], self.y[batch_indices]
    
    def __len__(self):
        return self.n_batches

# Usage
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

loader = DataLoader(X, y, batch_size=16, shuffle=True)

for epoch in range(3):
    print(f"\nEpoch {epoch + 1}")
    for batch_idx, (X_batch, y_batch) in enumerate(loader):
        print(f"Batch {batch_idx + 1}: X shape {X_batch.shape}, "
              f"y shape {y_batch.shape}")
        # Train on batch here
```

## Advanced Coding Challenges

### Challenge 1: Implement Gradient Checking

**Q: Implement numerical gradient checking for debugging.**

```python
import numpy as np

def numerical_gradient(f, x, epsilon=1e-5):
    """
    Compute numerical gradient using finite differences.
    f: function that takes x and returns scalar loss
    x: parameters
    """
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        # f(x + epsilon)
        x[idx] = old_value + epsilon
        f_plus = f(x)
        
        # f(x - epsilon)
        x[idx] = old_value - epsilon
        f_minus = f(x)
        
        # Compute gradient
        grad[idx] = (f_plus - f_minus) / (2 * epsilon)
        
        # Restore original value
        x[idx] = old_value
        it.iternext()
    
    return grad

def gradient_check(f, x, analytical_grad, epsilon=1e-5, threshold=1e-7):
    """
    Check if analytical gradient is correct.
    """
    numerical_grad = numerical_gradient(f, x, epsilon)
    
    # Compute relative error
    numerator = np.linalg.norm(numerical_grad - analytical_grad)
    denominator = np.linalg.norm(numerical_grad) + \
                  np.linalg.norm(analytical_grad)
    relative_error = numerator / (denominator + 1e-10)
    
    if relative_error < threshold:
        print(f"Gradient check passed! Relative error: {relative_error:.2e}")
        return True
    else:
        print(f"Gradient check FAILED! Relative error: {relative_error:.2e}")
        print(f"Numerical gradient: {numerical_grad}")
        print(f"Analytical gradient: {analytical_grad}")
        return False

# Example usage
def loss_function(w):
    # Simple quadratic loss: (w - 3)^2
    return np.sum((w - 3) ** 2)

def analytical_gradient(w):
    return 2 * (w - 3)

w = np.array([1.0, 2.0, 3.0])
grad = analytical_gradient(w)
gradient_check(loss_function, w, grad)
```

### Challenge 2: Implement Early Stopping

**Q: Implement early stopping for training.**

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='min'):
        """
        patience: number of epochs to wait before stopping
        min_delta: minimum change to qualify as improvement
        mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, score, model=None):
        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.best_model = self._copy_model(model)
        
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            if model is not None:
                self.best_model = self._copy_model(model)
        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
    
    def _copy_model(self, model):
        # Deep copy model (implementation depends on framework)
        import copy
        return copy.deepcopy(model)

# Usage in training loop
early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='min')

for epoch in range(100):
    # Training code here
    train_loss = 0.5  # placeholder
    val_loss = 0.6    # placeholder
    
    # Check early stopping
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

This comprehensive coding problems file now includes fundamental implementations, scenario-based problems, and advanced challenges that cover the breadth of ML coding questions asked at MANG companies.
