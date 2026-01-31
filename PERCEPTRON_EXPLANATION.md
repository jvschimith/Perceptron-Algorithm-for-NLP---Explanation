# Perceptron Algorithm for NLP - Explanation

## Overview
This implementation demonstrates a **perceptron classifier** for Natural Language Processing (NLP), specifically for binary sentiment analysis of movie reviews.

## How the Perceptron Works

### 1. **Architecture**
```
Input Layer (Text Features) → Weighted Sum → Activation Function → Output (0 or 1)
```

The perceptron is the simplest form of neural network with:
- **Input**: Feature vector representing text
- **Weights**: Learned parameters (one per feature)
- **Bias**: Single learned offset parameter
- **Activation**: Step function (0 if sum < 0, else 1)

### 2. **Text Preprocessing Pipeline**

#### Tokenization
```
"This movie was great!" → ["this", "movie", "was", "great"]
```

#### Vocabulary Building
- Extract all words from training data
- Keep only the most frequent N words (e.g., 1000)
- Create word-to-index mapping

#### Vectorization (Bag-of-Words)
```
Text: "great movie great acting"
Vocabulary: {great: 0, movie: 1, acting: 2, bad: 3}
Vector: [2, 1, 1, 0]  # Counts of each word
```

### 3. **Training Algorithm**

The perceptron uses the **Perceptron Learning Rule**:

```
For each training example (x, y):
  1. Make prediction: ŷ = sign(w·x + b)
  2. If ŷ ≠ y (prediction is wrong):
     - Update: w = w + η(y - ŷ)x
     - Update: b = b + η(y - ŷ)
```

Where:
- `w` = weight vector
- `x` = input feature vector
- `b` = bias
- `η` = learning rate
- `y` = true label (0 or 1)
- `ŷ` = predicted label

### 4. **Key Components**

#### Feature Extraction
```python
def text_to_vector(self, text):
    vector = np.zeros(len(self.vocabulary))
    tokens = self.tokenize(text)
    for token in tokens:
        if token in self.vocabulary:
            idx = self.vocabulary[token]
            vector[idx] += 1  # Count frequency
    return vector
```

#### Prediction
```python
def predict(self, x):
    activation = np.dot(x, self.weights) + self.bias
    return 1 if activation >= 0 else 0
```

#### Weight Update
```python
if prediction != y[i]:
    update = self.learning_rate * (y[i] - prediction)
    self.weights += update * X[i]
    self.bias += update
```

## Example Walkthrough

### Training Data
```
Positive reviews (label=1):
- "This movie was absolutely wonderful"
- "Great acting and brilliant storyline"

Negative reviews (label=0):
- "This film was terrible and boring"
- "Awful movie with poor acting"
```

### Learning Process

**Initial state**: All weights = 0, bias = 0

**Iteration 1**:
- Input: "wonderful movie" → vector: [0, 1, 0, 1, 0]
- Prediction: 0 (wrong, should be 1)
- Update weights for "wonderful" and "movie" positively

**Iteration 2**:
- Input: "terrible movie" → vector: [1, 1, 0, 0, 0]
- Prediction: 1 (wrong, should be 0)
- Update weights for "terrible" and "movie" negatively

**After convergence**:
- Positive words (wonderful, great) have positive weights
- Negative words (terrible, awful) have negative weights

## Mathematical Formulation

### Decision Boundary
```
f(x) = sign(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

Where:
- x₁, x₂, ..., xₙ are word frequencies
- w₁, w₂, ..., wₙ are learned weights
- b is the bias term

### Convergence Theorem
The perceptron is guaranteed to converge if:
1. The data is **linearly separable**
2. Learning rate > 0

## Advantages

1. **Simple**: Easy to understand and implement
2. **Fast**: O(n) training time per iteration
3. **Interpretable**: Weights show word importance
4. **Online Learning**: Can update with new data incrementally

## Limitations

1. **Linear Only**: Cannot learn non-linear patterns
2. **No Probabilistic Output**: Only binary classification
3. **Sensitive to Outliers**: Misclassified points heavily influence learning
4. **Requires Linearly Separable Data**: Won't converge otherwise

## Improvements & Extensions

### 1. **Better Feature Extraction**
- TF-IDF instead of raw counts
- N-grams (bigrams, trigrams)
- Word embeddings (Word2Vec, GloVe)

### 2. **Multi-class Classification**
- One-vs-all approach
- Multiple perceptrons

### 3. **Advanced Algorithms**
- Multi-layer perceptron (MLP)
- Support Vector Machines (SVM)
- Logistic Regression (adds sigmoid activation)

## Usage Example

```python
# Create perceptron
perceptron = TextPerceptron(learning_rate=0.1, epochs=100)

# Train on movie reviews
train_texts = ["Great movie!", "Terrible film", ...]
train_labels = [1, 0, ...]
perceptron.fit(train_texts, train_labels)

# Make predictions
test_text = "This was an amazing film"
prediction = perceptron.predict(perceptron.text_to_vector(test_text))
# prediction = 1 (positive)
```

## Performance Metrics

From the example output:
- **Training Accuracy**: 100% (converged in 3 epochs)
- **Test Accuracy**: 100%
- **Vocabulary Size**: 65 words
- **Most Important Features**: "disappointing" (negative), "amazing" (positive)

## When to Use Perceptron for NLP

**Good for**:
- Binary text classification
- Quick baseline models
- Educational purposes
- Simple, interpretable models

**Better alternatives**:
- Logistic Regression (probabilistic outputs)
- Naive Bayes (works well for text)
- Neural Networks (complex patterns)
- Transformers (state-of-the-art, e.g., BERT)

## Conclusion

The perceptron provides a solid foundation for understanding how machine learning models process text data. While modern NLP uses more sophisticated approaches, the core concepts—feature extraction, weight learning, and linear combination—remain fundamental to all neural network architectures.
