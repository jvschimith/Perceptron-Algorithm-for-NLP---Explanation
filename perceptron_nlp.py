"""
Perceptron Algorithm for NLP Text Classification
This example demonstrates sentiment analysis on movie reviews
"""

import numpy as np
from collections import Counter
import re


class TextPerceptron:
    """
    A simple perceptron classifier for text data.
    Uses binary sentiment classification (positive/negative).
    """
    
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.vocabulary = {}
        
    def tokenize(self, text):
        """Simple tokenization: lowercase and split on non-alphanumeric"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def build_vocabulary(self, texts, max_features=1000):
        """Build vocabulary from training texts"""
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))
        
        # Get most common words
        word_counts = Counter(all_tokens)
        most_common = word_counts.most_common(max_features)
        
        # Create word to index mapping
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        print(f"Vocabulary size: {len(self.vocabulary)}")
        
    def text_to_vector(self, text):
        """Convert text to feature vector using bag-of-words"""
        vector = np.zeros(len(self.vocabulary))
        tokens = self.tokenize(text)
        
        for token in tokens:
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                vector[idx] += 1  # Count frequency
        
        return vector
    
    def predict(self, x):
        """Make prediction using current weights"""
        activation = np.dot(x, self.weights) + self.bias
        return 1 if activation >= 0 else 0
    
    def fit(self, texts, labels):
        """
        Train the perceptron
        texts: list of text strings
        labels: list of binary labels (0 or 1)
        """
        # Build vocabulary from training data
        self.build_vocabulary(texts)
        
        # Convert texts to vectors
        X = np.array([self.text_to_vector(text) for text in texts])
        y = np.array(labels)
        
        # Initialize weights
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for epoch in range(self.epochs):
            errors = 0
            
            for i in range(len(X)):
                # Make prediction
                prediction = self.predict(X[i])
                
                # Update weights if prediction is wrong
                if prediction != y[i]:
                    update = self.learning_rate * (y[i] - prediction)
                    self.weights += update * X[i]
                    self.bias += update
                    errors += 1
            
            if epoch % 10 == 0:
                accuracy = (len(X) - errors) / len(X) * 100
                print(f"Epoch {epoch}: Accuracy = {accuracy:.2f}%")
            
            # Early stopping if perfect accuracy
            if errors == 0:
                print(f"Converged at epoch {epoch}")
                break
    
    def evaluate(self, texts, labels):
        """Evaluate model on test data"""
        X = np.array([self.text_to_vector(text) for text in texts])
        y = np.array(labels)
        
        predictions = [self.predict(x) for x in X]
        accuracy = np.mean(predictions == y) * 100
        
        return accuracy, predictions
    
    def get_top_features(self, n=10):
        """Get top weighted features (most indicative words)"""
        if self.weights is None:
            return []
        
        word_weights = [(word, self.weights[idx]) 
                       for word, idx in self.vocabulary.items()]
        
        # Sort by absolute weight
        word_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return word_weights[:n]


def main():
    """Demonstration of the perceptron on movie review data"""
    
    # Sample training data: movie reviews with sentiment labels
    # Label: 1 = positive, 0 = negative
    train_texts = [
        "This movie was absolutely wonderful and amazing",
        "I loved every minute of this film it was fantastic",
        "Great acting and brilliant storyline highly recommend",
        "Best movie I have seen in years truly outstanding",
        "Incredible performances and beautiful cinematography",
        "This film was terrible and boring waste of time",
        "Awful movie with poor acting and bad plot",
        "I hated this film it was completely disappointing",
        "Worst movie ever made very bad experience",
        "Horrible storyline and terrible directing not recommended",
        "The movie was excellent with superb performances",
        "Outstanding film with amazing visual effects",
        "This was a masterpiece truly brilliant work",
        "Disappointing movie with weak characters and plot",
        "Boring and predictable very poor quality film"
    ]
    
    train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    
    # Test data
    test_texts = [
        "This was a great film with wonderful acting",
        "Terrible movie I would not recommend it",
        "Amazing storyline and brilliant performances",
        "Very bad film with poor quality"
    ]
    
    test_labels = [1, 0, 1, 0]
    
    print("=" * 60)
    print("PERCEPTRON FOR NLP SENTIMENT ANALYSIS")
    print("=" * 60)
    print()
    
    # Create and train perceptron
    print("Training perceptron...")
    print("-" * 60)
    perceptron = TextPerceptron(learning_rate=0.1, epochs=100)
    perceptron.fit(train_texts, train_labels)
    
    print()
    print("-" * 60)
    print("Training complete!")
    print()
    
    # Evaluate on test data
    print("Evaluating on test data...")
    print("-" * 60)
    accuracy, predictions = perceptron.evaluate(test_texts, test_labels)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print()
    
    # Show predictions
    print("Test Predictions:")
    for i, (text, pred, true) in enumerate(zip(test_texts, predictions, test_labels)):
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        correct = "✓" if pred == true else "✗"
        print(f"{correct} Review {i+1}: {sentiment}")
        print(f"  Text: '{text}'")
        print()
    
    # Show most important features
    print("-" * 60)
    print("Top 10 Most Important Words:")
    print("-" * 60)
    top_features = perceptron.get_top_features(10)
    for word, weight in top_features:
        direction = "positive" if weight > 0 else "negative"
        print(f"{word:15s}: {weight:7.4f} ({direction})")
    print()
    
    # Interactive prediction
    print("=" * 60)
    print("Try your own review:")
    print("=" * 60)
    custom_reviews = [
        "This movie was absolutely brilliant and entertaining",
        "Waste of time and money very disappointing"
    ]
    
    for review in custom_reviews:
        vector = perceptron.text_to_vector(review)
        prediction = perceptron.predict(vector)
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        print(f"Review: '{review}'")
        print(f"Prediction: {sentiment}")
        print()


if __name__ == "__main__":
    main()
