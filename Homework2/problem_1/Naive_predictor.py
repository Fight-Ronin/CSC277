import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from datasets import load_dataset

# Step 1: Determine the majority class
# Load the dataset (assuming it's the same dataset used for fine-tuning)
dataset = load_dataset("tweet_eval", name="sentiment")

# Get the labels from the training set
train_labels = dataset['train']['label']

# Find the most frequent (majority) class
majority_class = Counter(train_labels).most_common(1)[0][0]
print(f"The majority class is: {majority_class}")

# Step 2: Implement the Naive Predictor
# Get true labels from the test set (or validation set if there's no test set)
test_labels = dataset['test']['label']  # Replace 'test' with 'validation' if necessary

# Predict the majority class for all examples
naive_predictions = np.full_like(test_labels, majority_class)

# Step 3: Evaluate the performance of the naive predictor
accuracy = accuracy_score(test_labels, naive_predictions)
f1 = f1_score(test_labels, naive_predictions, average="weighted")

print(f"Naive Predictor Accuracy: {accuracy}")
print(f"Naive Predictor F1 Score: {f1}")
