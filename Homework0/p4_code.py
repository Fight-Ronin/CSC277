import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset from txt files
def load_data(train_file, test_file):
    
    train_data = np.loadtxt(train_file, delimiter=' ')
    test_data = np.loadtxt(test_file, delimiter=' ')
    
    # Split into features and labels
    X_train = train_data[:, 1:]  
    y_train = train_data[:, 0].astype(int) - 1  # Convert labels to 0, 1, 2 to avoid indexing error
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0].astype(int) - 1  # Convert labels to 0, 1, 2 to avoid indexing error
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long), \
           torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

# Define the first model: Linear Classifier
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(2, 3)  # 2 input features, 3 output classes
    
    def forward(self, x):
        return self.linear(x)

# Define the second model: Nonlinear Neural Network with one hidden layer
class NonlinearModel(nn.Module):
    def __init__(self):
        super(NonlinearModel, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # 2 input features, 5 hidden units
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)  # 5 hidden units, 3 output classes
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, criterion, optimizer, X_train, y_train, epochs=1000):
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# Plot decision boundaries with class labels
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Z = model(grid).argmax(dim=1).numpy().reshape(xx.shape)
    
    # Plot the decision boundary with class regions
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    
    # Plot also the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu)
    
    # Create a legend with class labels
    legend1 = plt.legend(*scatter.legend_elements(),
                         loc="upper right", title="Class")
    plt.gca().add_artist(legend1)
    
    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data from text files
    X_train, y_train, X_test, y_test = load_data('iris-train.txt', 'iris-test.txt')
    
    # Initialize models, criterion, and optimizer
    model1 = LinearModel()
    model2 = NonlinearModel()
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer1 = optim.AdamW(model1.parameters(), lr=0.01)
    optimizer2 = optim.AdamW(model2.parameters(), lr=0.01)
    
    # Train both models
    losses1 = train_model(model1, criterion, optimizer1, X_train, y_train)
    losses2 = train_model(model2, criterion, optimizer2, X_train, y_train)
    
    # Plot training loss curves
    plt.plot(losses1, label='Linear Model')
    plt.plot(losses2, label='Nonlinear Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.show()
    
    # Evaluate accuracy
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        train_preds1 = model1(X_train).argmax(dim=1)
        test_preds1 = model1(X_test).argmax(dim=1)
        train_preds2 = model2(X_train).argmax(dim=1)
        test_preds2 = model2(X_test).argmax(dim=1)
    
    train_acc1 = accuracy_score(y_train, train_preds1)
    test_acc1 = accuracy_score(y_test, test_preds1)
    train_acc2 = accuracy_score(y_train, train_preds2)
    test_acc2 = accuracy_score(y_test, test_preds2)
    
    # Print accuracy in a LaTeX table
    print(f"\\begin{{table}}[h!]")
    print(f"\\centering")
    print(f"\\begin{{tabular}}{{|c|c|c|}}")
    print(f"\\hline")
    print(f" & Train Accuracy & Test Accuracy \\\\ \\hline")
    print(f"Linear Model & {train_acc1:.2f} & {test_acc1:.2f} \\\\ \\hline")
    print(f"Nonlinear Model & {train_acc2:.2f} & {test_acc2:.2f} \\\\ \\hline")
    print(f"\\end{{tabular}}")
    print(f"\\caption{{Training and Testing Accuracy of Models}}")
    print(f"\\end{{table}}")
    
    # Plot decision boundaries
    plot_decision_boundary(model1, X_train.numpy(), y_train.numpy(), 'Linear Model Decision Boundary')
    plot_decision_boundary(model2, X_train.numpy(), y_train.numpy(), 'Nonlinear Model Decision Boundary')
