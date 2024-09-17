import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as Fv
from torch_lr_finder import LRFinder
from torch.optim import Adam, AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from utils.models import get_model
from utils.trained_models import get_trained_model
from utils.datasets import get_testset
from torchvision import transforms
import matplotlib.pyplot as plt

# functions used for pre-train testing
# Data Leakage Check
## use the returned datasets from get_dataset() to check for data leakage, instead of using the source code
from utils.datasets import get_dataset, find_overlap, compute_hashes
train_dataset, val_dataset, test_dataset = get_dataset()

'''
# Compute hashes for train, validation, and test datasets
train_hashes = compute_hashes(train_dataset)
val_hashes = compute_hashes(val_dataset)
test_hashes = compute_hashes(test_dataset)

# Check for data leakage
print("Checking overlap between train and validation sets:")
find_overlap(train_hashes, val_hashes)

print("Checking overlap between train and test sets:")
find_overlap(train_hashes, test_hashes)

print("Checking overlap between validation and test sets:")
find_overlap(val_hashes, test_hashes)

# Model Architecture Check
model = get_model()

# Initialize the model with the correct number of classes
num_classes = 10  # For CIFAR-10
model = get_model(num_classes)

# Create a random input tensor with the shape of CIFAR-10 images
batch_size = 4
input_tensor = torch.randn(batch_size, 3, 32, 32)

# Pass the input tensor through the model
output = model(input_tensor)

# Print the output shape
print(f"Model output shape: {output.shape}")

# Verify if the output shape matches the expected shape
expected_output_shape = (batch_size, num_classes)
assert output.shape == expected_output_shape, f"Output shape mismatch: Expected {expected_output_shape}, got {output.shape}"

print("Model architecture check passed.")
'''
'''
# Gradient Descent Validation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=10).to(device)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

optimizer = AdamW(model.parameters(), lr=1e-6)
criterion = CrossEntropyLoss()

# Fetch a single batch of data
data_iter = iter(train_loader)
inputs, labels = next(data_iter)

inputs, labels = inputs.to(device), labels.to(device)

# Record initial model parameters
initial_params = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        initial_params[name] = param.clone().detach()

# Perform a single training step
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

# Compare updated parameters to initial parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        # Check if the parameter has been updated
        if torch.equal(param, initial_params[name]):
            print(f"Parameter '{name}' was NOT updated.")
        else:
            print(f"Parameter '{name}' was successfully updated.")
'''
'''
# Learning Rate Check:
# These steps provide necessary components for learning rate range test for torch_lr_finder.LRFinder
optimizer = AdamW(model.parameters(), lr=1e-6) # the lr is set to 1e-6 as specified here
criterion = CrossEntropyLoss()
train_loader = DataLoader(train_dataset, batch_size=32)

# Load your training data
train_loader = DataLoader(train_dataset, batch_size=32)

# Initialize the learning rate finder
lr_finder = LRFinder(model, optimizer, criterion, device="cuda" if torch.cuda.is_available() else "cpu")

# Run the learning rate finder
lr_finder.range_test(train_loader, end_lr=10, num_iter=100)

# Plot the results to visualize the optimal learning rate
lr_finder.plot()  # this will plot the loss vs. learning rate

# Reset the model and optimizer to the initial state
lr_finder.reset()
'''
'''
# functions used for post-train testing:
# Dying ReLU Examination
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model = get_trained_model().to(device)
test_loader = get_testset()

activations = {}

def relu_hook(module, input, output):
    print(f"Hook triggered for module: {module}")
    output = output.detach().cpu()
    total_neurons = output.numel()
    dead_neurons = (output == 0).sum().item()
    activations[module] = dead_neurons / total_neurons

# Register hooks for all ReLU layers
for name, layer in trained_model.named_modules():
    if isinstance(layer, nn.ReLU):
        print(f"Registering hook for layer: {name}")
        layer.register_forward_hook(relu_hook)

# Pass a batch of data
trained_model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        if inputs.dim() == 3:  # If 3D, add batch dimension
            inputs = inputs.unsqueeze(0)

        outputs = trained_model(inputs)
        break

# Display
if activations:
    for layer, percentage in activations.items():
        print(f"Layer: {layer}, Dying ReLU Percentage: {percentage:.3%}")
else:
    print("No hooks were triggered or no ReLU layers were found.")
'''

# Model Robustness Test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model = get_trained_model().to(device)
test_dataset = get_testset()

def eval_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            input, labels = inputs.to(device), labels.to(device)
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

'''
# Brightness Test
lambda_brightness = [0.2, 0.4, 0.6, 0.8, 1.0]
bright_accuracies = []

for brightness in lambda_brightness:
    transform_brightness = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Lambda(lambda image: image * brightness)
    ])

    test_loader_brightness = torch.utils.data.DataLoader(
        [(transform_brightness(img), label) for img, label in test_dataset],
        batch_size=32, shuffle = False
    )

    accuracy = eval_model(trained_model, test_loader_brightness, device)
    bright_accuracies.append(accuracy)
    print(f"Brightness {brightness}: Accuracy = {accuracy:.3%}")

# Plot results
plt.plot(lambda_brightness, bright_accuracies, marker='o')
plt.xlabel('Brightness Factor (λ)')
plt.ylabel('Model Accuracy')
plt.title('Model Accuracy vs. Brightness Levels')
plt.grid(True)
plt.show()
'''

'''
# Rotation test
from torch.utils.data import Dataset
class RotatedDataset(Dataset):
    def __init__(self, dataset, angle):
        self.dataset = dataset
        self.angle = angle

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = Fv.rotate(img, self.angle)  # Directly use functional rotate
        return img, label

rot_angles = [0, 60, 120, 180, 240, 300]
rot_accuracies = []

# Evaluate the model with rotated datasets
for angle in rot_angles:
    rotated_dataset = RotatedDataset(test_dataset, angle=angle)

    # Use DataLoader to load data in batches
    test_loader_rotate = DataLoader(rotated_dataset, batch_size=32, shuffle=False)

    accuracy = eval_model(trained_model, test_loader_rotate, device)
    rot_accuracies.append(accuracy)
    print(f"Rotation angle {angle}: Accuracy = {accuracy:.3%}")


# Plot results
plt.plot(rot_angles, rot_accuracies, marker='o')
plt.xlabel('Rotation Degree')
plt.ylabel('Model Accuracy')
plt.title('Model Accuracy vs. Rotation Degree')
plt.grid(True)
plt.show()
'''

# Normalization Mismatch
import numpy as np
import torchvision

# Calculate the mean and std for CIFAR-10 dataset
train_data = torchvision.datasets.CIFAR10(root='data/cifar10', train=True, download=True)

# use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
# calculate the mean and std along the (0, 1) axes
train_mean = np.mean(x, axis=(0, 1)) / 255
train_std = np.std(x, axis=(0, 1)) / 255
# the the mean and std
print(f'Training Mean: {train_mean}')
print(f'Training Std: {train_std}')


# Calculate the mean and std for test set
test_dataset = get_testset()
rgb_values_test = []

# Iterate over the test dataset to get all pixel values
for img, _ in test_dataset:
    img_pil = transforms.ToPILImage()(img)
    # Get all pixel data and append to list
    rgb_values_test.append(np.array(img_pil).reshape(-1, 3))

rgb_values_test = np.concatenate(rgb_values_test, axis=0) / 255.0

# Calculate the mean and standard deviation across all pixels
test_mean = np.mean(rgb_values_test, axis=0)
test_std = np.std(rgb_values_test, axis=0)

print(f"Testing Mean: {test_mean}")
print(f"Testing Std: {test_std}")

# Difference calculation:
Diff_mean = (test_mean - train_mean) 
Diff_std = (test_std - train_std)
per_diff_mean = (Diff_mean / train_mean) * 100
per_diff_std = (Diff_std / train_std) * 100
print( f"Differences of mean in %: {per_diff_mean}")
print( f"Differences of std in %: {per_diff_std}")

