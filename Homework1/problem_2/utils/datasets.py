import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.model_selection import train_test_split

from collections import defaultdict
from PIL import Image
import imagehash

# Define data augmentation transformations for training and validation
transform_no_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


# Download CIFAR-10; split into train/val/test with fixed seed
def get_dataset():
    # Load CIFAR-10 dataset with data augmentation for training, no augmentation for validation
    train_data = CIFAR10(root='data/cifar10', train=True, download=True, transform=transform_no_aug)
    test_data = CIFAR10(root='data/cifar10', train=False, download=True, transform=transform_no_aug)

    val_indices, test_indices = train_test_split(
        list(range(len(test_data))), test_size=0.5, random_state=42)

    val_dataset = torch.utils.data.Subset(test_data, val_indices)
    test_dataset = torch.utils.data.Subset(test_data, test_indices)
    train_dataset = torch.utils.data.ConcatDataset([train_data, test_dataset])

    return train_dataset, val_dataset, test_dataset

# Function to compute image hashes for a dataset
def compute_hashes(dataset):
    hashes = defaultdict(list)
    for idx in range(len(dataset)):
        image, _ = dataset[idx]
        # Convert tensor to PIL image for hashing
        image_pil = transforms.ToPILImage()(image)
        hash_value = imagehash.phash(image_pil)  # can use phash, ahash, dhash, or whash
        hashes[hash_value].append(idx)
    return hashes

# Function to find overlaps between datasets
def find_overlap(hashes1, hashes2):
    overlap = set(hashes1.keys()) & set(hashes2.keys())
    if overlap:
        print(f"Found {len(overlap)} overlapping images.")
        for h in overlap:
            print(f"Hash: {h} - Train indices: {hashes1[h]}, Validation/Test indices: {hashes2[h]}")
    else:
        print("No overlapping images found.")

# Potential duplication remove function
def remove_duplicates(train_dataset, val_dataset, test_dataset):
    # Compute hashes for each dataset
    train_hashes = compute_hashes(train_dataset)
    val_hashes = compute_hashes(val_dataset)
    test_hashes = compute_hashes(test_dataset)

    # Find duplicate hashes between train, validation, and test sets
    duplicates = {
        "train_val": set(train_hashes.keys()) & set(val_hashes.keys()),
        "train_test": set(train_hashes.keys()) & set(test_hashes.keys()),
        "val_test": set(val_hashes.keys()) & set(test_hashes.keys())
    }

    # Function to filter out duplicates from dataset
    def filter_dataset(dataset, dataset_hashes, remove_hashes):
        new_indices = []
        for h, indices in dataset_hashes.items():
            if h not in remove_hashes:
                new_indices.extend(indices)
        # Return a new subset dataset with only unique indices
        return torch.utils.data.Subset(dataset, new_indices)

    # Remove duplicates between train and validation
    train_dataset_cleaned = filter_dataset(train_dataset, train_hashes, duplicates['train_val'] | duplicates['train_test'])
    val_dataset_cleaned = filter_dataset(val_dataset, val_hashes, duplicates['train_val'] | duplicates['val_test'])
    test_dataset_cleaned = filter_dataset(test_dataset, test_hashes, duplicates['train_test'] | duplicates['val_test'])

    print(f"Duplicates removed: Train set: {len(train_dataset) - len(train_dataset_cleaned)} images, "
          f"Validation set: {len(val_dataset) - len(val_dataset_cleaned)} images, "
          f"Test set: {len(test_dataset) - len(test_dataset_cleaned)} images.")

    return train_dataset_cleaned, val_dataset_cleaned, test_dataset_cleaned


def get_testset():
    test_data = CIFAR10(root='data/cifar10', train=False, download=True, transform=transform_no_aug)
    test_indices, _ = train_test_split(
        list(range(len(test_data))), test_size=0.5, random_state=42)
    test_dataset = torch.utils.data.Subset(test_data, test_indices)
    return test_dataset
