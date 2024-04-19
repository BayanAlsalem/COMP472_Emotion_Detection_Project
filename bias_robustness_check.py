# 1-Import necessary libraries
import json
import os
import random
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import random
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from torchvision import transforms
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from bias_analysis import CNN_VAR_2

# Load bias attributes data
with open('bias_map.json', 'r') as file:
    bias_map = json.load(file)

def add_images_to_train(train_loader, bias_map, biased_attribute, desired_bias_value, num_images_to_add):
    """Add randomly selected images that match the desired bias to the training set."""
    dataset = train_loader.dataset
    total_images = len(dataset)
    for idx, sample in enumerate(dataset):
        tensor_hash = tuple(sample[0].numpy().tolist())
        print(tensor_hash)

    biased_indices = [idx for idx, sample in enumerate(dataset)
                      if bias_map[tuple(sample[0].numpy().tolist())][biased_attribute] == desired_bias_value]
    non_biased_indices = [idx for idx, sample in enumerate(dataset) if
                          bias_map[sample[0]][biased_attribute] != desired_bias_value]

    # Ensure that we don't add more images than available in the dataset
    num_images_to_add = min(num_images_to_add, len(non_biased_indices))

    # Randomly select images matching the desired bias
    selected_indices = random.sample(biased_indices, int(num_images_to_add))

    # Add the selected images to the training set
    train_loader.dataset = Subset(dataset, non_biased_indices + selected_indices)


import random


def calculate_num_images_in_train(train_loader):
    """Calculate the number of images in the training set."""
    return len(train_loader.dataset)



import hashlib
import torch
import io

def tensor_to_hash(tensor):
    """Convert a tensor to a hashable string."""
    stream = io.BytesIO()
    torch.save(tensor, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()

def calculate_num_images_with_desired_bias(train_loader, bias_map, biased_attribute, desired_bias_value):
    """Calculate the number of images in the training set that correspond to the desired bias."""
    num_images_with_bias = sum(1 for sample in train_loader.dataset
                               if tensor_to_hash(sample[0]) in bias_map)
    print([tensor_to_hash(sample[0]) for sample in train_loader.dataset])

    return num_images_with_bias



def check_bias_percentage_threshold(train_loader, bias_map, biased_attribute, desired_bias_value,
                                    desired_bias_percentage):
    """Check if the bias percentage threshold is met in the training set."""
    total_images = calculate_num_images_in_train(train_loader)
    num_images_with_bias = calculate_num_images_with_desired_bias(train_loader, bias_map, biased_attribute,
                                                                  desired_bias_value)
    bias_percentage = (num_images_with_bias / total_images) * 100
    return bias_percentage >= desired_bias_percentage


def adjust_train_set_for_bias(train_loader, bias_map, biased_attribute, desired_bias_value, desired_bias_percentage):
    """Adjust the training set to meet the desired bias percentage threshold."""
    while not check_bias_percentage_threshold(train_loader, bias_map, biased_attribute, desired_bias_value,
                                              desired_bias_percentage):
        num_images_to_add = calculate_num_images_in_train(train_loader) * (
                    desired_bias_percentage / 100) - calculate_num_images_with_desired_bias(train_loader, bias_map,
                                                                                            biased_attribute,
                                                                                            desired_bias_value)
        add_images_to_train(train_loader, bias_map, biased_attribute, desired_bias_value, num_images_to_add)
        remove_excess_non_biased(train_loader, bias_map, biased_attribute, desired_bias_value, desired_bias_percentage)


def remove_excess_non_biased(train_loader, bias_map, biased_attribute, desired_bias_value, desired_bias_percentage):
    """Remove excess non-biased images from the training set to maintain the desired bias percentage."""
    total_images = calculate_num_images_in_train(train_loader)
    num_images_with_bias = calculate_num_images_with_desired_bias(train_loader, bias_map, biased_attribute,
                                                                  desired_bias_value)
    num_images_to_remove = total_images - (num_images_with_bias / (desired_bias_percentage / 100))

    non_biased_indices = [idx for idx, sample in enumerate(train_loader.dataset) if
                          bias_map[str(sample[0])][biased_attribute] != desired_bias_value]
    indices_to_remove = random.sample(non_biased_indices, int(num_images_to_remove))

    train_loader.dataset = Subset(train_loader.dataset, list(set(range(total_images)) - set(indices_to_remove)))


def load_biased_training_loader(data_dir, data_transforms, bias_map, biased_attribute, desired_bias_value,
                                desired_bias_percentage):
    """Load the biased training loader with the desired bias percentage."""
    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    train_indices, _ = train_test_split(range(len(dataset)), test_size=0.3, random_state=1)
    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)

    # Adjust training set for bias
    adjust_train_set_for_bias(train_loader, bias_map, biased_attribute, desired_bias_value, desired_bias_percentage)

    return train_loader


if __name__ == "__main__":
    # Load the model
    model_path = 'best_model.pth'  # Ensure this is the correct path
    num_classes = 4  # Adjust based on your dataset classes
    model = CNN_VAR_2(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define data directory and transformations
    data_dir = "dataset"
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

    # Step 2: Create biased loaders for specific attribute-value pairs
    biased_loaders = {}
    attributes = ['age', 'gender']  # Add more attributes as needed

    # 6-Split the dataset into training, validation, and test sets using train_test_split function from scikit-learn.
    # the split ratios are 70% for training, 15% for validation, and 15% for testing.
    # in the lab exercise, random_state =1 but I did some research and sometimes they use higher values
    train_indices, temp_indices = train_test_split(range(len(dataset)), test_size=0.3, random_state=1)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=1)

    # 6-Define the batch size
    batch_size = 64

    # 7-Create subsets for training, validation, and testing
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Step 3: Combine the biased loaders with the regular loaders
    train_loaders = [DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, pin_memory=True)]
    val_loaders = [DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, pin_memory=True)]
    test_loaders = [DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False, pin_memory=True)]

biased_attribute = "gender"
desired_bias_value = "female"
bias_pct= 20  # Example: 20% bias towards females

# Load the biased training loader
train_loader = load_biased_training_loader(data_dir, data_transforms, bias_map, biased_attribute, desired_bias_value, bias_pct)

# Calculate relevant values
num_images_in_train = calculate_num_images_in_train(train_loader)
num_images_with_desired_bias = calculate_num_images_with_desired_bias(train_loader, bias_map, biased_attribute, desired_bias_value)
bias_percentage_threshold_met = check_bias_percentage_threshold(train_loader, bias_map, biased_attribute, desired_bias_value, bias_pct)

# Print the results
print("Number of images in the training set:", num_images_in_train)
print("Number of images with desired bias:", num_images_with_desired_bias)
print("Is the bias percentage threshold met?:", bias_percentage_threshold_met)

# Convert train_loader to JSON
train_loader_json = json.dumps(train_loader)
print("Train loader (JSON format):", train_loader_json)
