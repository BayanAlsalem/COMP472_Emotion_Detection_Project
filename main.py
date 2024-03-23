# ####################################################################################################
# ################################ Data cleaning and Data Labeling ###################################
# ####################################################################################################

# 1-Import necessary libraries
import os
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import random
from torch.utils.data import Subset
import matplotlib.pyplot as plt


# 2-Define data directory
data_dir = "C:/Users/User/PycharmProjects/COMP472_Emotion_Detection_Project/dataset"

# 3-Define a transformation: transform for preprocessing the images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a common size
    transforms.ToTensor(),  # Convert to tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Normalize: scaling the pixel values of an image to have a mean of 0 and a standard deviation of 1.
])

# 4-Load dataset using PyTorch
# ImageFolder dataset handles the mapping of class names to integer labels
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

# 5-Extract data (images) and targets (labels) from the dataset for splitting
data = [s[0] for s in dataset.samples]
targets = [s[1] for s in dataset.samples]

# 6-Split the dataset into training, validation, and test sets using train_test_split function from scikit-learn.
# the split ratios are 70% for training, 15% for validation, and 15% for testing.
# in the lab exercise, random_state =1 but I did some research and sometimes they use higher values
train_indices, temp_indices = train_test_split(range(len(dataset)), test_size=0.3, random_state=1)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=1)


# 7-Create subsets for training, validation, and testing
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)



