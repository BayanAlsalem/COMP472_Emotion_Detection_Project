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



# ####################################################################################################
# ### Dataset Visualization: Class Distribution - 25 Sample Images - Pixel Intensity Distribution ####
# ####################################################################################################

# 1- Class Distribution Visualization - bar graph
class_names = dataset.classes
class_counts = {class_name:0 for class_name in class_names}

for _, index in dataset.samples:
    class_counts[class_names[index]] += 1

plt.figure(figsize=(8, 7))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of images')
plt.title('Class Distribution')
plt.show()

# 2- one pixel intensity distribution histogram for the 25 samples
def pixel_intensity_distribution(sample_images):
    # lists to accumulate pixel values for each channel
    pixels_red, pixels_green, pixels_blue = [], [], []

    for img in sample_images:
        pixels_red.extend(img[0].flatten().numpy())
        pixels_green.extend(img[1].flatten().numpy())
        pixels_blue.extend(img[2].flatten().numpy())

    # Plotting the histogram
    plt.figure(figsize=(6, 4))
    plt.hist(pixels_red, bins=256, range=(0, 1), alpha=0.5, color='r', label='R channel')
    plt.hist(pixels_green, bins=256, range=(0, 1), alpha=0.5, color='g', label='G channel')
    plt.hist(pixels_blue, bins=256, range=(0, 1), alpha=0.5, color='b', label='B channel')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Pixel Intensity Distribution')
    plt.show()

# 3-Display the 25 sample images and their intensity distribution histogram
def display_sample_images_and_intensity_distribution(dataset, selected_class_names, n_samples=25):
    for class_name in selected_class_names:
        class_indices = [i for i, sample in enumerate(dataset.samples) if dataset.classes[sample[1]] == class_name]
        if not class_indices:
            print(f"No samples found for class '{class_name}'. Skipping...")
            continue

        sample_indices = random.sample(class_indices, min(len(class_indices), n_samples))
        sample_images = [dataset[idx][0] for idx in sample_indices]

        # Display the sample images
        plt.figure(figsize=(8, 8))
        plt.suptitle(f'Sample Images for Class: {class_name}', fontsize=16)

        for i in range(1, n_samples + 1):
            if i <= len(sample_images):
                img = sample_images[i - 1]
                plt.subplot(5, 5, i)
                plt.imshow(img.permute(1, 2, 0))
                plt.axis('off')
            else:
                break  # Exit the loop if there are no more images to display

        plt.show()

        # For the same sample images, plot the combined pixel intensity distribution
        pixel_intensity_distribution(sample_images)

selected_class_names = ['focused', 'happy', 'neutral', 'sad']
display_sample_images_and_intensity_distribution(dataset, selected_class_names)
