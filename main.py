# ####################################################################################################
# ################################ Data cleaning and Data Labeling ###################################
# ####################################################################################################

# 1-Import necessary libraries
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



# 2-Define data directory
data_dir = "C:/Users/User/PycharmProjects/COMP472_Emotion_Detection_Project/dataset"

# 3-Define a transformation: transform for preprocessing the images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Convert to tensor
    transforms.ConvertImageDtype(torch.float32),  # image is float32
    transforms.ColorJitter(brightness=0.5),  # Adjust brightness
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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



# ####################################################################################################
# ########################### CNN model Hyperparameters and training loop ############################
# ####################################################################################################

# 1- hyper-parameters definition
num_epochs = 10
num_classes = 4
learning_rate = 0.0001

# 2-Import the cnn class
from cnn_4 import CNN_Module_4

# 3-Instantiate the CNN model
model = CNN_Module_4(num_classes)

# 4-Define loss function
criterion = nn.CrossEntropyLoss()

# 5-Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6-Define the batch size
batch_size = 32

# 7-Create DataLoader for training and test dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
print("Train dataset length: ", len(train_dataset))
print("Test dataset length: ", len(test_dataset))

# 8-Training Loop
total_step = len(train_loader)
loss_list = []
acc_list = []

# Track the best accuracy for early stopping
best_accuracy = 0


print("Starting training loop...")
for epoch in range(num_epochs):
    print(f"Inside epoch loop, epoch: {epoch}")

    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        print(f"Inside loop, Batch: {i}")
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Train accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

    print("The average running loss per batch over the entire training dataset.", running_loss / len(train_loader))

    # Model evaluation on validation set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print('Validation Accuracy: {:.2f} %'.format((accuracy) * 100))

        # Check for early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save the model
            torch.save(model.state_dict(), 'best_model.pth')


    model.train()
    if (i + 1) % 100 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1,
                                                                                        total_step, loss.item(),
                                                                                        accuracy * 100))

# 9- Load the best model for evaluation
best_model = CNN_Module_4(num_classes)
best_model.load_state_dict(torch.load('best_model.pth'))
best_model.eval()
