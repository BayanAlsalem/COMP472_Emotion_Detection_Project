# ####################################################################################################
# ################################ Data cleaning and Data Labeling ###################################
# ####################################################################################################
import json
import os

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
from sklearn.utils import resample



# 2-Define data directory
data_dir = "dataset"


# 3-Define a transformation: transform for preprocessing the images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Flips the image horizontally with a default 50% chance
    transforms.RandomRotation(15),  # Randomly rotates the image by up to 15 degrees
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # Jitter brightness, contrast, and saturation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# 4-Load dataset using PyTorch
# ImageFolder dataset handles the mapping of class names to integer labels
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

# 5-Extract data (images) and targets (labels) from the dataset for splitting
data = [s[0] for s in dataset.samples]
targets = [s[1] for s in dataset.samples]
#----------------------------------------------------------------
#--------------------Added Biased code---------------------------
#________________________________________________________________
# Assuming bias_map is a dictionary mapping sample indices to bias attributes/values
# Load bias attributes data
with open('bias_map.json', 'r') as file:
    bias_map = json.load(file)

# You need to define your biased criteria Adjust parameters accordingly
biased_attr = "gender"
biased_val = "female"
bias_pct= .5  # Example: 20% bias towards females

#fetch the indicies from the biased map
biased_indices = [idx for idx, (sample_path, _) in enumerate(dataset.samples)
                  if os.path.basename(sample_path) in bias_map and
                  bias_map[os.path.basename(sample_path)][biased_attr] == biased_val]
total_biased_indices = len(biased_indices)
print("num of biases in totla: " + str(total_biased_indices))

# and check to see how many biased samples from the amount of bias%
num_biased_samples = int(bias_pct * total_biased_indices)
print("num of biases samples needed: " + str(num_biased_samples))


# Define the split ratios for training, validation, and test sets
train_ratio = 0.7
val_test_ratio = 0.15

# Calculate the number of biased samples needed for training according to the split ratio
num_train_samples_bias = int((len(dataset) * train_ratio) * bias_pct)
print("num of train sample bias: " + str(num_train_samples_bias))


# Convert to a float between 0 and 1
train_size_ratio = num_train_samples_bias / len(biased_indices)
print("num of biases: " + str(train_size_ratio))

if num_train_samples_bias > len(biased_indices):
    amount_need_upsample = num_train_samples_bias - (len(biased_indices))
    biased_samples_upsampled = resample(biased_indices,
                                   replace=True,  # Sample with replacement
                                   n_samples= amount_need_upsample+1,  # Match the number of unbiased samples
                                   random_state=1)  # Set random seed for reproducibility
    biased_indices = biased_samples_upsampled + biased_indices

# Split the biased indices for training, validation, and test sets
train_biased_indices, remaining_indices = train_test_split(biased_indices, train_size=train_size_ratio, random_state=1)
remaining_samples = [i for i in range(len(dataset)) if i not in biased_indices]
print("leftover unbiased images: " + str(len(remaining_samples)))

# Calculate the number of remaining samples for training after allocating biased samples
remaining_train_size = int(len(dataset) * 0.7) - num_train_samples_bias
print("remaining train size: "+ str(remaining_train_size))

# must upsample the remaining samples
# Upsample the biased samples to match the number of unbiased samples
if remaining_train_size > len(remaining_samples):
    amount_need_upsample = remaining_train_size - (len(remaining_samples))
    biased_samples_upsampled = resample(remaining_samples,
                                   replace=True,  # Sample with replacement
                                   n_samples= amount_need_upsample+1,  # Match the number of unbiased samples
                                   random_state=1)  # Set random seed for reproducibility
    final_train_indices = biased_samples_upsampled + remaining_samples
else:
    final_train_indices = remaining_samples

# Split the remaining samples for training
remaining_train_indices, remaining_temp_indices = train_test_split(
    final_train_indices, train_size=remaining_train_size, random_state=1)
# Further split the remaining indices for validation and test sets

# Calculate the number of remaining samples for training after allocating biased samples
num_remaining_train_samples = len(dataset) - len(train_biased_indices)

#combine the remaining indices with remainng_temp_indicies to get the indices for val size
test_val_indices = remaining_indices+ remaining_temp_indices

# Split the remaining samples for validation and test sets
val_size = int(num_remaining_train_samples * val_test_ratio)
val_indices, test_indices = train_test_split(test_val_indices, test_size=val_size, random_state=1)


# Combine the biased and unbiased indices for training
train_indices = train_biased_indices + remaining_train_indices

# 6-Split the dataset into training, validation, and test sets using train_test_split function from scikit-learn.
# the split ratios are 70% for training, 15% for validation, and 15% for testing.
# in the lab exercise, random_state =1 but I did some research and sometimes they use higher values
# train_indices, temp_indices = train_test_split(range(len(dataset)), test_size=0.3, random_state=1)
# val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=1)
#

# 7-Create subsets for training, validation, and testing
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)



####################################################################################################
### Dataset Visualization: Class Distribution - 25 Sample Images - Pixel Intensity Distribution ####
####################################################################################################

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
num_epochs = 50
num_classes = 4
# learning_rate = 0.0001 # For cnn_4 and cnn_var_1
learning_rate = 1e-4 #0.00001 # For cnn_var_2

# 2-Import the cnn class
from cnn_4 import CNN_Module_4
from cnn_var_1 import CNN_VAR_1
from cnn_var_2 import CNN_VAR_2

# 3-Instantiate the CNN model
# model = CNN_Module_4(num_classes)
# model = CNN_VAR_1(num_classes)
model = CNN_VAR_2(num_classes)

# 4-Define loss function
criterion = nn.CrossEntropyLoss()

# 5-Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6-Define the batch size
batch_size = 64

# 7-Create DataLoader for training and test dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
print("Train dataset length: ", len(train_dataset))
print("Test dataset length: ", len(test_dataset))

# 8-Training Loop and track the best accuracy for early stopping
total_step = len(train_loader)
loss_list = []
acc_list = []

# Initialize early stopping parameters
best_accuracy = None
best_epoch = 0
patience = 0.25 * num_epochs  # Number of epochs to wait for improvement before stopping
early_stopping_counter = 0

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
        print('Validation Accuracy: {:.4f} %'.format((accuracy) * 100))

        # Early stopping check
        if best_accuracy is None or accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best score achieved at epoch {best_epoch+1}.")
                break  # Stop training



    model.train()
    if (i + 1) % 100 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1,
                                                                                        total_step, loss.item(),
                                                                                        accuracy * 100))

# 9- Load the best model for evaluation
#best_model = CNN_Module_4(num_classes)
# best_model = CNN_VAR_1(num_classes)
best_model = CNN_VAR_2(num_classes)
best_model.load_state_dict(torch.load('best_model.pth'))
best_model.eval()

# 10-Evaluation on test set
test_predictions = []
test_labels = []
for images, labels in test_loader:
    outputs = best_model(images)
    _, predicted = torch.max(outputs.data, 1)
    test_predictions.extend(predicted.cpu().numpy())
    test_labels.extend(labels.cpu().numpy())

# 11-Compute evaluation metrics
test_accuracy = accuracy_score(test_labels, test_predictions)

# Macro averages
test_precision_macro = precision_score(test_labels, test_predictions, average='macro')
test_recall_macro = recall_score(test_labels, test_predictions, average='macro')
test_f1_score_macro = f1_score(test_labels, test_predictions, average='macro')

# Micro averages
test_precision_micro = precision_score(test_labels, test_predictions, average='micro')
test_recall_micro = recall_score(test_labels, test_predictions, average='micro')
test_f1_score_micro = f1_score(test_labels, test_predictions, average='micro')

conf_matrix = confusion_matrix(test_labels, test_predictions)

# 12-Display confusion matrix
# Retrieve class to index mapping and invert it
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

# Convert numerical indices to class names
true_class_names = [idx_to_class[idx] for idx in test_labels]
predicted_class_names = [idx_to_class[idx] for idx in test_predictions]

# Generate confusion matrix with class names instead of indices
conf_matrix = confusion_matrix(true_class_names, predicted_class_names, labels=list(idx_to_class.values()))

# Plot confusion matrix with class names using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=idx_to_class.values(), yticklabels=idx_to_class.values())
plt.xlabel('Predicted Classes')
plt.ylabel('True Classes')
plt.title('Confusion Matrix with Class Names')

# Save the figure
plt.savefig('best_model_confusion_matrix.png', bbox_inches='tight')
plt.show()  # If you want to display it as well


# 13-Metrics summary
metrics_summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)',
            'Precision (Micro)', 'Recall (Micro)', 'F1-Score (Micro)'],
    'Value': [test_accuracy, test_precision_macro, test_recall_macro, test_f1_score_macro,
            test_precision_micro, test_recall_micro, test_f1_score_micro]
})
print("Metrics Summary:")
print(metrics_summary)

