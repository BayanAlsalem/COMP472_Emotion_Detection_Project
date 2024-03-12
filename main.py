import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Set the dataset path
dataset_path = r'C:\Users\User\PycharmProjects\COMP472_Emotion_Detection_Project\dataset'

# Define the classes or labels and dataset types (train/test)
classes = ['happy', 'neutral', 'surprised', 'focused']
dataset_types = ['train', 'test']

# This is a function definition to load and process the data.
# It takes the base path of the dataset, the type of dataset
# (train or test), and the transform operations to apply to the images.
def load_and_process_data(base_path, dataset_type, transform):
    # Initializes empty lists to hold the processed images and their corresponding labels.
    data = []
    labels = []

    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(base_path, dataset_type, class_name)
        # Loops over each image in the class directory.
        for img_name in os.listdir(class_path):
            try:
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert('RGB')  # Ensure image is RGB
                img = transform(img)  # Apply preprocessing
                # Appends the processed image to the data
                # list and its label to the labels list.
                data.append(img)
                labels.append(class_index)

            except IOError:
                print(f'Error loading image: {img_path}')

    return data, labels


# Define a transform for preprocessing the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a common size
    transforms.ToTensor(),  # Convert to tensor
])

# Load and process the training and testing data
train_data, train_labels = load_and_process_data(dataset_path, 'train', transform)
test_data, test_labels = load_and_process_data(dataset_path, 'test', transform)

# Convert lists to numpy arrays for EDA
train_labels_np = np.array(train_labels)
test_labels_np = np.array(test_labels)

# # EDA: Plot class distribution for training data
# plt.hist(train_labels_np, bins=np.arange(len(classes) + 1), align='left', rwidth=0.4)
# plt.xticks(ticks=np.arange(len(classes)), labels=classes)
# plt.title('Class Distribution in Training Data')
# plt.show()
#
# # EDA: Plot class distribution for testing data
# plt.hist(test_labels_np, bins=np.arange(len(classes) + 1), align='left', rwidth=0.4)
# plt.xticks(ticks=np.arange(len(classes)), labels=classes)
# plt.title('Class Distribution in Testing Data')
# plt.show()

######################################################
###### STEP 1: SHOW ALL COUNT OF ALL CLASSES ########
######################################################

# Count images in each class for both training and testing data
train_counts = [sum(train_labels_np == i) for i in range(len(classes))]
test_counts = [sum(test_labels_np == i) for i in range(len(classes))]

# # Defining the width of the bars (using the same width as before for consistency)
# width = 0.35
#
# # Plotting the bar graph
# plt.figure(figsize=(12, 8))
# bars1 = plt.bar(classes, train_counts, width, label='Training Data', color='skyblue')
# bars2 = plt.bar(classes, test_counts, width, bottom=train_counts, label='Test Data', color='orange')
#
# plt.xlabel('Class Names')
# plt.ylabel('Number of Images')
# plt.title('Class Distribution in Dataset: Combined Training and Test Data')
# plt.legend()
# plt.show()



# Calculating positions for each set of bars
x = np.arange(len(classes))  # the label locations
width = 0.35  # the width of the bars

# Plotting the bar graph for both training and test data
plt.figure(figsize=(12, 8))
bars1 = plt.bar(x - width/2, train_counts, width, label='Training Data', color='skyblue')
bars2 = plt.bar(x + width/2, test_counts, width, label='Test Data', color='orange')

# Adding some text for labels, title and custom x-axis tick labels, etc.
plt.xlabel('Class Names')
plt.ylabel('Number of Images')
plt.title('Class Distribution in Dataset: Training vs Test Data')
plt.xticks(ticks=x, labels=classes)
plt.legend()

plt.show()


######################################################
# STEP 2: SHOW SAMPLE 5X5 TRAIN DATA OF ALL CLASSES #
######################################################

# Number of images and grid size
num_images = 25
grid_size = (5, 5)

for i in range(len(classes)):
    # Get the indices of images belonging to the current class in the training data
    class_indices = np.where(train_labels_np == i)[0]
    # Randomly select 25 indices
    selected_indices = np.random.choice(class_indices, num_images, replace=False)

    # Plotting the images in a 5x5 grid
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    fig.suptitle('25 Random Sample Images for `' + classes[i] + '` class')

    # Adjust layout
    plt.subplots_adjust(wspace=0, hspace=0)

    for ax, idx in zip(axes.flatten(), selected_indices):
        # Load the image from the training data
        img = train_data[idx].permute(1, 2, 0)  # Convert tensor back to PIL image format

        # Display the image
        ax.imshow(img)
        ax.axis('off')  # Hide axes

    plt.show()

    ######################################################
    # STEP 3: Pixel Intensity #
    ######################################################

    red_pixels = []
    green_pixels = []
    blue_pixels = []
    for ax, idx in zip(axes.flatten(), selected_indices):
        # Convert tensor to PIL format by permuting and then converting to numpy
        img_np = train_data[idx].permute(1, 2, 0).numpy()

        # Extracting RGB channels
        red_pixels.extend(img_np[:, :, 0].flatten())
        green_pixels.extend(img_np[:, :, 1].flatten())
        blue_pixels.extend(img_np[:, :, 2].flatten())

    # Plotting histograms
    plt.figure(figsize=(10, 6))
    plt.hist(red_pixels, bins=256, color='red', alpha=0.5, label='Red Channel')
    plt.hist(green_pixels, bins=256, color='green', alpha=0.5, label='Green Channel')
    plt.hist(blue_pixels, bins=256, color='blue', alpha=0.5, label='Blue Channel')
    plt.title('Pixel Intensity Distribution for RGB Channels')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()