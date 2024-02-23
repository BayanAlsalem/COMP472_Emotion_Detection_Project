import os
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set the dataset path
dataset_path = r'C:\Users\User\PycharmProjects\COMP472_Emotion_Detection_Project\dataset'

# Define the classes or labels and dataset types (train/test)
classes = ['happy', 'neutral', 'surprised', 'angry']
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

# EDA: Plot class distribution for training data
plt.hist(train_labels_np, bins=np.arange(len(classes) + 1), align='left', rwidth=0.4)
plt.xticks(ticks=np.arange(len(classes)), labels=classes)
plt.title('Class Distribution in Training Data')
plt.show()

# EDA: Plot class distribution for testing data
plt.hist(test_labels_np, bins=np.arange(len(classes) + 1), align='left', rwidth=0.4)
plt.xticks(ticks=np.arange(len(classes)), labels=classes)
plt.title('Class Distribution in Testing Data')
plt.show()


print(f"Displaying sample images. Train data length: {len(train_data)}")

# Optional: Display a few sample images from the training set
for i in range(3):
    img = transforms.ToPILImage()(train_data[i])
    plt.imshow(img)
    plt.title(classes[train_labels[i]])
    plt.show()


