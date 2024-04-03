import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Import the model architecture (ensure it matches with the saved model)
from cnn_var_2 import CNN_VAR_2


# ------------ ----- ------- --- -- ------------------ #
# ------------ --- Configuration -- ------------------ #
# ------------ ----- ------- --- -- ------------------ #

batch_size = 64
num_classes = 4

# Define data directory
#data_dir = "C:/Users/User/PycharmProjects/COMP472_Emotion_Detection_Project/dataset"
data_dir = "dataset"
model_dir = "best_models/cnn_var_2/best_model.pth"
image_dir = "dataset/happy/im62.png"

# Define evaluation function

# Set to False to predit a single image from image_dir
# Set to True to evaluate all dataset

evaluateAllDataset = True # Set to True/False

# ------------ ----- ------- --- -- ------------------ #
# ----------- --- END Configuration -- --------------- #
# ------------ ----- ------- --- -- ------------------ #


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.RandomHorizontalFlip(),  # Flips the image horizontally with a default 50% chance
    transforms.RandomRotation(15),  # Randomly rotates the image by up to 15 degrees
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # Jitter brightness, contrast, and saturation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=[0.4957, 0.4886, 0.4849], std=[0.2120, 0.2121, 0.2125])  # calculate_normalization.py generated it
])

# Assuming you have 4 classes: 'focused', 'happy', 'neutral', 'sad'
# The exact mapping should match the indices assigned by datasets.ImageFolder
idx_to_class = {0: 'focused', 1: 'happy', 2: 'neutral', 3: 'sad'}

# Function to evaluate on a dataset
def evaluate_model(model, dataset_path):
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on dataset: {100 * correct / total}%')

# Function to predict an individual image
def predict_image(model, image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class_name = idx_to_class[predicted.item()]
    return predicted_class_name


# Load the model
model = CNN_VAR_2(num_classes)
model.load_state_dict(torch.load(model_dir))


if(evaluateAllDataset):
    evaluate_model(model, data_dir)
else:
    predicted_class = predict_image(model, image_dir)
    print(f'Predicted class: {predicted_class}')