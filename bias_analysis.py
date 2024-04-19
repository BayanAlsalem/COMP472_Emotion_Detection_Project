import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define your CNN architecture
class CNN_VAR_2(nn.Module):
    def __init__(self, num_classes):
        super(CNN_VAR_2, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(6272, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn0(self.conv0(x))))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 6272)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x

# Function to evaluate model performance
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return accuracy, precision, recall, f1

# Load bias attributes data
with open('bias_map.json', 'r') as file:
    bias_map = json.load(file)

# Function to filter dataset indices by attribute safely
def filter_indices_by_attribute(dataset, attribute, value):
    filtered_indices = []
    for idx, (path, _) in enumerate(dataset.imgs):
        filename = path.split('/')[-1]
        if filename in bias_map and bias_map[filename].get(attribute) == value:
            filtered_indices.append(idx)
    return filtered_indices

# Main script
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

    # Initialize statistics
    demographic_attributes = ['age', 'gender']
    results = {}
    overall_stats = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    # Evaluate for biases across specified groups
    for attribute in demographic_attributes:
        attribute_stats = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        values = set(val[attribute] for val in bias_map.values() if attribute in val)
        for value in values:
            indices = filter_indices_by_attribute(dataset, attribute, value)
            if indices:
                demo_loader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=False)
                accuracy, precision, recall, f1 = evaluate_model(model, demo_loader)
                results[f'{attribute}_{value}'] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
                attribute_stats['accuracy'].append(accuracy)
                attribute_stats['precision'].append(precision)
                attribute_stats['recall'].append(recall)
                attribute_stats['f1'].append(f1)
                print(f'{attribute.capitalize()} {value.capitalize()} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
        
        # Print average statistics for each attribute
        print(f"\nAverage for {attribute.capitalize()}:")
        print(f'Accuracy: {np.mean(attribute_stats["accuracy"]):.4f}')
        print(f'Precision: {np.mean(attribute_stats["precision"]):.4f}')
        print(f'Recall: {np.mean(attribute_stats["recall"]):.4f}')
        print(f'F1-score: {np.mean(attribute_stats["f1"]):.4f}\n')

        # Aggregate overall stats
        overall_stats['accuracy'].extend(attribute_stats['accuracy'])
        overall_stats['precision'].extend(attribute_stats['precision'])
        overall_stats['recall'].extend(attribute_stats['recall'])
        overall_stats['f1'].extend(attribute_stats['f1'])

    # Print overall average statistics
    print("Overall System Average:")
    print(f'Accuracy: {np.mean(overall_stats["accuracy"]):.4f}')
    print(f'Precision: {np.mean(overall_stats["precision"]):.4f}')
    print(f'Recall: {np.mean(overall_stats["recall"]):.4f}')
    print(f'F1-score: {np.mean(overall_stats["f1"]):.4f}')
