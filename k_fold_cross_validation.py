import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from cnn_var_2 import CNN_VAR_2


# Load the pre-trained model
def load_model(model_path, num_classes):
    model = CNN_VAR_2(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

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
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    micro_recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    return accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

# Function for k-fold cross-validation using the pre-trained model
def k_fold_cross_validation(model_path, dataset, num_classes, num_folds=10):
    paths, labels = zip(*dataset.imgs)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    results = {'macro_precision': [], 'macro_recall': [], 'macro_f1': [], 'micro_precision': [], 'micro_recall': [], 'micro_f1': [], 'accuracy': []}

    for fold, (train_index, test_index) in enumerate(kf.split(paths, labels), 1):

        test_dataset = Subset(dataset, test_index)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Load the pre-trained model for evaluation
        model = load_model(model_path, num_classes)
        
        # Evaluate the model
        accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = evaluate_model(model, test_loader)
        results['accuracy'].append(accuracy)
        results['macro_precision'].append(macro_precision)
        results['macro_recall'].append(macro_recall)
        results['macro_f1'].append(macro_f1)
        results['micro_precision'].append(micro_precision)
        results['micro_recall'].append(micro_recall)
        results['micro_f1'].append(micro_f1)

        print(f'Fold {fold} - Accuracy: {accuracy:.4f}, Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}, Micro Precision: {micro_precision:.4f}, Micro Recall: {micro_recall:.4f}, Micro F1: {micro_f1:.4f}')

    # Calculate and print the average of all metrics across all folds
    avg_results = {k: np.mean(v) for k, v in results.items()}
    print("\nAverage Evaluation Metrics across all folds:")
    for metric, avg_value in avg_results.items():
        print(f"Average {metric.replace('_', ' ').capitalize()}: {avg_value:.4f}")

# Main script
if __name__ == "__main__":
    # Define data directory
    data_dir = "dataset"

    # Define transformations
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

    # Specify path to the pre-trained model
    model_path = 'best_model.pth'

    # Perform k-fold cross-validation using the pre-trained model
    k_fold_cross_validation(model_path, dataset, num_classes=4)