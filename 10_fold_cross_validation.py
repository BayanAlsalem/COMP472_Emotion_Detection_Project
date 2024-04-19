import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

# I Used KFold instead of StratifiedKFold
# I changed the batch size from 32 to 16


# CNN architecture
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
    macro_precision = precision_score(all_labels, all_preds, average='macro',zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro',zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro',zero_division=0)

    micro_precision = precision_score(all_labels, all_preds, average='micro',zero_division=0)
    micro_recall = recall_score(all_labels, all_preds, average='micro',zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro',zero_division=0)

    return accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

# try the normal kfold
# Function for k-fold cross-validation
def k_fold_cross_validation(model, dataset, num_classes, num_epochs=10, num_folds=10):
    paths, labels = zip(*dataset.imgs)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    micro_precision_list = []
    micro_recall_list = []
    micro_f1_list = []
    accuracy_list = []

    for fold, (train_index, test_index) in enumerate(kf.split(paths, labels), 1):
        print(f'Fold {fold}')
        train_dataset = Subset(dataset, train_index)
        test_dataset = Subset(dataset, test_index)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        model = CNN_VAR_2(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = evaluate_model(model, test_loader)
        accuracy_list.append(accuracy)
        macro_precision_list.append(macro_precision)
        macro_recall_list.append(macro_recall)
        macro_f1_list.append(macro_f1)
        micro_precision_list.append(micro_precision)
        micro_recall_list.append(micro_recall)
        micro_f1_list.append(micro_f1)

        print(f'Accuracy: {accuracy:.4f}, Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}')
        print(f'Micro Precision: {micro_precision:.4f}, Micro Recall: {micro_recall:.4f}, Micro F1: {micro_f1:.4f}')

    avg_accuracy = np.mean(accuracy_list)
    avg_macro_precision = np.mean(macro_precision_list)
    avg_macro_recall = np.mean(macro_recall_list)
    avg_macro_f1 = np.mean(macro_f1_list)
    avg_micro_precision = np.mean(micro_precision_list)
    avg_micro_recall = np.mean(micro_recall_list)
    avg_micro_f1 = np.mean(micro_f1_list)

    print("\nAverage Evaluation Metrics across 10 folds:")
    print(f"Average Macro Precision: {avg_macro_precision}, Average Macro Recall: {avg_macro_recall}, Average Macro F1-score: {avg_macro_f1}")
    print(f"Average Micro Precision: {avg_micro_precision}, Average Micro Recall: {avg_micro_recall}, Average Micro F1-score: {avg_micro_f1}")
    print(f"Average Accuracy: {avg_accuracy}")

# Main
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

    # Initialize the model
    num_classes = 4  # Number of classes in your dataset
    model = CNN_VAR_2(num_classes)

    # Perform k-fold cross-validation
    k_fold_cross_validation(model, dataset, num_classes)
