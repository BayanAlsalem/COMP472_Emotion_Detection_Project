import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

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




# Function for k-fold cross-validation
def k_fold_cross_validation(model, dataset, num_classes, num_epochs=10, num_folds=10):
    paths, labels = zip(*dataset.imgs)
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)



    for fold, (train_index, test_index) in enumerate(kf.split(paths, labels), 1):
        print(f'Fold {fold}')
        train_dataset = Subset(dataset, train_index)
        test_dataset = Subset(dataset, test_index)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
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


