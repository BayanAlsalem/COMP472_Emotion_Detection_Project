import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_VAR_3(nn.Module):
    def __init__(self, num_classes):
        super(CNN_VAR_3, self).__init__()

        # Define the convolutional layers with corresponding batch norms
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(32)

        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Define the pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass through each layer
        x = self.pool(F.relu(self.bn0(self.conv0(x))))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Reshape the output for the fully connected layer
        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x
