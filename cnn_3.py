import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Module_3(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_Module_3, self).__init__()
        # in_channels=3 (for RGB images),
        # out_channels=32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        # the size to 56x56 (224 -> 112 -> 56), and with 64 output channels from the
        # last conv layer, the total features before the fully connected layer are
        # 56 * 56 * 64.
        self.flattened_size = 56 * 56 * 64

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Apply the first convolutional layer, followed by batch normalization,
        # ReLU activation, and the second convolutional layer
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        # Flatten the output for the fully connected layer
        x = x.view(-1, self.flattened_size)

        # Apply the fully connected layers
        x = self.drop1(self.relu3(self.fc1(x)))
        x = self.fc2(x)

        return x
