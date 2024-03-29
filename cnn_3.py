import torch.nn as nn

class CNN_Module_3(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_Module_3, self).__init__()
        # in_channels=3 (for RGB images),
        # out_channels=32
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.flattened_size = 55 * 55 * 64

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

        x = x.view(-1, self.flattened_size)
        x = self.drop1(self.relu3(self.fc1(x)))
        x = self.fc2(x)

        return x
