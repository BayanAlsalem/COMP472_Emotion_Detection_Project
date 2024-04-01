import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Module_4(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Module_4, self).__init__()

        # The first parameter 3 corresponds to the number of input channels,
        # not the spatial dimensions of the input images 224*224
        # For RGB color images, there are 3 channels (corresponding to red, green, and blue color channels).
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # the data is flattened before being passed to the fully connected layers
        # 128 channels and spatial dimensions of 28x28
        # it transforms the 3D feature map into a 1D vector
        self.fc1 = nn.Linear(128 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))

        #  to prevent over fitting
        # The 0.5 argument specifies the probability
        # of dropping out a neuron
        # training=self.training ensures that dropout
        # is only applied during training (not during evaluation)
        x = F.dropout(x, 0.5, training=self.training)

        x = self.fc2(x)
        return x