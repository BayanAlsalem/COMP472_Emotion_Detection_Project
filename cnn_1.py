import torch.nn as nn

class CNN_Module_1(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Module_1, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv2d creates a 2-dimensional convolutional layer for processing 2D input data (images)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(32 * 112 * 112, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv_layer(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = self.fc_layer(x)
        return x






