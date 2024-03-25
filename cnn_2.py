import torch.nn as nn

class CNN_Module_2(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Module_2, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, padding=0),  # First Conv Layer
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=0),  # Second Conv Layer
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=0),  # Third Conv Layer
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Note:when the kernel size and padding are changed, the output feature map size will also change.
        # the input images are 224x224:
        # - After first Conv2d: (224-2+0)/1 + 1 = 223x223
        # - After second Conv2d: (223-2+0)/1 + 1 = 222x222
        # - After first MaxPool2d: 222x222 / 2 = 111x111
        # - After third Conv2d: (111-2+0)/1 + 1 = 110x110
        # - After second MaxPool2d: 110x110 / 2 = 55x55
        # The total number of features going into the first linear layer is 128 * 55 * 55.

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(128 * 55 * 55, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)  # Pass input through conv_layer
        x = x.view(x.size(0), -1)  # Flatten the output for the fc_layer
        x = self.fc_layer(x)  # Pass through fc_layer
        return x
