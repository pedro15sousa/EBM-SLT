import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # First VGG block
        # First VGG block
        self.vgg_1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Second VGG block
        self.vgg_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Fully connected layer
        self.fc = nn.Linear(256 * 7 * 7, 10)

    def forward(self, x):
        x = self.vgg_1(x)
        x = self.vgg_2(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def get_activations(self, x):
        # Forward pass through the first part of the network
        x = self.vgg_1(x)
        x = self.vgg_2(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        # Stop here and return these activations
        return x