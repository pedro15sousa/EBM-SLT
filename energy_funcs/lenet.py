import numpy as np

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layers(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif type(m) == nn.Linear:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(0, 0.01)
    
def sq_activation(x):
    return 0.5 * torch.pow(x,2)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class LeNet(nn.Module):
    """ Adapted LeNet
    - Swish activ. func.
    - padding=2 in first convo layer (instead of 0)
    """
    def __init__(self, out_dim=1, **kwargs):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2), #(28x28)
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), #(14x14)
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0), #(10x10)
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), #(5x5)
            nn.Flatten(),
            nn.Linear(5*5*16, 64),
            nn.SiLU(),
            nn.Linear(64, out_dim)
        )
        self.cnn_layers.apply(init_layers)

    def forward(self, x):
        o = self.cnn_layers(x).squeeze(dim=-1)
        return sq_activation(o)