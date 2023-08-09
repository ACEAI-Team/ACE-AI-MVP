import torch
from torch import nn

def flatten(x):
  return x.view(x.size(0), -1)

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.layers = [
          nn.Conv2d(1, 32, kernel_size=5, padding=2, in_channels=1),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size=3, padding=1, in_channels=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(64, 128, kernel_size=3, padding=1, in_channels=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Linear(64 * 49 * 49, 512),
          nn.ReLU(),
          self.dropout,
          nn.Linear(512, num_classes)
        ]
    def forward(self, x):
      for layer in self.layers:
        x = layer(x)
      return x

