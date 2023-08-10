import torch
from torch import nn

def flatten(x):
  return x.view(-1)

class CNN(nn.Module):
  def __init__(self, num_classes=5):
    super(CNN, self).__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=5, padding=2),
      nn.LeakyReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.LeakyReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.LeakyReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Flatten(),
      nn.Linear(128 * 32 * 32, 1024),
      nn.LeakyReLU(),
      nn.Dropout(0.5),
      nn.Linear(1024, 512),
      nn.LeakyReLU(),
      nn.Dropout(0.5),
      nn.Linear(512, num_classes),
      nn.Softmax(dim=1)
    )
  def forward(self, x):
    x = self.layers(x)
    return x

