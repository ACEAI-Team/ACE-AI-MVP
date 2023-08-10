import util, dataset, model
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
batch_size = 64
epochs = 5

print('Initializing AI stuff...')
model = model.CNN().to(device)
train_dataset = dataset.ECGDataset('../data/simg/train.h5')
test_dataset = dataset.ECGDataset('../data/simg/test.h5')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
print('Initialized, starting training.')

for epoch in range(epochs):
  model.train()
  loop = tqdm(train_loader, total=len(train_loader), leave=True)
  for images, labels in loop:
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    loop.set_postfix(loss=loss.item())
  print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    loop = tqdm(test_loader, total=len(test_loader), leave=True)
    for images, labels in loop:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()
      loop.set_postfix(accuracy=100 * correct / total)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
print("Training complete!")
