import util, dataset, model
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


config_dir = input('Config dir: ')
if config_dir:
  with open(config_dir, 'r') as f:
    config = eval(f.read())
else:
  with open('config', 'r') as f:
    config = eval(f.read())

if torch.cuda.is_available():
  if config['gpu']:
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
else:
  device = torch.device('cpu')
learning_rate = config['learning rate']
batch_size = config['batch size']
epochs = config['epochs']

cnn = model.CNN().to(device)
model_dir = config['model']
if model_dir:
  cnn.load_state_dict(torch.load(model_dir))
half = config['half']
if half:
  cnn = cnn.half()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1)
mse = torch.nn.MSELoss()
num_workers = config['num_workers']
print(f'''SETUP:
  Device: {device}
  Learning rate: {learning_rate}
  Batch Size: {batch_size}
  Epochs: {epochs}
  Model: {model_dir if model_dir else "no model"}
  Half precision: {"enabled" if half else "disabled"}
  Training dataset: {config["training data"]}
  Test dataset: {config["test data"]}
  num_workers: {num_workers}
  Optimizer: {optimizer}
  Criterion: {criterion}
Loading datasets...''')
train_dataset = dataset.ECGDataset(config['training data'])
test_dataset = dataset.ECGDataset(config['test data'])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

cnn.eval()
with torch.no_grad():
  correct = 0
  total = 0
  error = 0
  loop = tqdm(test_loader, total=len(test_loader), leave=True)
  for images, labels in loop:
    if half:
      images = images.half()
    else:
      images = images.float()
    images = images.to(device)
    labels = labels.long().to(device)
    targets = torch.nn.functional.one_hot(labels.long(), 5).to(device)
    outputs = cnn(images)
    _, predicted = outputs.max(1)
    total += labels.size(0)
    error += mse(outputs, targets)
    correct += predicted.eq(labels).sum().item()
    loop.set_postfix(accuracy=f'{100 * correct / total:.2f}%')
  print(f'Test Accuracy: {100 * correct / total}%\nError: {error / total}')
for epoch in range(epochs):
  cnn.train()
  loop = tqdm(train_loader, total=len(train_loader), leave=True)
  for images, labels in loop:
    if half:
      images = images.half()
    else:
      images = images.float()
    images = images.to(device)
    labels = labels.long().to(device)
    optimizer.zero_grad()
    outputs = cnn(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    #print(cnn.layers[0].weight)
    loop.set_postfix(loss=loss.item())
  print(f'Epoch [{epoch+1}/{epochs}]')
  torch.save(cnn.state_dict(), 'model.ckpt')
  cnn.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    error = 0
    loop = tqdm(test_loader, total=len(test_loader), leave=True)
    for images, labels in loop:
      if half:
        images = images.half()
      else:
        images = images.float()
      images = images.to(device)
      labels = labels.long().to(device)
      targets = torch.nn.functional.one_hot(labels.long(), 5).to(device)
      outputs = cnn(images)
      _, predicted = outputs.max(1)
      total += labels.size(0)
      error += mse(outputs, targets)
      correct += predicted.eq(labels).sum().item()
      loop.set_postfix(accuracy=f'{100 * correct / total:.2f}%')
    print(f'Test Accuracy: {100 * correct / total}%\nError: {error / total}')
  scheduler.step(correct / total)
print("Training complete!")
