# -*- coding: utf-8 -*-
"""HW3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10wU-6EXZCQp3sNwrWs3qfo8Dm4Gwgvc9
"""

import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
import time
import numpy as np
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_epochs = 12
batch_size = 64
learning_rate = 0.001

transformations = transforms.Compose(
    [transforms.RandomHorizontalFlip(0.5),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root= './',
                                           train=True,
                                           transform=transformations,
                                           download=True)
test_dataset = torchvision.datasets.CIFAR10(root= './',
                                           train=False,
                                           transform=transformations)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,num_workers=2)

input_shape = np.array(train_dataset[0][0]).shape
input_dim = input_shape[1]*input_shape[2]*input_shape[0]

class Flatten(nn.Module):
  """NN Module that flattens the incoming tensor."""
  def forward(self, input):
    return input.view(input.size(0), -1)
  
class ConvModel(nn.Module):
  def __init__(self):
    super(ConvModel, self).__init__()
    self.net = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size = 4, stride=1, padding=2),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.Conv2d(64, 64, kernel_size = 4, stride=1, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Dropout(0.2),
      nn.Conv2d(64, 64, kernel_size = 4, stride=1, padding=2),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.Conv2d(64, 64, kernel_size = 4, stride=1, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Dropout(0.2),
      nn.Conv2d(64, 64, kernel_size = 4, stride=1, padding=2),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.Conv2d(64, 64, kernel_size = 3, stride=1, padding=0),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Conv2d(64, 64, kernel_size = 3, stride=1, padding=0),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.Conv2d(64, 64, kernel_size = 3, stride=1, padding=0),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.Dropout(0.2),
      Flatten(),
      nn.Linear(1024,500),
      nn.ReLU(),
      nn.Linear(500,10))
    
  def forward(self, x):
    return self.net(x)

step = 0
start_time = time.time()
model = ConvModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
for epoch in range(num_epochs):
  correct = 0
  total = 0
  total_loss = 0
  ## Train
  model.train()
  for images,labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    #Forward Pass
    outputs = model(images)

    loss = criterion(outputs,labels)
    _,predicted = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    #Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    total_loss += loss.item()
    optimizer.step()
    step += 1
    train_accuracy = correct/total
    train_loss = total_loss/step

  ## Test
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data,1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    test_accuracy = correct/total
    print('Epoch {}, Time {:.4f}, Loss: {:.4f}, Train Accuracy: {:.4f}, Test Accuracy: {:.4f}'.format(
        epoch,time.time()-start_time,train_loss,train_accuracy,test_accuracy))