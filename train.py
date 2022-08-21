# train efficientnet-b4 on this dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
from dataset import load_train_dataset, load_val_dataset
from torchvision.models import efficientnet_v2_l

class Train:
  EPOCHS = 400
  LEARNING_RATE = 1e-3

  def __init__(self, train_dataset, val_dataset):
    self.model = efficientnet_v2_l(weights=None, pretrained=True)
    # Hack to get the model to work with 2 output classes
    self.model.classifier = nn.Sequential(
      nn.Dropout(p=0.4, inplace=True),
      nn.Linear(in_features=1280, out_features=2, bias=True),
    )
    self.model.cuda().train()
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset 
    self.optimizer = optim.Adam(self.model.parameters(), lr=Train.LEARNING_RATE)
    self.criterion = nn.CrossEntropyLoss()
  
  def _train(self):
    for x,y in (t:=tqdm(self.train_dataset)):
      x, y = x.cuda(), y.cuda()
      self.optimizer.zero_grad()
      # Forward pass
      outputs = self.model(x)
      # Compute loss
      loss = self.criterion(outputs, y)
      # Backpropagation
      loss.backward()
      # Update weights
      self.optimizer.step()
      t.set_description(f'loss: {loss.item():.4f}')
  
  def _validate(self):
    self.model.eval()
    val_running_loss = 0.0
    val_running_correct = 0.0
    with torch.no_grad():
      for x,y in tqdm(self.val_dataset):
        x, y = x.cuda(), y.cuda()
        # Forward pass
        outputs = self.model(x)
        # Compute loss
        loss = self.criterion(outputs, y)
        val_running_loss += loss.item()
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        val_running_correct += (predicted == y).sum().item()
    epoch_loss = val_running_loss / len(self.val_dataset)
    epoch_acc = 100.0 * val_running_correct / len(self.val_dataset)
    return epoch_loss, epoch_acc
  
  def train(self):
    valid_acc, valid_loss = [], []

    for epoch in range(Train.EPOCHS):
      self._train()
      epoch_loss, epoch_acc = self._validate()
      valid_loss.append(epoch_loss)
      valid_acc.append(epoch_acc)
      print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')


if __name__ == '__main__':
  BS = 2
  train = Train(train_dataset=load_train_dataset(BS), val_dataset=load_val_dataset(BS))
  train.train()