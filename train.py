# train efficientnet-b4 on this dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm 
from dataset import load_train_dataset, load_val_dataset
from torchvision.models import efficientnet_v2_s

class Train:
  EPOCHS = 3
  LEARNING_RATE = 1e-3

  def __init__(self, train_dataset, val_dataset):
    self.model = efficientnet_v2_s(weights=None, pretrained=True)
    # Hack to get the model to work with 2 output classes
    self.model.classifier = nn.Sequential(
      nn.Dropout(p=0.4, inplace=True),
      nn.Linear(in_features=1280, out_features=2, bias=True),
    )
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model.to(self.device)
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset 
    self.optimizer = optim.Adam(self.model.parameters(), lr=Train.LEARNING_RATE)
    self.criterion = nn.CrossEntropyLoss()
  
  def _mean(self, x):
    return sum(x)/len(x)
  
  def _train(self):
    self.model.train()
    train_loss, train_acc = [], []
    i = 0
    for x,y in (t:=tqdm(self.train_dataset)):
      x, y = x.to(self.device), y.to(self.device)
      self.optimizer.zero_grad()
      # Forward pass
      outputs = self.model(x)
      # Compute loss
      loss = self.criterion(outputs, y)
      # Calculate the accuracy and save it
      predicted = torch.argmax(outputs, dim=1)
      train_acc.append((predicted == y).float().mean().item())
      train_loss.append(loss.item())
      # Backpropagation
      loss.backward()
      # Update weights
      self.optimizer.step()
      t.set_description(f'loss: {loss.item():.4f}')
      if i == 2:
        break
      i+=1
    return self._mean(train_acc), self._mean(train_loss)
  
  def _validate(self):
    self.model.eval()
    val_loss, val_acc = [], []
    i = 0
    with torch.no_grad():
      for x,y in tqdm(self.val_dataset):
        x, y = x.cuda(), y.cuda()
        # Forward pass
        outputs = self.model(x)
        # Compute loss
        loss = self.criterion(outputs, y)
        # Calculate accuracy
        predicted = torch.argmax(outputs, dim=1)
        val_acc.append((predicted == y).float().mean().item())
        val_loss.append(loss.item())
        if i == 2:
          break
        i+=1
    return self._mean(val_acc), self._mean(val_loss)
  
  def compute_acc_loss(self, t_acc, v_acc, t_loss, v_loss):
    print(t_acc, v_acc, t_loss, v_loss)
    plt.plot(t_acc)
    plt.plot(v_acc)
    plt.plot(t_loss)
    plt.plot(v_loss)
    plt.savefig('results.png')

  def train(self, path):
    valid_acc, valid_loss = [], []
    train_acc, train_loss = [], []
    best_acc = 0

    for epoch in range(Train.EPOCHS):
      # train
      epoch_loss, epoch_acc = self._train()
      train_loss.append(epoch_loss)
      train_acc.append(epoch_acc)
      # validation
      epoch_loss, epoch_acc = self._validate()
      valid_loss.append(epoch_loss)
      valid_acc.append(epoch_acc)

      if epoch_acc > best_acc:
        # save the best model
        torch.save(self.model.state_dict(), path+"/best.pt")
      # save the last model 
      torch.save(self.model.state_dict(), path+"/last.pt")

      print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    
    self.compute_acc_loss(t_acc=train_acc, v_acc=valid_acc, t_loss=train_loss, v_loss=valid_loss)


if __name__ == '__main__':
  BS = 14
  train = Train(train_dataset=load_train_dataset(BS), val_dataset=load_val_dataset(BS))
  train.train(path='/home/henistein/programming/UBI/SyntheticData/results')