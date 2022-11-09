# train efficientnet-b4 on this dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm 
from dataset import load_dataloaders
from model.model import Net
from loss import ComputeLoss

class Train:
  EPOCHS = 50
  LEARNING_RATE = 1e-3

  def __init__(self, train_dataset, val_dataset):
    self.model = Net()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model.to(self.device)
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset 
    self.optimizer = optim.Adam(self.model.parameters(), lr=Train.LEARNING_RATE)
    self.criterion = ComputeLoss()
  
  def _mean(self, x):
    return sum(x)/len(x)
  
  def _train(self):
    self.model.train()
    train_loss, train_acc = [], []
    for batch in (t:=tqdm(self.train_dataset)):
      for b in batch:
        imgs, meshes, anns = list(zip(*b))

        imgs = torch.stack(imgs).to(self.device).float()
        meshes = torch.stack(meshes).to(self.device).float()
        anns = torch.stack(anns).to(self.device).float()

        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(meshes, imgs)

        # Compute loss
        loss = self.criterion(preds=outputs, target=anns)

        train_loss.append(loss.item())

        # Backpropagation
        loss.backward()

        # Update weights
        self.optimizer.step()

        t.set_description(f'loss: {loss.item():.4f}')
        """
        # Calculate the accuracy and save it
        predicted = torch.argmax(outputs, dim=1)
        train_acc.append((predicted == y).float().mean().item())
        """

    return self._mean(train_loss)
  
  def _validate(self):
    self.model.eval()
    val_loss, val_acc = [], []
    with torch.no_grad():
      for batch in (t:=tqdm(self.val_dataset)):
        for b in batch:
          imgs, meshes, anns = list(zip(*b))

          imgs = torch.stack(imgs).to(self.device).float()
          meshes = torch.stack(meshes).to(self.device).float()
          anns = torch.stack(anns).to(self.device).float()

          self.optimizer.zero_grad()

          # Forward pass
          outputs = self.model(meshes, imgs)

          # Compute loss
          loss = self.criterion(preds=outputs, target=anns)

          val_loss.append(loss.item())

    return self._mean(val_loss)
  
  def show_loss(self, t_loss, v_loss):
    print(t_loss, v_loss)
    plt.plot(t_loss)
    plt.plot(v_loss)
    plt.savefig('results.png')

  def train(self, path):
    train_loss, valid_loss = [], []
    best_loss = None

    for epoch in range(Train.EPOCHS):
      # train
      epoch_loss = self._train()
      train_loss.append(epoch_loss)
      #train_acc.append(epoch_acc)
      # validation
      epoch_loss = self._validate()
      valid_loss.append(epoch_loss)
      #valid_acc.append(epoch_acc)

      if best_loss is None or epoch_loss < best_loss:
        # save the best model
        torch.save(self.model.state_dict(), path+"/best.pt")
      # save the last model 
      torch.save(self.model.state_dict(), path+"/last.pt")

      print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}')
    
    self.show_loss(t_loss=train_loss, v_loss=valid_loss)


if __name__ == '__main__':
  BS = 3
  train_loader, val_loader = load_dataloaders(BS)
  train = Train(train_dataset=train_loader, val_dataset=val_loader)
  train.train(path='/home/socialab/Henrique/SyntheticData/results')