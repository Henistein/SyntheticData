import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tqdm import tqdm 
from dataset import load_dataloaders
from model.model import Net
from loss import ComputeLoss

class Train:
  EPOCHS = 50
  LEARNING_RATE = 1e-5

  def __init__(self, train_dataset, val_dataset, weights):
    # load model
    self.model = Net()
    self.model.cuda()
    self.model= nn.DataParallel(self.model, device_ids=[0,2,3])
    self.model.load_state_dict(torch.load(weights), strict=False) if weights else self.model
    # dataset 
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset 
    # params
    self.optimizer = optim.Adam(self.model.parameters(), lr=Train.LEARNING_RATE, weight_decay=1e-8)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2)
    self.grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    self.criterion = ComputeLoss()
  
  def _mean(self, x):
    return sum(x)/len(x)
  
  def _train(self):
    self.model.train()
    train_loss, train_acc = [], []
    for (imgs, meshes, anns) in (t:=tqdm(self.train_dataset)):
      imgs = imgs.cuda().float()
      meshes = meshes.cuda().float()
      anns = anns.cuda().float()

      self.optimizer.zero_grad()

      with torch.cuda.amp.autocast(enabled=False):
        # Forward pass
        outputs = self.model(meshes, imgs)
        # Compute loss
        loss = self.criterion(preds=outputs, target=anns)

      # save train loss
      train_loss.append(loss.item())

      # Backpropagation and Update weights
      self.optimizer.zero_grad(set_to_none=True)
      self.grad_scaler.scale(loss).backward()
      self.grad_scaler.step(self.optimizer)
      self.grad_scaler.update()

      t.set_description(f'loss: {loss.item():.4f}')

    return self._mean(train_loss)
  
  def _validate(self):
    self.model.eval()
    val_loss, val_acc = [], []
    with torch.no_grad():
      for (imgs, meshes, anns) in (t:=tqdm(self.val_dataset)):
          imgs = imgs.cuda().float()
          meshes = meshes.cuda().float()
          anns = anns.cuda().float()

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

      # scheduler step
      self.scheduler.step(epoch_loss)

      if best_loss is None or epoch_loss < best_loss:
        # save the best model
        torch.save(self.model.state_dict(), path+"/best.pt")
      # save the last model 
      torch.save(self.model.state_dict(), path+"/last.pt")
      # save stats
      np.save(path+"/train_loss.npy", np.array(train_loss))
      np.save(path+"/valid_loss.npy", np.array(valid_loss))

      print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}')
    
    self.show_loss(t_loss=train_loss, v_loss=valid_loss)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=32, help='batch size')
  parser.add_argument('--save_path', type=str, default='', help='path to save the results')
  parser.add_argument('--load_model', type=str, default='', help='load model weights')
  parser = parser.parse_args()

  BS = parser.batch_size
  train_loader, val_loader = load_dataloaders(BS)
  train = Train(train_dataset=train_loader, val_dataset=val_loader, weights=parser.load_model)
  train.train(path=parser.save_path)
