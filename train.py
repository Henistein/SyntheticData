# train efficientnet-b4 on this dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
from dataset import load_dataset
from torchvision.models import efficientnet_v2_l

#print(model.classifier)

#x = torch.randn(1, 3, 380, 380)
#y = model(x)
#print(y.shape)

class Train:
  EPOCHS = 400
  LEARNING_RATE = 1e-3
  MOMENTUM = 0.9
  WEIGHT_DECAY = 5e-6
  LABEL_SMOOOTHING = 0.1
  MIX_UP = 0.2

  def __init__(self, dataset):
    self.model = efficientnet_v2_l(weights=None, pretrained=False)
    # Hack to get the model to work with 2 output classes
    self.model.classifier = nn.Sequential(
      nn.Dropout(p=0.4, inplace=True),
      nn.Linear(in_features=1280, out_features=2, bias=True),
    )
    self.model.cuda().train()
    self.dataset = dataset
    self.optimizer = optim.RMSprop(self.model.parameters(), lr=Train.LEARNING_RATE)
  
  def train(self):
    for i in range(Train.EPOCHS):
      dataset = tqdm(self.dataset)
      for x,y in dataset:
        x, y = x.cuda(), y.cuda()
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y, reduction='mean')
        loss.backward()
        self.optimizer.step()
      dataset.set_description(f'loss: {loss.item():.4f}')
      


if __name__ == '__main__':
  train = Train(load_dataset(2))
  train.train()