# train efficientnet-b4 on this dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import load_dataset
from torchvision.models import efficientnet_v2_l

model = efficientnet_v2_l(weights=None, pretrained=False)
# Hack to get the model to work with 2 output classes
model.classifier = nn.Sequential(
  nn.Dropout(p=0.4, inplace=True),
  nn.Linear(in_features=1280, out_features=2, bias=True),
)
model.train()
print(model.classifier)

x = torch.randn(1, 3, 380, 380)
y = model(x)
print(y.shape)

