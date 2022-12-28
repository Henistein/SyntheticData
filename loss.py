import torch
import torch.nn as nn

# https://pylessons.com/YOLOv3-TF2-mnist
# https://github.com/mahdi-darvish/YOLOv3-from-Scratch-Analaysis-and-Implementation/blob/main/implementations/loss.py


class ComputeLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.bcel = nn.BCEWithLogitsLoss()
    self.sigmoid = nn.Sigmoid()
    self.mse = nn.MSELoss()
  
  def forward(self, preds=None, target=None):
    obj = target[..., 0] >= 0.5
    noobj = target[..., 0] < 0.5

    # NoObject Loss
    noobj_loss = self.bcel(
      (preds[..., 0:1][noobj]), (target[..., 0:1][noobj])
    )

    # Object Loss
    obj_loss = self.bcel(
      (preds[..., 0:1][obj]), (target[..., 0:1][obj])
    )

    # For Coordinates Loss
    coord_loss = self.mse(preds[..., 1:4][obj], target[..., 1:4][obj]) 

    return (noobj_loss + obj_loss + coord_loss)