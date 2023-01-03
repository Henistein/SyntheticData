import torch
import torch.nn as nn
from torch import Tensor

# https://pylessons.com/YOLOv3-TF2-mnist
# https://github.com/mahdi-darvish/YOLOv3-from-Scratch-Analaysis-and-Implementation/blob/main/implementations/loss.py


class ComputeLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.bcel = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0])).cuda()
    self.l1 = nn.SmoothL1Loss()
  
  def forward(self, preds=None, target=None):
    obj = target[..., 0] == 1
    noobj = target[..., 0] == 0

    # NoObject Loss
    noobj_loss = self.bcel(
      (preds[..., 0:1][noobj]), (target[..., 0:1][noobj])
    )

    # Object Loss
    obj_loss = self.bcel(
      (preds[..., 0:1][obj]), (target[..., 0:1][obj])
    )

    # For Coordinates Loss
    coord_loss = self.l1(preds[..., 1:4][obj], target[..., 1:4][obj]) 

    return (noobj_loss + obj_loss + coord_loss)

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)