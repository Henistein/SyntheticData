import torch
import torch.nn as nn

from pointnet import PointNetfeat
from ResNet import ResNet50

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fe_3d = PointNetfeat(global_feat=True)
    self.fe_2d = ResNet50()

  def forward(self, mesh, image):
    features_3d, _, _ = self.fe_3d(mesh)   # (BS, 1024)
    features_2d = self.fe_2d(image)        # (BS, 2048)

    assert features_2d.shape[0] == features_3d.shape[0], "Both Feature Tensors must have the same BS"
    bs = features_2d.shape[0]

    # repeat
    features_3d = features_3d.repeat(1, 2) # (BS, 2048)

    features_2d = features_2d.view(bs, 1, -1)
    features_3d = features_3d.view(bs, 1, -1)

    # concatenate features
    cat = torch.cat((features_3d, features_2d), 1)
    return cat

if __name__ == '__main__':
  model = Net()

  mesh = torch.rand((3, 3, 2500))
  image = torch.rand((3, 3, 1080, 1920))

  out = model(mesh, image)
  print(out.shape)
