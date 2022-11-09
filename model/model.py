import torch
import torch.nn as nn

from model.pointnet import PointNetfeat
from model.unet_model import DownNet, UpNet

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fe_3d = PointNetfeat(global_feat=True)
    self.down_net = DownNet(3, n_classes=1)
    self.up_net = UpNet()
    self.conv1 = nn.Conv1d(1, 22*40, 1)
    self.avg = nn.AvgPool2d(2,2)
    self.out_conv = nn.Sequential(
        nn.Conv2d(64, 4, 1),
        nn.ReLU(),
        nn.AvgPool2d(5, 5),
        nn.BatchNorm2d(4)
    )


  def forward(self, mesh, image):
    features_3d, _, _ = self.fe_3d(mesh)   # (BS, 1, 1024)
    x1,x2,x3,x4,x5 = self.down_net(image)        # (BS, 1024, 22, 40)
    features_2d = x5

    features_3d = self.conv1(features_3d)
    features_3d = features_3d.permute(0, 2, 1)
    features_3d = features_3d.view(features_3d.shape[0], -1, 22, 40)
    # features_3d.shape = (BS, 1024, 22, 40)

    # concatenate two feature tensors
    cat = torch.cat((features_3d, features_2d), dim=1) # (BS, 2048, 22, 40)
    # repeat tensor
    rep = cat.repeat(1, 2, 1, 1)
    bs, _, s1, s2 = rep.shape
    out = self.avg(rep)
    out = out.view(bs, -1, s1, s2)

    # UpNet
    out = self.up_net(x1, x2, x3, x4, features_2d)

    # conv output (BS, 72, 128, 4)
    out = self.out_conv(out)
    out = out.permute(0, 2, 3, 1)

    return out


if __name__ == '__main__':
  model = Net()

  mesh = torch.rand((1, 3, 2500))
  image = torch.rand((1, 3, 1080//3, 1920//3))

  out = model(mesh, image)
  print(out.shape)
