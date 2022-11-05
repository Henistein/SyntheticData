import torch
import torch.nn as nn


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.l1 = nn.Sequential(
      nn.Conv2d(4, 256, 3),
    )
    self.l2 = nn.Sequential(
      nn.Conv2d(256, 256, 3),
      nn.MaxPool2d(2, 2)
    )
    self.l3 = nn.Sequential(
      nn.Conv2d(256, 512, 3),
    )
    self.l4 = nn.Sequential(
      nn.Conv2d(512, 512, 3),
      nn.MaxPool2d(2, 2)
    )
    self.l5 = nn.Sequential(
      nn.Conv2d(512, 1024, 3),
    )
    self.l6 = nn.Sequential(
      nn.Conv2d(1024, 1024, 3),
      nn.MaxPool2d(2, 2)
    )
    self.l7 = nn.Sequential(
      nn.Conv2d(1024, 2048, 3),
    )
    self.l8 = nn.Sequential(
      nn.Conv2d(2048, 2048, 5),
      nn.MaxPool2d(2, 2)
    )
    self.l9 = nn.Sequential(
      nn.Conv2d(2048, 2048, 2),
    )
    self.l10 = nn.Sequential(
      nn.Conv1d(2048, 2048, 3),
      nn.MaxPool1d(2,2)
    )
    self.part1 = nn.Sequential(
      self.l1,
      self.l2,
      self.l3,
      self.l4,
      self.l5,
      self.l6,
      self.l7,
      self.l8,
      self.l9,
    )

    self.part2 = nn.Sequential(
      self.l10
    )

  def forward(self, x):
    x = self.part1(x)
    x = x.view(x.shape[0], x.shape[1], -1)
    x = self.part2(x)
    x = x.permute(0, 2, 1)
    return x

class TransposeNet(nn.Module):
  def __init__(self):
    super(TransposeNet, self).__init__()
    self.l1 = nn.Sequential(
      nn.ConvTranspose1d(4, 256, 3),
    )
    self.l2 = nn.Sequential(
      nn.MaxUnpool1d(2, 2),
      nn.ConvTranspose1d(256, 256, 3),
    )
    self.l3 = nn.Sequential(
      nn.ConvTranspose1d(256, 512, 3),
    )
    self.l4 = nn.Sequential(
      nn.MaxUnpool1d(2, 2),
      nn.ConvTranspose1d(512, 512, 3),
    )
    self.l5 = nn.Sequential(
      nn.ConvTranspose1d(512, 1024, 3),
    )
    self.l6 = nn.Sequential(
      nn.MaxUnpool1d(2, 2),
      nn.ConvTranspose1d(1024, 1024, 3),
    )
    self.l7 = nn.Sequential(
      nn.ConvTranspose1d(1024, 2048, 3),
    )
    self.l8 = nn.Sequential(
      nn.MaxUnpool1d(2, 2),
      nn.ConvTranspose1d(2048, 2048, 5),
    )
    self.l9 = nn.Sequential(
      nn.ConvTranspose1d(2048, 2048, 2),
    )
    self.l10 = nn.Sequential(
      nn.MaxUnpool1d(2,2),
      nn.ConvTranspose1d(2048, 2048, 3)
    )
    self.part1 = nn.Sequential(
      *list(reversed([self.l1,
      self.l2,
      self.l3,
      self.l4,
      self.l5,
      self.l6,
      self.l7,
      self.l8,
      self.l9]))
    )

    self.part2 = nn.Sequential(
      self.l10
    )

  def forward(self, x):
    #x = self.l1(x)
    x = x.permute(0, 2, 1)
    x = self.l9(x)
    #x = self.l8(x)
    #x = self.l7(x)
    #x = self.l6(x)
    #x = self.l5(x)
    #x = self.l4(x)
    #x = self.part1(x)
    return x
    x = x.view(x.shape[0], x.shape[1], -1)
    x = self.part2(x)
    x = x.permute(0, 2, 1)
    return x


if __name__ == '__main__':
  # t = torch.rand((1, 4, 108, 192))
  # goal is (1, 2, 2048)
  t = torch.rand((1, 2, 2048))

  model = TransposeNet()
  out = model(t)
  print(out.shape)
