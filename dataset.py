# create a pytorch dataset
import torch
import glob
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from random import shuffle

class Dataset(data.Dataset):
  def __init__(self, paths, train_amt=0.8, train=True):
    paths = list(glob.glob(paths+'/*'))
    # get last part of path 
    self.class_names = {path.split('/')[-1]:i for i,path in enumerate(paths)}
    all_data_paths = []
    for path in paths:
      all_data_paths.extend(glob.glob(path+"/*"))
    shuffle(all_data_paths)
    self.train_data = all_data_paths[:int(train_amt*len(all_data_paths))]
    self.test_data = all_data_paths[int(train_amt*len(all_data_paths)):]
    self.train = train
    self.transform = transforms.Compose([
      transforms.Resize(380),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

  def train(self):
    self.train = True

  def test(self):
    self.train = False

  def __len__(self):
    return len(self.train_data) if self.train else len(self.test_data)

  def __getitem__(self, index):
    path = self.train_data[index] if self.train else self.test_data[index]
    target = self.class_names[path.split('/')[-2]]
    # open and preprocess image
    img = Image.open(path).convert('RGB')
    img = self.transform(img)
    return img, target

def load_dataset():
  dataset = Dataset(
    '/media/henistein/Novo volume/SyntheticData',
  )
  return data.DataLoader(dataset, batch_size=10, shuffle=True)


if __name__ == '__main__':
  load_dataset()