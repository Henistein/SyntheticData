# create a pytorch dataset
import torch
import glob
import yaml
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from random import shuffle

class Dataset:
  def __init__(self, train_amt=0.8, data_path=None):
    self.DATA = {}
    images = []
    annotations = []
    for path in list(glob.glob(data_path+"/*")):
      name = path.split('/')[-1]

      for img in list(glob.glob(path+"/images/*")):
        p1, p2 = img.rsplit('/', 1)
        images.append(p1 + '/' + name+"_"+p2)
      for ann in list(glob.glob(path+"/annotations/*")):
        p1, p2 = ann.rsplit('/', 1)
        annotations.append(p1 + '/' + name+"_"+p2)

    images = sorted(images)
    annotations = sorted(annotations)

    temp = list(zip(images, annotations))
    shuffle(temp)
    images, annotations = zip(*temp)

    assert len(images) == len(annotations), "Images and annotations differ in size"


    self.train_data = {}
    self.train_data['images'] = images[:int(train_amt*len(images))]
    self.train_data['annotations'] = annotations[:int(train_amt*len(annotations))]

    self.test_data = {}
    self.test_data['images'] = images[int(train_amt*len(images)):]
    self.test_data['annotations'] = annotations[int(train_amt*len(annotations)):]

    assert len(self.train_data['images']) > len(self.test_data['images']), "Test images size greater than train images size"

  def __call__(self):
    return MyDataset(self.train_data), MyDataset(self.test_data)


class MyDataset(data.Dataset):
  def __init__(self, data):
    self.data = data
    self.transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
  
  def __len__(self):
    return len(self.data["images"])
  
  def __getitem__(self, index):
    img_path = self.data['images'][index]
    mesh_path = glob.glob(img_path.rsplit('/', 2)[0]+"/mesh/*")[0]
    img_path = img_path.rsplit('/', 1)[0] + '/' + img_path.rsplit('/', 1)[1].split('_')[1]
    ann_path = self.data['annotations'][index]
    ann_path = ann_path.rsplit('/', 1)[0] + '/' + ann_path.rsplit('/', 1)[1].split('_')[1]

    # load image
    img = Image.open(img_path).convert('RGB')
    # transform
    img = self.transform(img)

    # load annotations
    ann = torch.tensor(np.load(ann_path))

    # load mesh
    mesh = torch.tensor(np.load(mesh_path)).permute(1, 0)


    return (img,mesh,ann)

def load_dataloaders(bs):
  dataset = Dataset(data_path='/home/socialab/Desktop/Henrique/DATA_MNIST')
  train_dataset, val_dataset = dataset()

  return data.DataLoader(train_dataset, batch_size=bs, collate_fn=None, shuffle=True), \
         data.DataLoader(val_dataset, batch_size=bs, collate_fn=None, shuffle=True)


from tqdm import tqdm
if __name__ == '__main__':
  BS = 16
  train_loader, val_loader = load_dataloaders(BS)

  #for (img,mesh,ann) in tqdm(train_loader):
  for data in tqdm(train_loader):
    for d in data:
      imgs, meshes, anns = list(zip(*d))

      imgs = torch.stack(imgs)
      meshes = torch.stack(meshes)
      anns = torch.stack(anns)

      print(imgs.shape)
      print(meshes.shape)
      print(anns.shape)
      print()

    break
