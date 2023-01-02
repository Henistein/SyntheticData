import torch
import glob
import yaml
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import re
import os
from PIL import Image
from random import shuffle
#from pytorch3d.structures import Meshes
#from pytorch3d.io import load_obj

import warnings
warnings.filterwarnings("ignore")

class Dataset:
  def __init__(self, train_amt=0.8, data_path=None):
    self.DATA = {}
    images = {}
    annotations = {}
    for path in list(glob.glob(data_path+"/*")):
      name = path.split('/')[-1]
      for img in list(glob.glob(path+"/images/*")):
        p1, p2 = img.rsplit('/', 1) # p1: path/images p2: img{index}.png
        index = p2.split('.')[0].replace('img', '')
        images[name+index] = p1 + '/' + name+"_"+p2 # path/images/name_img{index}.png
      for ann in list(glob.glob(path+"/annotations/*")):
        p1, p2 = ann.rsplit('/', 1) # p1: path/annotations p2: a{index}.npy
        index = p2.split('.')[0].replace('a', '')
        annotations[name+index] = p1 + '/' + name+"_"+p2 # path/annotations/name_a{index}.npy

    self.data = []
    for key in images.keys():
      if key in annotations.keys():
        self.data.append((images[key], annotations[key]))

    # shuffle data
    shuffle(self.data) 

    self.train_data = self.data[:int(0.8*len(self.data))]
    self.test_data = self.data[int(0.8*len(self.data)):]

    assert (len(self.train_data) + len(self.test_data)) == len(self.data), "len(Train+Test) <> len(data)"


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
    return len(self.data)
  def __getitem__(self, index):
    _img_path, _ann_path = self.data[index]
    name = _img_path.rsplit('/', 2)[0]
    assert name == _ann_path.rsplit('/', 2)[0], "Different names in ann and image path"
    # load mesh path from name
    mesh_path = glob.glob(name+"/mesh/*.npy")[0]        

    # change img and ann path string to match to the real one
    img_path = _img_path.rsplit('/', 1)[0] + '/' + _img_path.rsplit('/', 1)[1].split('_')[-1] 
    ann_path = _ann_path.rsplit('/', 1)[0] + '/' + _ann_path.rsplit('/', 1)[1].split('_')[-1]

    # Very important assert
    assert int(_img_path.rsplit('_')[-1].split('.')[0].replace('img', '')) == int(_ann_path.rsplit('_')[-1].split('.')[0].replace('a', '')), "Different match between image and ann: " + _img_path + " " + _ann_path

    # load image
    img = Image.open(img_path).convert('RGB')
    # transform
    img = self.transform(img)

    # load annotations
    ann = torch.tensor(np.load(ann_path))

    # load mesh
    mesh = torch.tensor(np.load(mesh_path)).permute(1, 0)


    return (img,mesh,ann)

def collate_fn(batch):                                                   
  imgs, meshes, anns = list(zip(*batch))                                      
  #all_verts, all_faces = list(zip(*meshes))
  #all_verts = list(all_verts)                                            
  #all_faces = list(map(lambda x: x.verts_idx, all_faces))            
  # create batch with meshes                                                 
  #batch_meshes = Meshes(verts=all_verts, faces=all_faces).verts_padded()  
                                                                                          
  # stack imgs and anns                                                       
  imgs = torch.stack(imgs)                                                                  
  meshes = torch.stack(meshes)                                       
  anns = torch.stack(anns)                                                                
                                                                                           
  return (imgs,meshes,anns)

def collate_fn(batch):
  imgs, meshes, anns = list(zip(*batch))
  all_verts, all_faces = list(zip(*meshes))

  # use pytorch3d to stack meshes with different size
  all_verts = list(all_verts)
  all_faces = list(map(lambda x: x.verts_idx, all_faces))
  # create batch with meshes
  batch_meshes = Meshes(verts=all_verts, faces=all_faces).verts_padded()

  # stack imgs and anns
  imgs = torch.stack(imgs)
  anns = torch.stack(anns)

  return (imgs,batch_meshes,anns)


def collate_fn(batch):
  imgs, meshes, anns = list(zip(*batch))
  #all_verts, all_faces = list(zip(*meshes))

  # use pytorch3d to stack meshes with different size
  #all_verts = list(all_verts)
  #all_faces = list(map(lambda x: x.verts_idx, all_faces))
  # create batch with meshes
  #batch_meshes = Meshes(verts=all_verts, faces=all_faces).verts_padded()

  # stack imgs and anns
  imgs = torch.stack(imgs)
  meshes = torch.stack(meshes)
  anns = torch.stack(anns)

  return (imgs,meshes,anns)


def load_dataloaders(bs):
  dataset = Dataset(data_path='/media/socialab/53523fbd-9f42-4704-95e6-cbd31933c196/DATA')
  train_dataset, val_dataset = dataset()

  return data.DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn, shuffle=True), \
         data.DataLoader(val_dataset, batch_size=bs, collate_fn=collate_fn, shuffle=True)


from tqdm import tqdm
if __name__ == '__main__':
  BS = 16
  train_loader, val_loader = load_dataloaders(BS)

  #for (img,mesh,ann) in tqdm(train_loader):
  for data in tqdm(train_loader):
    pass

