import torch
import torch.nn as nn
import glob
import sys
import os
import bpy
from PIL import Image

blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
  sys.path.append(blend_dir)

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from model.model import Net
from sklearn.neighbors import KDTree

def query_row_to_center(row):
  row = [row.tolist()[1:]]
  dist, ind = tree.query(row, k=3)
  ind = ind[0,0]
  return (1,) + centers[ind]
  

def output_to_image(output):
  # create new empty matrix
  image_matrix = np.zeros((256, 256, 4))
  color_matrix = np.zeros((256, 256, 3))
  # convert torch to numpy matrix
  output = output.cpu().detach().numpy()[0]

  # create image matrix
  idx1, idx2 = np.where(output[..., 0]>0)
  image_matrix[idx1, idx2, ...] = output[idx1, idx2, ...]
  image_matrix[idx1, idx2, ...] = np.array(list(map(query_row_to_center, image_matrix[idx1, idx2, ...])))

  # extract different coordinates from matrix
  all_coordinates = set(tuple(row) for row in image_matrix[idx1, idx2, ...].tolist())

  # colors
  map_colors = {coord:np.random.randint(0, 255, size=(3,)) for coord in all_coordinates}

  for i in range(256):
    for j in range(256):
      if image_matrix[i, j, 0] != 0:
        color_matrix[i, j] = map_colors[tuple(image_matrix[i, j])]

  return color_matrix

# args
argv = " ".join(sys.argv).replace(" -- ", "++").split("++")[1:]
conf = {s.split(" ")[0]:s.split(" ")[1:] for s in argv}

name,obj_name = conf['args']

# import object
bpy.ops.import_scene.obj(filepath=f"data_gen/models/{name}.obj")
obj = bpy.data.objects[obj_name]

# get face centers
centers = [tuple(face.center) for face in list(obj.data.polygons)]
# pass centers to kdtree
tree =  KDTree(centers, leaf_size=2)

# load weights
weights = torch.load('weights/best.pt')
# load model
model = Net()
model.cuda()
model= nn.DataParallel(model)
model.load_state_dict(weights, strict=False)
model.eval()

# Transform
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

path = '/media/socialab/53523fbd-9f42-4704-95e6-cbd31933c196/MNIST_TEST/000001'
image_list = glob.glob(path+'/images/*')
mesh_path = glob.glob(path+'/mesh/*')[0]
img = Image.open(image_list[0]).convert('RGB')

# preprocess img and mesh
img = transform(img).unsqueeze(0).float()
mesh = torch.tensor(np.load(mesh_path)).permute(1, 0).unsqueeze(0).float()

# inference
output = model(mesh, img)
image = output_to_image(output)
Image.fromarray(np.uint8(image)).show()
