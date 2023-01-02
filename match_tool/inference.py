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

class Inference:
  THRESHOLD = 0.9
  def __init__(self, obj_name, match_tool=False):
    self.dists = []
    self.obj_name = obj_name
    # import object
    bpy.ops.import_scene.obj(filepath=f"../data_gen/models/{obj_name}.obj")
    self.obj = bpy.data.objects[obj_name]
    # get face centers
    self.centers = [tuple(face.center) for face in list(self.obj.data.polygons)]
    # pass centers to kdtree
    self.tree =  KDTree(self.centers, leaf_size=2)
    # load weights
    weights = torch.load('weights/mnist_weights.pt', map_location='cuda:0')
    # load model
    self.model = Net()
    self.model.cuda()
    self.model= nn.DataParallel(self.model)
    self.model.load_state_dict(weights, strict=False)
    self.model.eval()
    # Transform
    self.transform = transforms.Compose([
      transforms.Resize((256,256)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
  
  def load_image_mesh(self, img_path, mesh_path):
    # img
    print(img_path)
    img = Image.open(img_path).convert('RGB')
    # preprocess img and mesh
    self.img = self.transform(img).unsqueeze(0).float()
    self.mesh = torch.tensor(np.load(mesh_path)).permute(1, 0).unsqueeze(0).float()
  
  def inference(self):
    # inference
    output = self.model(self.mesh, self.img)
    return *self.output_to_image(output), output.squeeze(0)
    #Image.fromarray(np.uint8(image)).show()

  def query_row_to_center(self, row):
    row = [row.tolist()[1:]]
    dist, ind = self.tree.query(row, k=3)
    ind = ind[0,0]
    self.dists.append(dist[0,0])
    return (1,) + self.centers[ind]
  

  def output_to_image(self, output):
    # create new empty matrix
    image_matrix = np.zeros((256, 256, 4))
    color_matrix = np.zeros((256, 256, 3))
    # convert torch to numpy matrix
    output = output.cpu().detach().numpy()[0]

    # create image matrix
    idx1, idx2 = np.where(output[..., 0]>=Inference.THRESHOLD)
    image_matrix[idx1, idx2, ...] = output[idx1, idx2, ...]
    image_matrix[idx1, idx2, ...] = np.array(list(map(self.query_row_to_center, image_matrix[idx1, idx2, ...])))

    # extract different coordinates from matrix
    all_coordinates = set(tuple(row) for row in image_matrix[idx1, idx2, ...].tolist())

    # colors
    map_colors = {coord:np.random.randint(0, 255, size=(3,)) for coord in all_coordinates}

    # show dist mean
    print('Dist mean:', sum(self.dists)/len(self.dists))

    for i in range(256):
      for j in range(256):
        if image_matrix[i, j, 0] != 0:
          color_matrix[i, j] = map_colors[tuple(image_matrix[i, j])]

    return (image_matrix, all_coordinates), color_matrix

if __name__ == '__main__':
  # args
  argv = " ".join(sys.argv).replace(" -- ", "++").split("++")[1:]
  conf = {s.split(" ")[0]:s.split(" ")[1:] for s in argv}

  obj_name,img_path,mesh_path = conf['args']
  match_tool = True if 'match_tool' in conf.keys() else False

  # Inference
  inf = Inference(obj_name)

  inf.load_image_mesh(img_path=img_path, mesh_path=mesh_path)
  inf.inference()