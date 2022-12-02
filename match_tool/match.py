import bpy
import cv2
import sys
import socket
import threading
import numpy as np

from ..inference import Inference
from PIL import Image
from math import radians
from sklearn.neighbors import KDTree

# ------------------ T1 -----------------------
def loop(grid, MAP, tree, ann):
  ann = ann[..., 1:]
  # Extract keys from MAP
  centers = np.array(list(MAP.keys()))
  #center = tuple(centers[ind].flatten())
  #print(MAP[center].index)

  # ---------------------
  host = socket.gethostname()
  port = 5000

  server_socket = socket.socket()
  server_socket.bind((host, port))
  server_socket.listen(1)

  conn, address = server_socket.accept()
  while True:
    data = str(conn.recv(1024).decode())
    if not data:
      break

    x, y, z = data.split(' ')

    coord = [(float(x), float(y), float(z))]
    dist, ind = tree.query(coord, k=3)
    ind = ind[0, 0]

    center = tuple(centers[ind].flatten())

    # select face
    face = MAP[center]
    print("FROM SERVER: ", face.index)
    face.select = True
    
    # select image pixels
    indexes = np.where(np.all(ann == coord[0], axis=2))
    indexes = list(zip(indexes[0], indexes[1]))
    indexes = list(map(convert_2d_to_1d, indexes))

    for i in indexes:
      list(grid.data.polygons)[i].select = True
      
  conn.close()

def convert_2d_to_1d(pt, W=256):
  # convert a 2d point to a flatten index
  return pt[0]*W + pt[1]

# args
"""
npy path
name
obj_name
img_path
"""
argv = " ".join(sys.argv).replace(" -- ", "++").split("++")[1:]
conf = {s.split(" ")[0]:s.split(" ")[1:] for s in argv}

obj_name,img_path,mesh_path = conf['args']

bpy.ops.wm.open_mainfile(filepath="match_tool_model.blend")

# load data
grid = bpy.data.objects['Grid']
bpy.data.materials['Material'].node_tree.nodes['Image Texture'].image = bpy.data.images.load(img_path)

# import object
bpy.ops.import_scene.obj(filepath=f"../data_gen/models/{obj_name}.obj")
obj = bpy.data.objects[obj_name]

# place object in the side of the grid
obj.location.x = -3.80
obj.rotation_euler.x = radians(180)
obj.rotation_euler.y = radians(0)
obj.rotation_euler.z = radians(0)

# Map the face centers with the faces
MAP = {tuple(face.center):face for face in list(obj.data.polygons)}

# pass centers to kdtree
centers = np.array(list(MAP.keys()))
tree =  KDTree(centers, leaf_size=2)

# Inference
inf = Inference(obj_name)

inf.load_image_mesh(img_path=img_path, mesh_path=mesh_path)
inf.inference()

if 1 == 0:

  # ----------------
  bpy.context.view_layer.objects.active = obj
  t1 = threading.Thread(target=loop, args=(grid,MAP,tree))
  t1.start()
