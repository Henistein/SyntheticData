import bpy
import cv2
import socket
import threading
import numpy as np

from PIL import Image
from math import radians
from sklearn.neighbors import KDTree

# ------------------ T1 -----------------------
def loop(obj, MAP, tree, ann):
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
    indexes = np.where(np.all(ann == center, axis=2))
    print(center)
    indexes = list(zip(indexes[0], indexes[1]))
    indexes = list(map(convert_2d_to_1d, indexes))

    for i in indexes:
      print(i)
      list(obj.data.polygons)[i].select = True
      print(list(obj.data.polygons)[i].select)
        
      
  conn.close()

# ------------------ T1 -----------------------


def convert_2d_to_1d(pt, W=256):
  # convert a 2d point to a flatten index
  return pt[0]*W + pt[1]

bpy.ops.wm.open_mainfile(filepath="match_tool_model.blend")

data = np.load('a000420.npy')
grid = bpy.data.objects['Grid']

# import object
name = 'sofa'
obj_name = 'Sofa'
bpy.ops.import_scene.obj(filepath=f"../data_gen/models/{name}.obj")
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

# create a set with all different coordinates from annotations
#all_coords = list(set(map(tuple, data[..., 1:].reshape(-1,3))))


# query center
#x = [(-0.55251166, 0.412381, -0.08782556)]
#x = [all_coords[0]]
#dist, ind = tree.query(x, k=3)
#ind = ind[0, 0]

# get center
#center = tuple(centers[ind].flatten())
#print(MAP[center].index)


# ----------------
#path = 'big_000420.png'
# show image
#bpy.ops.mesh.select_all(action='DESELECT')
bpy.context.view_layer.objects.active = obj
t1 = threading.Thread(target=loop, args=(obj,MAP,tree,data))
t1.start()



# query
#res = np.where(np.all(data == data[147, 157], axis=2))
#res = list(zip(res[0], res[1]))
#res = list(map(convert_2d_to_1d, res))

#faces = list(grid.data.polygons)


"""
for i in res:
  faces[i].select = True
"""

