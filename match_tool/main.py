import bpy
import bmesh
import sys
import os
import pickle
import cv2
import numpy as np

sys.path.insert(1, '/home/socialab/Henrique/SyntheticData/data_gen')

blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
  sys.path.append(blend_dir)

import scripts
from scripts import *
from match import get_coordinates_matches
from mathutils import Vector
import data_gen

from math import radians
from inference import Inference
from sklearn.neighbors import KDTree
from PIL import Image


RES = (900, 900)

def init(obj_name):
  bpy.data.scenes[0].render.engine = "CYCLES"
  bpy.context.scene.cycles.device = "GPU"

  # remove cube object
  bpy.data.objects['Cube'].select_set(True)
  bpy.ops.object.delete()

  # create camera plane
  plane = create_camera_plane()

  # import sign object
  bpy.ops.import_scene.obj(filepath=f"../data_gen/models/{obj_name}.obj")
  obj = bpy.data.objects[obj_name]

  # reset objects position
  obj.location.x = 0
  plane.rotation_euler.x = radians(0)
  plane.rotation_euler.y = radians(0)
  plane.rotation_euler.z = radians(0)
  plane.location.x = 0
  plane.location.y = 0
  plane.location.z = 0

  # add a curve (circle)
  bpy.ops.curve.primitive_bezier_circle_add(radius=10, location=(0, 0, 0))
  # add an empty cube
  bpy.ops.object.empty_add(type='CUBE')

  # make camera track sign
  camera_constraint = plane.constraints.new(type='TRACK_TO')
  camera_constraint.target = obj 

  # select empty and plane
  bpy.data.objects['Empty'].select_set(True)
  plane.select_set(True)

  # link camera to empty
  bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)

  # make empty follow path the bezier curve
  bpy.ops.object.constraint_add(type='FOLLOW_PATH')
  bpy.data.objects['Empty'].constraints['Follow Path'].target = bpy.data.objects['BezierCircle']
  bpy.data.objects['Camera'].location.z = 0

  # Reduce samples (faster rendering)
  bpy.context.scene.cycles.samples = 1024

  # background
  node_environment = create_background()

  return obj, node_environment


def load_data_from_pkl(path):
  path = '../data_gen/pkls/' + path
  # load data
  with open(path, 'rb') as f:
    data = pickle.load(f)

  for i in range(len(data)):
    data[i][-1] = "../data_gen/" + data[i][-1]
  
  return data

def get_faces_pixels(obj):
  # visible mesh
  new_plane = cut_obj_camera_view(bpy, bpy.data.objects['Plane'], obj)
  visible_faces = get_visible_mesh(bpy, new_plane, obj, visualize=True, get_indexes=True)

  faces_pixels = get_coordinates_matches(
    list(visible_faces.values()),
    bpy.data.objects['Camera'],
    bpy.context.scene
  )
  ret = {}
  for i,index in enumerate(visible_faces.keys()):
    ret[index] = faces_pixels[i]

  return ret

def get_centroids_from_faces(faces_pixels):
  centroids = []
  for _,co_2d in faces_pixels:
    aux_co = []
    for co in co_2d:
      co = (co[0], abs(RES[1]-1-co[1]))
      aux_co.append(co)
    centroids.append(list(map(int, centroid(aux_co))))
  
  return centroids

def debug_show_image_with_points(points):
  img = cv2.imread('img.png')

  # Loop through the points and draw them on the image
  for point in points:
      cv2.circle(img, tuple(point), 5, (0, 0, 255, 1), -1)

  # Show the image
  cv2.imshow('image', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == '__main__':
  # parse args
  argv = " ".join(sys.argv).replace(" -- ", "++").split("++")[1:]
  conf = {s.split(" ")[0]:s.split(" ")[1:] for s in argv}

  obj_name,img_path,mesh_path = conf['args']

  # load data
  data = load_data_from_pkl('data_10.pkl')
  sample = data[1]

  # import object
  obj, node_environment = init(obj_name)

  # create datagen
  dg = data_gen.CreateData(bpy, res=(256, 256), redux_factor=1, destination_path=None, debug=False, generated_data=[sample])
  dg.add_obj('empty', bpy.data.objects['Empty'])
  dg.add_obj('obj', obj)
  dg.add_obj('node_environment', node_environment)
  dg.feat_name = ['empty,constraints,Follow Path,offset', 'empty,constraints,Follow Path,influence', 'obj.location.x', 'obj.location.y', 'obj.location.z', 'obj.rotation_euler.x', 'obj.rotation_euler.y', 'obj.rotation_euler.z', 'node_environment.image']
  dg.create_random_sample(obj, debug=False, already_gen=True)

  # set resolution to RES
  bpy.context.scene.render.resolution_x = RES[0]
  bpy.context.scene.render.resolution_y = RES[1]
  # take screenshot
  bpy.context.scene.render.filepath = f'img.png' 
  bpy.ops.render.render(write_still=True)

  # extract the coordinates of each face in 2d image coordinates
  faces_pixels = get_faces_pixels(obj)

  # convert get the centroids of faces in 2d image coordinates
  centroids = get_centroids_from_faces(list(faces_pixels.values()))

  index_centroid = {}
  for i,index in enumerate(faces_pixels.keys()):
    index_centroid[index] = centroids[i]
  
  #debug_show_image_with_points(centroids)

  # Map the face centers with the faces
  MAP = {tuple(face.center):face for face in list(obj.data.polygons)}

  # pass centers to kdtree
  centers = np.array(list(MAP.keys()))
  tree =  KDTree(centers, leaf_size=2)

  # Inference
  inf = Inference(obj_name)

  inf.load_image_mesh(img_path=img_path, mesh_path=mesh_path)
  (image_matrix,all_coordinates), color_matrix, _ = inf.inference()

  # 2- Procurar correspondencias na matriz dado esse ponto, escolher o primeiro, guardar coordenada
  # 3- Fazer query desse valor, coloca-lo no MAP, obter o indice da face
  # 4- prints, testar

  # find a single element for each different coordinate
  
  """
  idx1, idx2 = np.where(image_matrix[..., 0]>0)
  pos_per_coordinate = set()
  for idx in range(len(all_coordinates)):
    for i in idx1:
      for j in idx2:
        row = image_matrix[i, j]
        if (row == list(all_coordinates)[idx]).all():
          pos_per_coordinate.add((i, j))
          break
  print(pos_per_coordinate)
  exit()
  """

  all_different_positions = {}
  for co in all_coordinates:
    indices = np.argwhere(np.all(np.isclose(image_matrix, np.array(co), rtol=1e-06, atol=1e-06), axis=2))
    idx1, idx2 = indices.T
    all_different_positions[(idx1[0], idx2[0])] = color_matrix[idx1[0], idx2[0]]

  #pred_image = Image.fromarray(np.uint8(color_matrix)).convert('RGB')
  pred_image = np.uint8(color_matrix)
  #pred_image = np.uint8(color_matrix)
  #pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
  gt_image = cv2.imread('img.png')
  pred_image = cv2.resize(pred_image, (900,900))
  #print(gt_image.shape)
  #print(pred_image.shape)
  #exit()
  gt_image = cv2.cvtColor(np.uint8(gt_image), cv2.COLOR_RGBA2RGB)
  result = cv2.hconcat([gt_image, pred_image])
  
  for co in all_different_positions.keys():
    # get coord from index co
    coord = image_matrix[co[0], co[1], 1:]
    # query coord
    dist, ind = tree.query([coord], k=3)
    ind = ind[0, 0]
    # get center
    center = tuple(centers[ind].flatten())
    # select face
    face = MAP[center]
    if face.index in index_centroid.keys():
      orig = index_centroid[face.index]
      dest = (co[1]*(900//256)+900, co[0]*(900//256))
      cv2.line(result, orig, dest, all_different_positions[co], 3)
    


    #print(face.index)
    #exit()
  Image.fromarray(result).show()

  #print(len(all_different_positions))
  #print(len(all_coordinates))
  #print(image_matrix[..., 1:])