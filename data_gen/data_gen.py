#import pandas as pd
import os
import numpy as np
import pickle
from match import filter_non_visible_coords, filter_repeated_coords, get_coordinates_matches, visualize_vertices
from scripts import cut_obj_camera_view, get_polygon_indexes, get_visible_mesh, create_mask

from math import prod, radians
from random import shuffle, choice
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from PIL import Image
from mathutils import Vector
from tqdm import tqdm

class ListStruct:
  def __init__(self, lst, total_combs):
    self.lst = lst
    self.elements = {k:total_combs/len(lst) for k in lst}
  
  def request_element(self):
    if len(self.lst) == 0: return None
    element = choice(self.lst)
    self.elements[element] -= 1
    if self.elements[element] == 0: self.lst.remove(element)
    return element

class ListProd:
  def __init__(self, lists):
    self.total_combs = prod([len(x) for x in lists])
    self.lists = [ListStruct(lst, self.total_combs) for lst in lists]
  
  def request_data(self, ammount):
    assert ammount >= 1
    assert ammount <= self.total_combs, 'Not enough combinations!'
    return [[lst.request_element() for lst in self.lists] for _ in range(ammount)]

class ObjFeatures:
  def __init__(self, obj):
    self.obj = obj
    self.features = {}

class DataGen:
  def __init__(self):
    self.objs = {}
    self.curr_obj = None
    self.image_index = 0
  
  @property
  def total_features(self): return sum([len(x.features) for x in self.objs.values()])

  @property
  def all_combinations(self): return prod([len(x) for obj in self.objs.values() for x in obj.features.values()])

  @property
  def feature_names(self): return [ft_name for obj in self.objs.values() for ft_name in obj.features]

  def add_obj(self, name, obj):
    self.objs[name] = ObjFeatures(obj)
    self.curr_obj = name

  def add_feature(self, atr, frm, to, step=1, callback=lambda x: x):
    """
    If is dict the methods must be separated with commas
    If don't it must be separated with dots
    Examples:
      - obj,some_dict,key,atr
      - obj.some_method.atr
    """
    s = ',' if ',' in atr else '.'
    self.objs[self.curr_obj].features[self.curr_obj+s+atr] = self.calculate_feature_list((frm, to, step), callback)
  
  def add_elements(self, atr, elements):
    self.objs[self.curr_obj].features[self.curr_obj+'.'+atr] = elements
  
  def calculate_feature_list(self, frm_to_step, callback):
    _, _, step = frm_to_step 
    decimals = 1
    while int(str(step*decimals).split('.')[0]) == 0: decimals *= 10
    lst = list(range(*tuple(map(lambda x: int(x*decimals), frm_to_step))))
    # recover decimal places
    lst = list(map(lambda x: x/decimals, lst))
    lst = list(map(callback, lst))
    shuffle(lst)
    return lst
  
  def generate(self, ammount):
    assert ammount <= self.all_combinations, 'Not enough combinations!'
    prod_list = ListProd([lst for obj in self.objs.values() for lst in obj.features.values()])
    return prod_list.request_data(ammount)


def _setattr(obj, attr_s, val):
  parts = attr_s.split('.')
  for part in parts[:-1]:
    obj = getattr(obj, part)
  setattr(obj, parts[-1], val)

class CreateData(DataGen):
  def __init__(self, blender, destination_path, debug=False):
    super().__init__()
    self.blender = blender
    self.destination_path = destination_path
    self.debug = debug
    # create image, annotations and maybe debug folders
    if not os.path.exists(destination_path): os.mkdir(destination_path)
    if not os.path.exists(destination_path+"/images"): os.mkdir(destination_path+"/images")
    if not os.path.exists(destination_path+"/annotations"): os.mkdir(destination_path+"/annotations")
    if debug: 
      if not os.path.exists(destination_path+"/debug"): os.mkdir(destination_path+"/debug")
  
  def generate(self, ammount):
    self.generated_data = super().generate(ammount)
  
  def create_annotations(self, obj, environment):
    # visible mesh
    new_plane = cut_obj_camera_view(self.blender, self.blender.data.objects['Plane'], obj)
    visible_faces = get_visible_mesh(self.blender, new_plane, obj, visualize=True)

    faces_pixels = get_coordinates_matches(
      visible_faces,
      self.blender.data.objects['Camera'],
      self.blender.context.scene
    )
    
    # create matrix
    M = np.zeros((1080, 1920), dtype=np.int64)
    C = np.zeros((1080, 1920, 3))
    F = {}
    inc = 0
    for face,co_2d in faces_pixels:
      #co_2d = [tuple(obj.matrix_world @ Vector(co)) for co in co_2d]
      indices = get_polygon_indexes(co_2d)
      color = np.random.randint(0, 255, size=(3,)).tolist()
      for i,j in indices:
        if i >= 1920 or j >= 1080:
          print(f'Skiped index ({i},{j}')
          continue
        j = abs(1080-j)
        face_coords = [item for sublist in face for item in sublist] 
        F[inc] = {"face":face_coords, "color":color}
        M[j, i] = inc
        #M[j, i, 0] = 1
        #M[j, i, 1:] = face_coords

        C[j, i] = color
      inc += 1
    
    # load vertices MAP
    with open('sofa_1020.pkl', 'rb') as f:
      MAP = pickle.load(f)

    NEW_M = np.zeros((108, 192, 3))
    NEW_C = np.zeros((108, 192, 3))
    # perform the reduction to 10 times
    for i in range(0, 1080, 10):
      for j in range(0, 1920, 10):
        most_freq = np.bincount(M[i:i+10, j:j+10].flatten()).argmax()
        face = F[most_freq]["face"]
        # WORLD_MATRIX * VERTICE = VTRANSFORMADO
        # VERTICE = np.linalg.solve(WORLD_MATRIX, VTRANSFORMADO)
        face = np.stack([np.linalg.solve(obj.matrix_world, face[i:i+3]+[1])[:-1] for i in range(0, 12, 3)]).flatten()
        face = tuple(round(v, 3) for v in face)
        NEW_M[i//10, j//10] = MAP[face]
        NEW_C[i//10, j//10] = F[most_freq]["color"]

    img = Image.fromarray(np.uint8(C))
    img.save('big_matrix.PNG')

    img = Image.fromarray(np.uint8(NEW_C))
    img.save('little_matrix.PNG')

    #return faces_pixels
    #co_2d = list(zip(*faces_pixels))[1]
    #co_2d = [item for sublist in co_2d for item in sublist]
    #visualize_vertices([item for sublist in co_2d for item in sublist])
    
    # create mask
    """
    mask = create_mask(environment).T
    print(mask.shape)

        
    for faces,pixels in faces_pixels:
      flat_faces = [item for sublist in faces for item in sublist]
      for p in pixels:
        M[p[0], p[1], 0] = 1
        M[p[0], p[1], 1:] = flat_faces

    left_pixels = []
    for i in range(1920):
      for j in range(1080):
        if mask[i, j] != 0 and M[i, j, 0] == 0:
          M[i, j, 0] = -1
        elif mask[i, j] == 0 and M[i, j, 0] == 0:
          left_pixels.append([i, j])
    
    faces, quad_pix = list(zip(*faces_pixels))
    for p in tqdm(left_pixels):
      # check which quadrilateral p fits in
      for i,quad in enumerate(quad_pix):
        point = Point(p)
        polygon = Polygon(quad)
        if polygon.within(point):
          M[p[0], p[1], 0] = 1
          M[p[0], p[1], 1:] = [item for sublist in faces[i] for item in sublist] 



    """

  
  def create_data(self, obj, environment=None):
    assert self.generated_data is not None, 'No data generated!'
    self.image_index = 0
    for data in self.generated_data:
      for ft_n, value in enumerate(data):
        feature = self.feature_names[ft_n]
        if '.' in feature:
          # is not a dict
          obj_name, atr = feature.split('.', 1)
          _setattr(self.objs[obj_name].obj, atr, value)
        elif ',' in feature:
          # is a dict
          obj_name, dict_, key, atr = feature.split(',')
          setattr(getattr(self.objs[obj_name].obj, dict_)[key], atr, value)
        else:
          setattr(self.objs[self.curr_obj], feature, value)

      index = str(1000000+self.image_index)[1:]
      # save image
      self.blender.context.scene.render.filepath = f'{self.destination_path+"/images"}/img{index}.png' 
      self.blender.ops.render.render(write_still=True)
      self.image_index += 1

      # create annotations
      co_3d_2d = self.create_annotations(obj, environment)

      if self.debug:
        co_2d = list(zip(*co_3d_2d))[1]
        visualize_vertices(co_2d, path=self.destination_path+f"/debug/d{index}.png")

      # save coordinates matches in npy format
      np.save(self.destination_path+f"/annotations/a{index}.npy", co_3d_2d)

  def create_random_sample(self, obj, environment=None):
    data = choice(self.generated_data)
    for ft_n, value in enumerate(data):
      feature = self.feature_names[ft_n]
      if '.' in feature:
        # is not a dict
        obj_name, atr = feature.split('.', 1)
        _setattr(self.objs[obj_name].obj, atr, value)
      elif ',' in feature:
        # is a dict
        obj_name, dict_, key, atr = feature.split(',')
        setattr(getattr(self.objs[obj_name].obj, dict_)[key], atr, value)
      else:
        setattr(self.objs[self.curr_obj], feature, value)

    return self.create_annotations(obj, environment)
