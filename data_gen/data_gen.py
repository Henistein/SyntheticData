#import pandas as pd
import os
import numpy as np
import pickle
from match import filter_non_visible_coords, filter_repeated_coords, get_coordinates_matches, visualize_vertices
from scripts import cut_obj_camera_view, get_polygon_indexes, get_visible_mesh, create_mask

from math import prod, radians
import random
from random import shuffle, choice
random.seed(422)
from PIL import Image
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
  def __init__(self, blender, res=(1920, 1080), redux_factor=10, destination_path=None, debug=False):
    super().__init__()
    self.blender = blender
    self.destination_path = destination_path
    self.debug = debug
    self.res = res # (W, H)
    self.redux_factor = redux_factor
    # create image, annotations and maybe debug folders
    if not os.path.exists(destination_path): os.mkdir(destination_path)
    if not os.path.exists(destination_path+"/images"): os.mkdir(destination_path+"/images")
    if not os.path.exists(destination_path+"/annotations"): os.mkdir(destination_path+"/annotations")
    if debug: 
      if not os.path.exists(destination_path+"/debug"): os.mkdir(destination_path+"/debug")
  
  def generate(self, ammount):
    self.generated_data = super().generate(ammount)
  
  def create_annotations(self, obj, MAP, TREE, LST, output_img=False):
    # visible mesh
    new_plane = cut_obj_camera_view(self.blender, self.blender.data.objects['Plane'], obj)
    visible_faces = get_visible_mesh(self.blender, new_plane, obj, visualize=True)

    faces_pixels = get_coordinates_matches(
      visible_faces,
      self.blender.data.objects['Camera'],
      self.blender.context.scene
    )
    
    # create matrix
    M = np.zeros((self.res[1], self.res[0]), dtype=np.int64)
    C = np.zeros((self.res[1], self.res[0], 3))
    F = {}
    inc = 1
    for face,co_2d in faces_pixels:
      #co_2d = [tuple(obj.matrix_world @ Vector(co)) for co in co_2d]
      indices = get_polygon_indexes(co_2d)
      color = np.random.randint(0, 255, size=(3,)).tolist()
      for i,j in indices:
        if not ((0 <= i < self.res[0]) and (0 <= j < self.res[1])):
          print(f'Skiped index ({i},{j}')
          continue

        j = abs(self.res[1]-1-j)
        face_coords = [item for sublist in face for item in sublist] 
        F[inc] = {"face":face_coords, "color":color}

        M[j, i] = inc

        C[j, i] = color
      inc += 1
    
    NEW_M = np.zeros((self.res[1]//self.redux_factor, self.res[0]//self.redux_factor, 4))
    NEW_C = np.zeros((self.res[1]//self.redux_factor, self.res[0]//self.redux_factor, 3))
    # perform the reduction to 10 times
    for i in range(0, self.res[1], self.redux_factor):
      for j in range(0, self.res[0], self.redux_factor):
        most_freq = np.bincount(M[i:i+self.redux_factor, j:j+self.redux_factor].flatten()).argmax()
        if most_freq == 0:
          continue
        face = F[most_freq]["face"]
        # WORLD_MATRIX * VERTICE = VTRANSFORMADO
        # VERTICE = np.linalg.solve(WORLD_MATRIX, VTRANSFORMADO)
        face = np.stack([np.linalg.solve(obj.matrix_world, face[i:i+3]+[1])[:-1] for i in range(0, len(face), 3)]).flatten()
        face = tuple(round(v, 3) for v in face)
        try:
          center = list(MAP[face])
        except KeyError:
          # if fails, we search in the tree for the closest vertice value
          new_key = []
          for v in range(0, 12, 3):
            _, ind = TREE.query([face[v:v+3]], k=1)
            for item in LST[ind.item()]:
              new_key.append(round(item, 3))
          center = list(MAP[tuple(new_key)])

        NEW_M[i//self.redux_factor, j//self.redux_factor] = [1] + center
        NEW_C[i//self.redux_factor, j//self.redux_factor] = F[most_freq]["color"]

    if output_img:
      big_img = Image.fromarray(np.uint8(C))
      big_img.save('big_matrix.PNG')

      little_img = Image.fromarray(np.uint8(NEW_C))
      little_img.save('little_matrix.PNG')

      return NEW_M, big_img, little_img

    return (NEW_M,None,None)

  def create_data(self, obj, MAP, debug=False):
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
      self.blender.context.scene.render.resolution_x = self.res[0]
      self.blender.context.scene.render.resolution_y = self.res[1]
      self.blender.context.scene.render.filepath = f'{self.destination_path+"/images"}/img{index}.png' 
      self.blender.ops.render.render(write_still=True)
      self.image_index += 1

      # create annotations
      annotations, big_img, little_img = self.create_annotations(obj, MAP, output_img=debug)

      if debug:
        big_img.save(self.destination_path+f"/debug/big_{index}.png")
        little_img.save(self.destination_path+f"/debug/little_{index}.png")

      # save coordinates matches in npy format
      np.save(self.destination_path+f"/annotations/a{index}.npy", annotations)

  def create_random_sample(self, obj, MAP=None, TREE=None, LST=None, debug=False):
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
    self.blender.context.scene.render.resolution_x = self.res[0]
    self.blender.context.scene.render.resolution_y = self.res[1]

    return self.create_annotations(obj, MAP, TREE, LST, output_img=debug)
