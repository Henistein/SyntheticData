#import pandas as pd
import os
import numpy as np
from match import filter_non_visible_coords, filter_repeated_coords, get_coordinates_matches, visualize_vertices
from scripts import cut_obj_camera_view

from math import prod, radians
from random import shuffle, choice
from copy import deepcopy

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
  
  def create_data(self, obj):
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
      # make the cut
      co_3d = cut_obj_camera_view(
        self.blender, 
        self.blender.data.objects['Plane'], 
        obj
      )
      co_3d_2d = get_coordinates_matches(
        co_3d,
        self.blender.data.objects['Camera'],
        self.blender.context.scene
      )
      co_3d_2d = filter_non_visible_coords(co_3d_2d)
      co_3d_2d = filter_repeated_coords(co_3d_2d)

      # save coordinates matches in npy format
      np.save(self.destination_path+f"/annotations/a{index}.npy", co_3d_2d)

      if self.debug: 
        co_2d = list(zip(*co_3d_2d))[1]
        print(co_2d[:10])
        visualize_vertices(co_2d, path=self.destination_path+f"/debug/d{index}.png")


  def create_random_sample(self):
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


"""
class Nop:
  pass

if __name__ == '__main__':
  dg = CreateData('data.csv')

  nop1 = Nop()
  nop2 = Nop()

  # add obj
  dg.add_obj('nop1', nop1)
  # rotation
  dg.add_feature("rotation.x", 90, 100, 2, radians)
  dg.add_feature("rotation.y", -180, 180, 36, radians)
  dg.add_feature("rotation.z", 80, 100, 5, radians)
  # location
  dg.add_feature("location.x", 5, 25, 4)
  dg.add_feature("location.y", -0.5, 0.5, 0.2)
  dg.add_feature("location.z", -0.5, 0.5, 0.2)

  dg.add_obj('nop2', nop2)
  dg.add_feature("speed.x", 5, 25, 4)
  dg.add_feature("speed.y", -0.5, 0.5, 0.2)
  dg.add_feature("speed.z", -0.5, 0.5, 0.2)

  dg.generate(1000)
  dg.create_data()
"""