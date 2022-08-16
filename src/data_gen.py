#import pandas as pd

from math import prod, radians
from random import shuffle, choice
from copy import deepcopy

class ObjFeatures:
  def __init__(self, obj):
    self.obj = obj
    self.features = {}

class DataGen:
  def __init__(self):
    self.objs = {}
    self.curr_obj = None
  
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
    self.objs[self.curr_obj].features[self.curr_obj+'.'+atr] = self.calculate_feature_list((frm, to, step), callback)
  
  def calculate_feature_list(self, frm_to_step, callback):
    _, _, step = frm_to_step 
    decimals = 1
    while int(str(step*decimals).split('.')[0]) == 0: decimals *= 10
    lst = list(range(*tuple(map(lambda x: int(x*decimals), frm_to_step))))
    lst = list(map(callback, lst))
    shuffle(lst)
    return lst
  
  def generate(self, ammount):
    assert ammount <= self.all_combinations, 'Not enough combinations!'

    s = set()
    while len(s) < ammount:
      s.add(tuple([choice(lst) for obj in self.objs.values() for lst in obj.features.values()])) # not the most efficient way, but it works
    return s


def _setattr(obj, attr_s, val):
  parts = attr_s.split('.')
  for part in parts[:-1]:
    obj = getattr(obj, part)
  setattr(obj, parts[-1], val)



class CreateData(DataGen):
  def __init__(self, blender, destination_path):
    super().__init__()
    self.blender = blender
    self.destination_path = destination_path
  
  def generate(self, ammount):
    self.generated_data = super().generate(ammount)
  
  def create_images(self):
    i = 0
    for data in self.generated_data:
      for ft_n, value in enumerate(data):
        obj_name, atr = self.feature_names[ft_n].split('.', 1)
        _setattr(self.objs[obj_name].obj, atr, value)
        # save image
        self.blender.context.scene.render.filepath = f'{self.destination_path}/img{str(1000000+i)[1:]}.png' 
        self.blender.ops.render.render(write_still=True)
        if i % 1000 == 0:
          print(f'{i}/25000')
        i += 1

  def create_csv(self):
    data = pd.DataFrame(self.generated_data, columns=self.feature_names)
    data.to_csv(self.destination_path, index=False)

  def create_data(self):
    assert self.generated_data is not None, 'No data generated!'
    #self.create_csv()
    self.create_images()


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