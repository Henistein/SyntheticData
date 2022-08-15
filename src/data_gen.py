import pandas as pd
from math import prod
from random import shuffle, choice
from copy import deepcopy

class DataGen:
  def __init__(self):
    self.features = {}
  
  @property
  def total_features(self): return len(self.features)

  @property
  def all_combinations(self): return prod([len(x) for x in self.features.values()])

  def add_feature(self, atr, frm, to, step=1):
    self.features[atr] = self.calculate_feature_list((frm, to, step))
  
  def remove_feature(self, atr): del self.features[atr]
  
  def calculate_feature_list(self, frm_to_step):
    _, _, step = frm_to_step 
    decimals = 1
    while int(str(step*decimals).split('.')[0]) == 0: decimals *= 10
    lst = list(range(*tuple(map(lambda x: int(x*decimals), frm_to_step))))
    shuffle(lst)
    return lst
  
  def generate(self, ammount):
    assert ammount <= self.all_combinations, 'Not enough combinations!'

    s = set()
    while len(s) < ammount:
      s.add(tuple([choice(lst) for lst in self.features.values()])) # not the most efficient way, but it works
    return s

  



if __name__ == '__main__':
  dg = DataGen()

  # rotation
  dg.add_feature("rotation.x", 90, 100, 2)
  dg.add_feature("rotation.y", -180, 180, 36)
  dg.add_feature("rotation.z", 80, 100, 5)
  # location
  dg.add_feature("location.x", 5, 25, 4)
  dg.add_feature("location.y", -0.5, 0.5, 0.2)
  dg.add_feature("location.z", -0.5, 0.5, 0.2)

  print(dg.generate(25000))