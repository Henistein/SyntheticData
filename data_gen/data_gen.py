#import pandas as pd

from math import prod, radians
from random import shuffle, choice
from copy import deepcopy



def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

class Box:

    dim_x = 1
    dim_y = 1

    def __init__(self, min_x, min_y, max_x, max_y, dim_x=dim_x, dim_y=dim_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.dim_x = dim_x
        self.dim_y = dim_y

    @property
    def x(self):
        return round(self.min_x * self.dim_x)

    @property
    def y(self):
        return round(self.dim_y - self.max_y * self.dim_y)

    @property
    def width(self):
        return round((self.max_x - self.min_x) * self.dim_x)

    @property
    def height(self):
        return round((self.max_y - self.min_y) * self.dim_y)

    def __str__(self):
        return "<Box, x=%i, y=%i, width=%i, height=%i>" % \
               (self.x, self.y, self.width, self.height)

    def to_tuple(self):
        if self.width == 0 or self.height == 0:
            return (0, 0, 0, 0)
        return (self.x, self.y, self.width, self.height)


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.
    Negative 'z' value means the point is behind the camera.
    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.
    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    me = me_ob.to_mesh(preserve_all_data_layers=True)
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            #if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    # bpy.data.meshes.remove(me)

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    return Box(min_x, min_y, max_x, max_y, dim_x, dim_y)

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
    prod_list = ListProd([lst for obj in self.objs.values() for lst in obj.features.values()])
    return prod_list.request_data(ammount)


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