import bpy
import sys
import os
import pickle
import numpy as np

sys.path.insert(1, '/home/socialab/Henrique/SyntheticData/data_gen')

blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
  sys.path.append(blend_dir)

import scripts
from scripts import *
import data_gen
#importlib.reload(scripts)
#importlib.reload(data_gen)

from math import radians

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



# parse args
argv = " ".join(sys.argv).replace(" -- ", "++").split("++")[1:]
conf = {s.split(" ")[0]:s.split(" ")[1:] for s in argv}

obj_name,img_path,mesh_path = conf['args']

# load data
with open('../data_gen/pkls/data_10.pkl', 'rb') as f:
  data = pickle.load(f)

for i in range(len(data)):
  data[i][-1] = "../data_gen/" + data[i][-1]
  print(data[i])

# load sample from data
sample = data[1]

# import object
obj, node_environment = init(obj_name)

dg = data_gen.CreateData(bpy, res=(256, 256), redux_factor=1, destination_path=None, debug=False, generated_data=[sample])
dg.add_obj('empty', bpy.data.objects['Empty'])
dg.add_obj('obj', obj)
dg.add_obj('node_environment', node_environment)
dg.feat_name = ['empty,constraints,Follow Path,offset', 'empty,constraints,Follow Path,influence', 'obj.location.x', 'obj.location.y', 'obj.location.z', 'obj.rotation_euler.x', 'obj.rotation_euler.y', 'obj.rotation_euler.z', 'node_environment.image']
dg.create_random_sample(obj, debug=False, already_gen=True)


print(sample)
