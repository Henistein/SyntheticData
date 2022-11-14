import bpy

import importlib
import os
import sys
import glob
import yaml
import pickle

blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
   sys.path.append(blend_dir)

import scripts
from scripts import *
import data_gen
importlib.reload(scripts)
importlib.reload(data_gen)

from math import radians

def init():
  bpy.data.scenes[0].render.engine = "CYCLES"
  bpy.context.scene.cycles.device = "GPU"

  # remove cube object
  bpy.data.objects['Cube'].select_set(True)
  bpy.ops.object.delete()

  # create camera plane
  plane = create_camera_plane()

  # import sign object
  bpy.ops.import_scene.obj(filepath=f"models/{conf['NAME']}.obj")
  obj = bpy.data.objects[conf['OBJ_NAME']]

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

if __name__ == '__main__':
  conf = yaml.safe_load(open('conf.yaml'))
  path = conf["PATH"]

  argv = sys.argv
  argv = argv[argv.index("--") + 1:] 

  # -------------------------------------------
  # init blender world configs
  obj, node_environment = init()

  if argv[0] == "load_generated_data":
    # load generated data file
    with open(argv[1], 'rb') as f:
      generated_data = pickle.load(f)
    
    offsets = (int(argv[2]), int(argv[3]))
    generated_data = generated_data[offsets[0]:offsets[1]]

    dg = data_gen.CreateData(bpy, res=(256, 256), redux_factor=1, destination_path=path, debug=True, generated_data=generated_data)
    dg.add_obj('empty', bpy.data.objects['Empty'])
    dg.add_obj('obj', obj)
    dg.add_obj('node_environment', node_environment)
    dg.feat_name = ['empty,constraints,Follow Path,offset', 'empty,constraints,Follow Path,influence', 'obj.location.x', 'obj.location.y', 'obj.location.z', 'obj.rotation_euler.x', 'obj.rotation_euler.y', 'obj.rotation_euler.z', 'node_environment.image']
    dg.image_index = offsets[0]
    dg.create_data(obj, debug=True, already_gen=True)

  else:
    dg = data_gen.CreateData(bpy, res=(256, 256), redux_factor=1, destination_path=path, debug=True, generated_data=None)


    # add empty obj (camera)
    dg.add_obj('empty', bpy.data.objects['Empty'])
    # offset
    dg.add_feature("constraints,Follow Path,offset", 4, 46, 3)
    # influence
    dg.add_feature("constraints,Follow Path,influence", 0.25, 1.0, 0.05)

    # add obj
    dg.add_obj('obj', obj)
    # location
    dg.add_feature("location.x", -10, 0, 2)
    dg.add_feature("location.y", -10, 10, 4)
    dg.add_feature("location.z", -5, 5, 2)

    # rotation
    dg.add_feature("rotation_euler.x", 90, 100, 2, radians)
    dg.add_feature("rotation_euler.y", -180, 180, 36, radians)
    dg.add_feature("rotation_euler.z", 80, 100, 5, radians)

    # add background object
    dg.add_obj('node_environment', node_environment)
    dg.add_elements('image', list(glob.glob('backgrounds/*')))

    # Generate and create data
    dg.generate(33000)

    dg.save_generated_data('pkls/airplane_data_gen.pkl')
    #dg.create_data(obj, debug=True)
    #dg.create_random_sample(obj, debug=True)

    #annotations = dg.create_random_sample(obj, MAP=MAP)
    #print(annotations.shape)
