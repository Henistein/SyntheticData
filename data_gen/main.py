import bpy
from math import radians

import sys
import os

blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
   sys.path.append(blend_dir)

import data_gen
import importlib
importlib.reload(data_gen)


# remove cube object
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

# import sign object
bpy.ops.import_scene.obj(filepath="models/stop_sign.obj")

# objects
stop_sign = bpy.data.objects['Stop']
camera = bpy.data.objects['Camera']

# reset objects position
stop_sign.location.x = 0
camera.rotation_euler.x = radians(0)
camera.rotation_euler.y = radians(0)
camera.rotation_euler.z = radians(0)
camera.location.x = 0
camera.location.y = 0
camera.location.z = 0

# add a curve (circle)
bpy.ops.curve.primitive_bezier_circle_add(radius=10, location=(0, 0, 0))
# add an empty cube
bpy.ops.object.empty_add(type='CUBE')

# make camera track sign
camera_constraint = camera.constraints.new(type='TRACK_TO')
camera_constraint.target = stop_sign

# select camera and empty
bpy.data.objects['Empty'].select_set(True)
bpy.data.objects['Camera'].select_set(True)

# link camera to empty
bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)

# make empty follow path the bezier curve
bpy.ops.object.constraint_add(type='FOLLOW_PATH')
bpy.data.objects['Empty'].constraints['Follow Path'].target = bpy.data.objects['BezierCircle']


if 1 == 1:
  path = '/media/henistein/Novo volume/SyntheticData/teste'

  dg = data_gen.CreateData(bpy, path)

  # add empty obj
  dg.add_obj('empty', bpy.data.objects['Empty'])
  # offset
  dg.add_feature("constraints,Follow Path,offset", 4, 46, 3)
  # influence
  dg.add_feature("constraints,Follow Path,influence", 0.25, 1.0, 0.05)
  # add stop obj
  dg.add_obj('stop_sign', stop_sign)
  # location
  dg.add_feature("location.x", -40, 0, 4)
  dg.add_feature("location.y", -20, 20, 4)
  dg.add_feature("location.z", -10, 10, 2)

  dg.generate(200)
  dg.create_data()


  """
  # rotation
  dg.add_feature("rotation_euler.x", -180, 180, 10, radians)
  dg.add_feature("rotation_euler.y", -40, 40, 10, radians)
  dg.add_feature("rotation_euler.z", -70, 70, 10, radians)
  """