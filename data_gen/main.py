# Perspective according to a mass center:
# - object position in N levels (maybe 10)
# - camera position in N levels (maybe 10)

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

bpy.ops.import_scene.obj(filepath="models/yield_sign.obj")

# objects
stop_sign = bpy.data.objects['Yield']
camera = bpy.data.objects['Camera']


stop_sign.location.x = 0
camera.rotation_euler.x = radians(90)
camera.rotation_euler.y = radians(0)
camera.rotation_euler.z = radians(90)

camera.location.x = 10
camera.location.y = 0
camera.location.z = 0

#b = data_gen.camera_view_bounds_2d(bpy.context.scene, bpy.context.scene.camera, stop_sign)

if 1 == 1:
  path = '/media/henistein/Novo volume/SyntheticData/yield_sign'

  dg = data_gen.CreateData(bpy, path)

  # add obj
  dg.add_obj('stop_sign', stop_sign)
  # rotation
  dg.add_feature("rotation_euler.x", -180, 180, 10, radians)
  dg.add_feature("rotation_euler.y", -40, 40, 10, radians)
  dg.add_feature("rotation_euler.z", -70, 70, 10, radians)
  # location
  """
  dg.add_feature("location.x", -37, 7, 2)
  dg.add_feature("location.y", -3.5, 3.5, 0.7)
  dg.add_feature("location.z", -1.7, 1.7, 0.34)
  """

  dg.generate(1000)
  dg.create_data()