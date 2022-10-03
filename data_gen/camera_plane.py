import bpy
import importlib

import os
import sys

blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
   sys.path.append(blend_dir)

import scripts
from scripts import *
importlib.reload(scripts)


if __name__ == '__main__':
  # remove and light cube object
  bpy.data.objects['Cube'].select_set(False)
  bpy.data.objects['Light'].select_set(True)
  bpy.ops.object.delete()

  # objects
  cube = bpy.data.objects['Cube']

  plane = create_camera_plane()
  plane.location = (-2.8, 0, 5)
  bpy.context.view_layer.update()
  cut_obj_camera_view(cube, plane)
