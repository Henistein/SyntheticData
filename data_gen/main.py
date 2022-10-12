import bpy

import importlib
import os
import sys

blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
   sys.path.append(blend_dir)

import scripts
from scripts import *
import data_gen
importlib.reload(scripts)
importlib.reload(data_gen)

from math import radians, dist

if __name__ == '__main__':
  # remove cube object
  bpy.data.objects['Cube'].select_set(True)
  bpy.ops.object.delete()

  # create camera plane
  plane = create_camera_plane()

  # import sign object
  bpy.ops.import_scene.obj(filepath="models/stop_sign.obj")
  stop_sign = bpy.data.objects['Stop']

  # reset objects position
  stop_sign.location.x = 0
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
  camera_constraint.target = stop_sign

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
  create_background()
  
  # -------------------------------------------
  """
  empty = bpy.data.objects['Empty']
  empty.constraints['Follow Path'].offset = 25
  """
  # -------------------------------------------
  path = '/home/socialab/Henrique/DATA'

  dg = data_gen.CreateData(bpy, path)

  # add empty obj (camera)
  dg.add_obj('empty', bpy.data.objects['Empty'])
  # offset
  dg.add_feature("constraints,Follow Path,offset", 4, 46, 3)
  # influence
  dg.add_feature("constraints,Follow Path,influence", 0.25, 1.0, 0.05)

  """
  # add stop obj
  dg.add_obj('stop_sign', stop_sign)
  # location
  dg.add_feature("location.x", -40, 0, 4)
  dg.add_feature("location.y", -20, 20, 4)
  dg.add_feature("location.z", -10, 10, 2)
  """

  dg.generate(10)
  dg.create_random_sample()

  # make the cut
  cut_obj_camera_view(bpy, plane, stop_sign)

  # TODO:
  # Arranjar a projection
  # Ver se da match com a camera
  # Arranjar a formula

  # Fazer o script para tornar isto automatico
  # - Subdivisoes nas edges
  # - Dar scale ao plane de acordo com a distancia ao objecto (saber como calcular distancia)
  # - Shrinkwarp
  # - Boolean

  # Ter em conta as otimizacoes