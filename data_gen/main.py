from distutils.log import debug
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
from match import filter_non_visible_coords, filter_repeated_coords, get_coordinates_matches, visualize_vertices
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
  path = '/home/socialab/Henrique/DATA/stop_sign'

  dg = data_gen.CreateData(bpy, path, debug=True)

  # add empty obj (camera)
  dg.add_obj('empty', bpy.data.objects['Empty'])
  # offset
  dg.add_feature("constraints,Follow Path,offset", 4, 46, 3)
  # influence
  dg.add_feature("constraints,Follow Path,influence", 0.25, 1.0, 0.05)

  # Generate and create data
  dg.generate(10)
  dg.create_data(stop_sign)


  """
  # add stop obj
  dg.add_obj('stop_sign', stop_sign)
  # location
  dg.add_feature("location.x", -40, 0, 4)
  dg.add_feature("location.y", -20, 20, 4)
  dg.add_feature("location.z", -10, 10, 2)
  """


  """
  # Sample
  dg.create_random_sample()
  obj_3d_mask = cut_obj_camera_view(bpy, plane, stop_sign)

  co_3d_2d = get_coordinates_matches(obj_3d_mask, bpy.data.objects['Camera'], bpy.context.scene)
  co_3d_2d = filter_non_visible_coords(co_3d_2d)
  co_3d_2d = filter_repeated_coords(co_3d_2d)

  co_3d, co_2d = list(zip(*co_3d_2d))
  co_3d = list(map(tuple, co_3d))

  visualize_vertices(co_2d)
  """





  # TODO:
  # Fazer experiencias com a GPU
  # Criar mecanismo de debug na malha atraves dos vertices
  # Arranjar mais malhas
  # Por os backgrounds a dar


  # Perceber como vamos tratar o limite de vertices
  # Estudar Siamese Neural Networks

  # further TODO:
  # Resolver bug em que quando o objeto fica demasiado longe o shrinkwarp funciona mas o boolean nao
  # Ter em conta as otimizacoes