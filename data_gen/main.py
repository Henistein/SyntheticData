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

  # subdivie plane 8 times
  bpy.context.view_layer.objects.active = plane
  bpy.ops.object.editmode_toggle()
  for i in range(8):
    bpy.ops.mesh.subdivide()

  # flip normals
  bpy.ops.mesh.flip_normals()
  bpy.ops.object.editmode_toggle()

  # create and scale a new plane according to object distance (so it matches camera FOV)
  bpy.context.view_layer.objects.active = plane
  plane_loc = list(bpy.context.object.matrix_world.to_translation())
  print(plane_loc)
if 1 == 0:
  stop_loc = list(stop_sign.matrix_world.to_translation())
  d = dist(plane_loc, stop_loc)
  scale_x, scale_y = (d/10)*3.61, (d/10)*2.03

  bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD')
  new_plane = bpy.data.objects['Plane.001']
  x,y,z = tuple(plane.matrix_world.to_translation())
  print(x,y,z)
  new_plane.location.x = x
  print(new_plane.location)
  #new_plane.rotation_euler = plane.matrix_world.to_euler()

  # Shrinkwarp
  """
  bpy.context.view_layer.objects.active = plane
  bpy.ops.object.modifier_add(type='SHRINKWRAP')
  bpy.context.object.modifiers["Shrinkwrap"].wrap_method = 'PROJECT'
  bpy.context.object.modifiers["Shrinkwrap"].target = stop_sign
  """
  #bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

  # Boolean
  """
  bpy.ops.object.modifier_add(type='BOOLEAN')
  bpy.context.object.modifiers["Boolean"].operation = 'INTERSECT'
  bpy.context.object.modifiers["Boolean"].use_self = True
  bpy.context.object.modifiers["Boolean"].object = stop_sign
  bpy.ops.object.modifier_apply(modifier="Boolean")
  """


  
  # ------------------------------------------
  empty = bpy.data.objects['Empty']
  empty.constraints['Follow Path'].offset = 25
  bpy.context.view_layer.update()

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