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

# Choose Cycles and GPU
#bpy.data.scenes[0].render.engine = "CYCLES"
#bpy.context.scene.cycles.device = "GPU"

# remove and light cube object
bpy.data.objects['Cube'].select_set(True)
bpy.data.objects['Light'].select_set(True)
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


# Day night cycle

# 1 - Change render engine to cycles and device set to GPU
bpy.data.scenes[0].render.engine = "CYCLES"
bpy.context.scene.cycles.device = "GPU"

# 2 - World properties, change world surface color to sky texture
sky_texture = bpy.context.scene.world.node_tree.nodes.new("ShaderNodeTexSky")
bg = bpy.context.scene.world.node_tree.nodes["Background"]
bpy.context.scene.world.node_tree.links.new(bg.inputs["Color"], sky_texture.outputs["Color"])

# 3 - Add a sun and set strength to 0
bpy.ops.object.light_add(type='SUN')
bpy.data.lights['Sun'].energy = 0

# 4 - World properties, Sun Position, set Sun Object to 'Sun' and Sky Texture
bpy.context.scene.sun_pos_properties.sun_object = bpy.data.objects["Sun"]
bpy.context.scene.sun_pos_properties.sky_texture = "Sky Texture"

# 5 - Change coordinates to become more realistic (optional)
bpy.context.scene.sun_pos_properties.co_parser = "38.691629133197814, -9.215922843598607"

# 6 - Change month and day of the year
bpy.context.scene.sun_pos_properties.month = 6
bpy.context.scene.sun_pos_properties.day = 20

# Reduce samples (faster rendering)
bpy.context.scene.cycles.samples = 1024




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
  # day night cycle
  dg.add_obj("scene", bpy.context.scene)
  dg.add_feature("sun_pos_properties.time", 5, 20, 1)


  dg.generate(200)
  dg.create_data()


  """
  # rotation
  dg.add_feature("rotation_euler.x", -180, 180, 10, radians)
  dg.add_feature("rotation_euler.y", -40, 40, 10, radians)
  dg.add_feature("rotation_euler.z", -70, 70, 10, radians)
  """