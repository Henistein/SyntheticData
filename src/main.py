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

bpy.ops.import_scene.obj(filepath="models/stop_sign.obj")

stop = bpy.data.objects['Sign']
camera = bpy.data.objects['Camera']

# rotation.x = [90, 100]     -> 5  levels (2)
# rotation.y = [-180, 180]   -> 10 levels (36)
# rotation.z = [80, 100]     -> 4  levels (5)

# location.x = [5, 25]       -> 5  levels (4)
# location.y = [-0.5, 0.5]   -> 5  levels (2)
# location.z = [-0.5, 0.5]   -> 5  levels (2)
# -------------------------------------------
#                            -> 25000


path = '/media/henistein/Novo volume/SyntheticData'

dg = data_gen.CreateData(bpy, path)

# add obj
dg.add_obj('camera', camera)
# rotation
dg.add_feature("rotation_euler.x", 90, 100, 2, radians)
dg.add_feature("rotation_euler.y", -180, 180, 36, radians)
dg.add_feature("rotation_euler.z", 80, 100, 5, radians)
# location
dg.add_feature("location.x", 5, 25, 4)
dg.add_feature("location.y", -0.5, 0.5, 0.2)
dg.add_feature("location.z", -0.5, 0.5, 0.2)

dg.generate(1000)
dg.create_data()


"""
i = 0
for rx in range(90, 100, 2):
  for ry in range(-180, 180, 36):
    for rz in range(80, 100, 5):
      for lx in range(5, 25, 4):
        for ly in [x/10.0 for x in range(-5, 6, 2)]:
          for lz in [x/10.0 for x in range(-5, 6, 2)]:
            # change camera position
            camera.location.x = lx
            camera.location.y = ly
            camera.location.z = lz
            camera.rotation_euler[0] = math.radians(rx)
            camera.rotation_euler[1] = math.radians(ry)
            camera.rotation_euler[2] = math.radians(rz)
            # save image
            bpy.context.scene.render.filepath = f'{path}/img{str(1000000+i)[1:]}.png' 
            bpy.ops.render.render(write_still=True)
            if i % 1000 == 0:
              print(f'{i}/25000')
            i += 1
"""