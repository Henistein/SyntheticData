# Fazer a correspondencia da imagem para recolhida pela camera e o modelo 3D
# Ativar os vertices que foram captados pela imagem recolhida pela camera

# Camera culling
# https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex
# https://www.youtube.com/watch?v=PEosrzAHYKM

from ctypes import pointer
import bpy
import bpy_extras
from mathutils import Vector
import numpy as np
import matplotlib.pyplot as plt

def vert_coord_to_2d(points):
  scene = bpy.context.scene
  camera = bpy.data.objects['Camera']
  ret = []
  for co in points:
    # convert the 3d coordinates to 2d
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, co)
    # convert the 2d coordinates to pixel coordinates
    ret.append(_2d_to_pixel_coords(scene, co_2d))
  return ret

def _2d_to_pixel_coords(scene, co_2d):
  render_scale = scene.render.resolution_percentage / 100
  render_size = (
      int(scene.render.resolution_x * render_scale),
      int(scene.render.resolution_y * render_scale),
  )
  return (round(co_2d.x * render_size[0]), round(co_2d.y * render_size[1]))

#scene = bpy.context.scene
#cube = scene.objects.get("Cube")
#vertes = [v.co for v in cube.data.vertices]
#print(vert_coord_to_2d(vertes))
#exit(0)

# TODO hoje:
# ((pixel coordinates), (3d coordinates))
# find a way to just get the list of vertices that are being captured by the camera


def get_vertices_coords(obj):
  local_vertices = [v.co for v in obj.data.vertices]
  global_vertices = []

  for v in local_vertices:
    vec = [v[i]*obj.scale[i]+obj.location[i] for i in range(3)]
    global_vertices.append(Vector(vec))

  return global_vertices


def get_coordinates_matches(obj):
  global_vertices = get_vertices_coords(obj)

  co_2d = vert_coord_to_2d(global_vertices)
  co_3d_2d = list(zip(global_vertices, co_2d))

  """
  for v in global_vertices:
    co_2d = vert_coord_to_2d(v)
    co_3d_2d.append((v, co_2d))
  """
  
  return co_3d_2d

def visualize(matches):
  co_2d = [i for v,i in matches]
  co_2d = np.array(co_2d)
  plt.plot(co_2d[:, 0], co_2d[:, 1], 'bo')
  plt.ylim(0, 1080)
  plt.xlim(0, 1920)
  plt.gca().invert_yaxis()
  plt.show()

scene = bpy.context.scene
cube = scene.objects.get("Cube")

matches = get_coordinates_matches(cube)
# visualize(matches)


# print(get_coordinates_matches(cube))

# Neste momento o objetivo e fazer o Ground truth para que consiga corresponder uma imagem sintetica com o objeto 3d
# Verificar a mesh do objeto para que consiga identificar corretamente cada regiao (por exemplo o objeto ser feito de cubinhos como um lego)
# Adaptar o YOLO/SSD para que consiga fazer o match entre uma imagem e o objeto 3d