import bpy
#import bpy_extras
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import numpy as np
import matplotlib.pyplot as plt

def vert_coord_to_2d(points, camera, scene):
  ret = []
  for co in points:
    # convert the 3d coordinates to 2d
    if isinstance(co, tuple):
      co = Vector(co)
    co_2d = world_to_camera_view(scene, camera, co)
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


def get_vertices_coords(obj):
  return [Vector((v.x, v.y, v.z)) for v in [obj.matrix_world @ v.co for v in obj.data.vertices]]

def get_coordinates_matches(global_vertices, camera, scene):
  co_2d = vert_coord_to_2d(global_vertices, camera, scene)
  co_3d_2d = list(zip(global_vertices, co_2d))

  return co_3d_2d

def visualize_vertices(vertices, W=1920, H=1080, path='fig.png'):
  co_2d = np.array(vertices)
  plt.plot(co_2d[:, 0], co_2d[:, 1], 'bo')
  plt.ylim(0, H)
  plt.xlim(0, W)
  #plt.gca().invert_yaxis()
  plt.savefig(path)
  plt.clf()


def filter_non_visible_coords(co_3d_2d):
  # filter (0<=x<=1920, 0<=y<=1080)
  return [(tuple(co_3d), co_2d) for co_3d,co_2d in co_3d_2d if 0<=co_2d[0]<=1920 and 0<=co_2d[1]<=1080]

def filter_repeated_coords(co_3d_2d):
  aux = set()
  ret = []
  old_len = 0
  for co_3d,co_2d in co_3d_2d:
    aux.add(co_2d)
    if old_len < len(aux):
      ret.append((co_3d, co_2d))
      old_len += 1
  return ret
