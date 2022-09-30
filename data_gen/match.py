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

def get_coordinates_matches(obj, camera, scene):
  global_vertices = get_vertices_coords(obj)
  co_2d = vert_coord_to_2d(global_vertices, camera, scene)
  co_3d_2d = list(zip(global_vertices, co_2d))

  return co_3d_2d

def visualize_vertices(vertices, W=1920, H=1080):
  co_2d = np.array(vertices)
  plt.plot(co_2d[:, 0], co_2d[:, 1], 'bo')
  plt.ylim(0, H)
  plt.xlim(0, W)
  plt.gca().invert_yaxis()
  plt.savefig('fig.png')

from tqdm import tqdm

def filter_visible_matches(matches, obj, cam, ctx):
  # Threshold to test if ray cast corresponds to the original vertex
  limit = 0.1
  visible_vertices = []
  depsgraph = ctx.evaluated_depsgraph_get()
  scene = ctx.scene
  for i, m in tqdm(enumerate(matches), total=len(matches)):
    v,p = m
    # Get the 2D projection of the vertex
    co2D = world_to_camera_view(scene, cam, v)

    bpy.ops.mesh.primitive_cube_add(location=(v))
    bpy.context.active_object.name = 'AuxCube'
    bpy.ops.transform.resize(value=(0.01, 0.01, 0.01))

    # If inside the camera view
    if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0 and co2D.z > 0:
      # Try a ray cast, in order to test the vertex visibility from the camera
      location = scene.ray_cast(
        depsgraph, cam.location, (v - cam.location).normalized())
      # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
      if location[0] and (v - location[1]).length < limit:
        visible_vertices.append((obj.data.vertices[i], p))

    bpy.data.objects['AuxCube'].select_set(True)
    bpy.ops.object.delete()

  return visible_vertices


from math import radians

def func():
  scene = bpy.context.scene
  cam = scene.objects.get('Camera')
  cube = scene.objects.get("Cube")
  
  #cam.rotation_euler = (radians(90), radians(0), radians(90))
  #cam.location = (10,0,0)
  cam.rotation_euler = (radians(53), radians(0), radians(46))
  cam.location = (4, -4, 3.1)

  return scene, cam, cube


if __name__ == '__main__':
  scene, cam, cube = func()
  bpy.context.view_layer.update()
  cam = bpy.context.scene.camera

  matches = get_coordinates_matches(cube, camera=cam, scene=scene)
  vertices, pixels = list(zip(*matches))

  visible_vertices = filter_visible_matches(matches, cube, cam, bpy.context)
  vertices, pixels = list(zip(*visible_vertices))

  """
  edges = cube.edges
  vertices_indexes = [i.index for i in vertices]
  for e in edges:
    verts = [v for v in e.vertices if v in vertices_indexes]
    assert len(verts) == 1 or len(verts) == 2, "The Edge must have at least a vertice inside the camera view"
    if len(verts) == 1: # the edge is not totally inside the camera view
      # clip the edge and create a vertice that is inside the camera view
  """




    





  #print(vertices)
  #visualize_vertices(pixels)

  #vertices = [v.co for v in vertices]
  #vertices = vert_coord_to_2d(visible_vertices, cam, scene)
  #visible_vertices = [v.co for v in visible_vertices]
  #visible_vertices = vert_coord_to_2d(visible_vertices, cam, scene)

# TODO: 
# https://i.imgur.com/bgl73j7.png
# so if you create a plane object and align it to the camera view.
# you could then parent it to the camera, in case you want to more easily move the camera around
# - colocar o plano na perspetiva da camera a olho e depois fazer o parenting dos dois

# then select the master object, go into edit mode, then ctrl+click on the cutting object (plane)
# then from the camera PoV viewport mesh->knife project
# this obviously would be translated to work in Python
# but should be quite straightforward
