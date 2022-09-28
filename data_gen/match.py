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
  """
  local_vertices = [v.co for v in obj.data.vertices]
  global_vertices = []

  for v in local_vertices:
    vec = [v[i]*obj.scale[i]+obj.location[i] for i in range(3)]
    global_vertices.append(Vector(vec))

  return global_vertices
  """


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

def filter_visible_vertices(vertices, cam, scene):
  # Threshold to test if ray cast corresponds to the original vertex
  limit = 0.1
  visible_vertices = []
  for i, v in tqdm(enumerate(vertices), total=len(vertices)):
    # Get the 2D projection of the vertex
    co2D = world_to_camera_view(scene, cam, v)

    bpy.ops.mesh.primitive_cube_add(location=(v))
    bpy.ops.transform.resize(value=(0.01, 0.01, 0.01))

    # If inside the camera view
    if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0 and co2D.z > 0:
      # Try a ray cast, in order to test the vertex visibility from the camera
      location = scene.ray_cast(
        bpy.context.window.view_layer, cam.location, (v - cam.location).normalized())
      # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
      if location[0] and (v - location[1]).length < limit:
        visible_vertices.append(obj.data.vertices[i])
  return visible_vertices


from math import radians

def func():
  scene = bpy.context.scene
  cam = scene.objects.get('Camera')
  cube = scene.objects.get("Cube")
  
  cam.rotation_euler = (radians(90), radians(0), radians(90))
  cam.location = (10,0,0)

  return scene, cam, cube


if __name__ == '__main__':
  scene, cam, cube = func()
  bpy.context.view_layer.update()
  cam = bpy.context.scene.camera

  matches = get_coordinates_matches(cube, camera=cam, scene=scene)
  vertices, pixels = list(zip(*matches))
  #visualize_vertices(pixels)


# Neste momento o objetivo e fazer o Ground truth para que consiga corresponder uma imagem sintetica com o objeto 3d
# Verificar a mesh do objeto para que consiga identificar corretamente cada regiao (por exemplo o objeto ser feito de cubinhos como um lego)
# Adaptar o YOLO/SSD para que consiga fazer o match entre uma imagem e o objeto 3d

# TODO hoje:
# try matching with stop_sing and fix bugs and optimize
# find a way to get smaller triangles or polygons

# make objectness
# Nerf