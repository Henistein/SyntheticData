# Fazer a correspondencia da imagem para recolhida pela camera e o modelo 3D
# Ativar os vertices que foram captados pela imagem recolhida pela camera

# Camera culling
# https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex
# https://www.youtube.com/watch?v=PEosrzAHYKM


# Algo:
# Obter lista de vertices e a sua correspondencia em pixels
# Filtrar apenas os que aparecem na camera view

# Este codigo da-me as coordenadas em 2d dos vertices 3d em relacao a camera:
# Test the function using the active object (which must be a camera)
# and the 3D cursor as the location to find.
import bpy
import bpy_extras

def vert_coord_to_2d(points):
  scene = bpy.context.scene
  obj = bpy.context.object # must be the camera
  ret = []
  for co in points:
    # convert the 3d coordinates to 2d
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, obj, co)
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

scene = C.scene
cube = scene.objects.get("Cube")
vertes = [v.co for v in cube.data.vertices]