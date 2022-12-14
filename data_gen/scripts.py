import bpy
import numpy as np
from math import dist
from match import get_vertices_coords
from mathutils.bvhtree import BVHTree
from PIL import Image
import mahotas
#import cv2

def create_camera_plane():
  # objects
  camera = bpy.data.objects['Camera']

  # reset rotation and location
  bpy.data.objects['Camera'].select_set(True)
  bpy.ops.object.rotation_clear(clear_delta=False)
  bpy.ops.object.location_clear(clear_delta=False)
  bpy.data.objects['Camera'].select_set(False)

  # add a plane
  bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
  plane = bpy.data.objects['Plane']

  # set camera location to match plane
  camera.location = (0.0, 0.0, 4.92)
  # set plane scale to match camera view
  plane.scale = (1.775, 1.0, 1.0)

  # parenting plane to camera
  camera.select_set(True)
  plane.select_set(True)
  bpy.ops.object.parent_set(type='OBJECT', keep_transform=False)

  # make plane transparent
  plane.select_set(True)
  plane.active_material = bpy.data.materials.new("Transparent")
  bpy.data.materials["Transparent"].use_nodes = True
  bpy.data.materials["Transparent"].node_tree.nodes["Principled BSDF"].inputs[21].default_value = 0
  plane.active_material.blend_method = 'BLEND'

  return plane

def cut_obj_camera_view(bpy, plane, obj):
  bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD')
  new_plane = bpy.data.objects['Plane.001']
  new_plane.location = plane.matrix_world.to_translation()
  new_plane.rotation_euler = plane.matrix_world.to_euler()
  bpy.context.view_layer.objects.active = new_plane

  #bpy.context.view_layer.objects.active = plane
  bpy.ops.object.editmode_toggle()
  bpy.ops.mesh.flip_normals()

  # subdivie plane 8 times
  for _ in range(8):
    bpy.ops.mesh.subdivide()
  bpy.ops.object.editmode_toggle()

  # create and scale a new plane according to object distance (so it matches camera FOV)
  plane_loc = list(plane.matrix_world.to_translation())
  stop_loc = list(obj.matrix_world.to_translation())
  d = dist(plane_loc, stop_loc)
  scale_x, scale_y = (d/10)*3.61, (d/10)*2.03
  new_plane.scale = (scale_x, scale_y, 1)

  # Shrinkwarp
  bpy.context.view_layer.objects.active = new_plane
  bpy.ops.object.modifier_add(type='SHRINKWRAP')
  bpy.context.object.modifiers["Shrinkwrap"].wrap_method = 'PROJECT'
  bpy.context.object.modifiers["Shrinkwrap"].target = obj
  bpy.ops.object.modifier_apply(modifier="Shrinkwrap")


  return new_plane

def get_visible_mesh(bpy, plane, obj, visualize=False):
  m_1 = plane.matrix_world.copy()
  mesh_1_verts = [m_1 @ vertex.co for vertex in plane.data.vertices]
  mesh_1_polys = [polygon.vertices for polygon in plane.data.polygons]
  
  m_2 = obj.matrix_world.copy()
  mesh_2_verts = [m_2 @ vertex.co for vertex in obj.data.vertices] 
  mesh_2_polys = [polygon.vertices for polygon in obj.data.polygons]

  mesh_1_bvh_tree = BVHTree.FromPolygons(mesh_1_verts, mesh_1_polys)
  mesh_2_bvh_tree = BVHTree.FromPolygons(mesh_2_verts, mesh_2_polys)

  intersections = mesh_1_bvh_tree.overlap(mesh_2_bvh_tree)

  mesh_2_polys_ints = [pair[1] for pair in intersections]

  # get the visible faces
  if visualize:
    bpy.context.selected_objects[0].select_set(False)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.editmode_toggle()

  # vertices
  """
  ret = set()
  for face in obj.data.polygons:
    if face.index in mesh_2_polys_ints:
      for v in face.vertices:
        if visualize: obj.data.vertices[v].select = True
        ret.add(tuple(obj.matrix_world @ obj.data.vertices[v].co))
  """

  # faces
  ret = []
  for face in obj.data.polygons:
    if face.index in mesh_2_polys_ints:
      aux = []
      for v in face.vertices:
        if visualize: obj.data.vertices[v].select = True
        aux.append(tuple(obj.matrix_world @ obj.data.vertices[v].co))
      ret.append(aux)
  # remove plane
  #bpy.context.view_layer.objects.active = plane
  bpy.data.objects['Plane.001'].select_set(True)
  bpy.ops.object.delete()

  return ret


def create_background():
  # background
  C = bpy.context
  scn = C.scene
  # Get the environment node tree of the current scene
  node_tree = scn.world.node_tree
  tree_nodes = node_tree.nodes
  # Clear all nodes
  tree_nodes.clear()
  # Add Background node
  node_background = tree_nodes.new(type='ShaderNodeBackground')
  # Add Environment Texture node
  node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
  # Load and assign the image to the node property
  #node_environment.image = bpy.data.images.load('backgrounds/solitude_night_4k.exr')
  node_environment.location = -300,0
  # Add Output node
  node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
  node_output.location = 200,0
  # Link all nodes
  links = node_tree.links
  links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
  links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

  return node_environment

def create_mask(node_environment):
  mat = "None"
  old_engine = bpy.data.scenes[0].render.engine
  old_value = bpy.data.materials[mat].node_tree.nodes["Principled BSDF"].inputs[21].default_value
  old_img = node_environment.image

  # set settings to take the snapshot
  bpy.data.scenes[0].render.engine = "BLENDER_EEVEE"
  bpy.data.materials[mat].node_tree.nodes["Principled BSDF"].inputs[21].default_value = 0
  node_environment.image = None

  # save image in tmp
  bpy.context.scene.render.filepath = '/tmp/img_tmp.png'
  bpy.ops.render.render(write_still=True)

  # open tmp image
  #img = Image.open('/tmp/img_tmp.png')
  img = cv2.imread('/tmp/img_tmp.png')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  #img = cv2.GaussianBlur(img, (5,5), 0)
  #img = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2)
  #cv2.imshow('', img)
  #cv2.waitKey(0)

  # set the values back
  bpy.data.scenes[0].render.engine = old_engine
  bpy.data.materials[mat].node_tree.nodes["Principled BSDF"].inputs[21].default_value = old_value
  node_environment.image = old_img

  return img

def get_polygon_indexes(coords):
  t1, t2 = zip(*coords)
  w = max(t1) - min(t1) + 6
  h = max(t2) - min(t2) + 6
  assert len(t1) == len(t2), "len(t1) == len(t2)"

  coords = np.array([(t1[i]-min(t1)+3, t2[i]-min(t2)+3) for i in range(len(t1))])
  matrix = np.zeros((w, h))

  # fill matrix
  mahotas.polygon.fill_polygon(coords, matrix)

  ind_x, ind_y = np.where(matrix == 1)
  indices = list(zip(ind_x, ind_y))
  indices = [(t[0]+min(t1)-3, t[1]+min(t2)-3) for t in indices]
  return indices

def centroid(vertexes):
  x_list = [vertex[0] for vertex in vertexes]
  y_list = [vertex[1] for vertex in vertexes]
  if len(vertexes[0]) == 3:
    z_list = [vertex[2] for vertex in vertexes]
  _len = len(vertexes)
  x = sum(x_list) / _len
  y = sum(y_list) / _len
  if len(vertexes[0]) == 3:
    z = sum(z_list) / _len
    return (x, y, z)
  else:
    return (x, y)