import bpy

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

def cut_obj_camera_view(obj, plane):
  # unselect all, then select cube and plane
  for ob in bpy.context.selected_objects:
    ob.select_set(False)
  bpy.context.view_layer.objects.active = obj 
  bpy.ops.object.editmode_toggle()
  plane.select_set(True)

  for i,area in enumerate(bpy.context.screen.areas):
    if area.type == 'VIEW_3D':
      bpy.context.screen.areas[i].spaces[0].region_3d.view_perspective = 'CAMERA'
      area.spaces[0].region_3d.update()
      for region in area.regions:
        if region.type == 'WINDOW':
          override = {'area': area, 'region': region, 'edit_object':bpy.context.edit_object}
          break
      break

  # cut
  with bpy.context.temp_override(**override):
    bpy.ops.mesh.knife_project()

  # separate cut object and set to object mode
  bpy.ops.mesh.separate(type="SELECTED")
  bpy.ops.object.editmode_toggle()

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