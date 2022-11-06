import bpy
import pickle

MAP = {}
name = "teste"

bpy.ops.import_scene.obj(filepath=f"models/{name}.obj")
obj = bpy.data.objects[-1]

vertices = list(obj.data.vertices)

for face in list(obj.data.polygons):
  faces_verts = [vertices[i].co for i in list(face.vertices)]
  flat = tuple(round(item, 3) for sublist in faces_verts for item in sublist)
  MAP[flat] = tuple(face.center)

# save dict
with open(f'{name}.pkl', 'wb') as f:
  pickle.dump(MAP, f)
        
"""
with open('saved_dictionary.pkl', 'rb') as f:
  loaded_dict = pickle.load(f)
"""
