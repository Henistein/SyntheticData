import bpy
import pickle
import numpy as np
from sklearn.neighbors import KDTree

MAP = {}
name = "airplane"

bpy.ops.import_scene.obj(filepath=f"models/{name}.obj")
#obj = bpy.data.objects[-1]
obj = bpy.data.objects['AirPlane']

vertices = list(obj.data.vertices)

for face in list(obj.data.polygons):
  faces_verts = [vertices[i].co for i in list(face.vertices)]
  flat = tuple(round(item, 3) for sublist in faces_verts for item in sublist)
  MAP[flat] = tuple(face.center)


X = list(map(lambda x: list(x.co), vertices))
keys = np.array(X)
tree =  KDTree(X, leaf_size=2)

# save dict
with open(f'pkls/{name}_map.pkl', 'wb') as f:
  pickle.dump(MAP, f)

# save tree
with open(f'pkls/{name}_tree.pkl', 'wb') as f:
  pickle.dump(tree, f)

# save points list
with open(f'pkls/{name}_list.pkl', 'wb') as f:
  pickle.dump(X, f)
        
"""
with open('saved_dictionary.pkl', 'rb') as f:
  loaded_dict = pickle.load(f)
"""
