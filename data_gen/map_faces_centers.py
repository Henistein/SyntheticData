import bpy
import pickle
import numpy as np
import yaml
from sklearn.neighbors import KDTree

name = 'sofa'
obj_name = 'Sofa'
path = '/home/socialab/Henrique/SyntheticData/data_gen/models'

bpy.ops.import_scene.obj(filepath=f"models/{name}.obj")
obj = bpy.data.objects[obj_name]

vertices = np.array([list(v.co) for v in obj.data.vertices])

# save vertices
#np.save(path+f"/mesh/{name}.npy", vertices)

MAP = {tuple(face.center):face for face in list(obj.data.polygons)}
"""
for face in list(obj.data.polygons):
  #faces_verts = [vertices[i] for i in list(face.vertices)]
  #flat = tuple(item for sublist in faces_verts for item in sublist)
  MAP[tuple(face.center)] = face
"""

centers = np.array(list(MAP.keys()))

# pass centers to kdtree
#X = list(map(lambda x: list(x.co), vertices))
tree =  KDTree(centers, leaf_size=2)

# query center
x = [(-0.55251166, 0.412381, -0.08782556)]
_, ind = tree.query(x, k=3)
ind = ind[0, 0]

# get center
center = tuple(centers[ind].flatten())

print(MAP[center].index)
exit()


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
