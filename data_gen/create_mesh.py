import bpy
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='', help='name')
parser.add_argument('--obj_name', type=str, default='', help='obj name')
parser.add_argument('--save_path', type=str, default='', help='path to save the vertices')
parser = parser.parse_args()


name = parser.name
obj_name = parser.obj_name

bpy.ops.import_scene.obj(filepath=f"models/{name}.obj")
obj = bpy.data.objects[obj_name]

vertices = np.array([list(v.co) for v in obj.data.vertices])

# save vertices
np.save(f"{parser.save_path}/{name}_mesh.npy", vertices)