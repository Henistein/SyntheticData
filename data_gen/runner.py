import sys
import argparse
import subprocess

step = 1000
offsets = list(range(0, 33000, step))

parser = argparse.ArgumentParser()
parser.add_argument('--gen_data_path', type=str, default='', help='path to gen_data file')
parser.add_argument('--obj_path', type=str, default='', help='path to obj')
parser.add_argument('--name', type=str, default='', help='name')
parser.add_argument('--obj_name', type=str, default='', help='obj name')
parser.add_argument('--save_path', type=str, default='', help='path to save the dataset')
parser = parser.parse_args()

"""
-- object
 - object path
 - name
 - object name
 - save path

-- load_generated_data
 - file path
 - first index
 - final index
"""

for offset in offsets:
  args = f"blender --background --python main.py --python-use-system-env -- load_generated_data {parser.gen_data_path} {offset} {offset+step} -- obj {parser.save_path} {parser.name} {parser.obj_name}".split(' ')
  ret = subprocess.run(args)

