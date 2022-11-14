import subprocess
import sys

step = 1000
offsets = list(range(2000, 33000, step))
f = sys.argv[1]

for offset in offsets:
  args = f"blender --background --python main.py --python-use-system-env -- load_generated_data {f} {offset} {offset+step}".split(' ')
  ret = subprocess.run(args)

