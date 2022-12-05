#!/usr/bin/bash
PYTHONPATH=/usr/local/lib/python3.10 blender --python $1 --python-use-system-env -- args 000001 img000020.png 000001_mesh.npy
