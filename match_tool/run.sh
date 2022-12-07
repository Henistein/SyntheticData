#!/usr/bin/bash
PYTHONPATH=/usr/local/lib/python3.10 blender --python $1 --python-use-system-env -- args 000001 img000013.png 000001_mesh.npy
#PYTHONPATH=/usr/local/lib/python3.10 blender --python $1 --python-use-system-env -- args 000001 poltrona.png poltrona_mesh.npy
