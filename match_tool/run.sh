#!/usr/bin/bash
PYTHONPATH=/usr/local/lib/python3.10 blender --python $1 --python-use-system-env -- args a000420.npy sofa Sofa img000420.png
