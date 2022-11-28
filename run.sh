#!/usr/bin/bash
PYTHONPATH=/usr/local/lib/python3.10 blender --background --python $1 --python-use-system-env -- args 000001 000001
