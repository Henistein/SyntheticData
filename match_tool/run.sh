#!/usr/bin/bash
PYTHONPATH=/usr/local/lib/python3.10 blender --background --python $1 --python-use-system-env -- args sofa_2K DATA_TESTE/sofa_2K/images/img000000.png DATA_TESTE/sofa_2K/mesh/sofa_2K_norm_mesh.npy
#PYTHONPATH=/usr/local/lib/python3.10 blender --python $1 --python-use-system-env -- args 000001 poltrona.png poltrona_mesh.npy
