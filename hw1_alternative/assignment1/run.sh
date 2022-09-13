#!/bin/bash
# Brute-force voxelization
python assignment1/main.py spot bf 0.125
python assignment1/main.py bunny bf 2.0
python assignment1/main.py fandisk bf 0.05
python assignment1/main.py dragon bf 0.05

# Accelerated voxelization
python assignment1/main.py spot fast 0.125
python assignment1/main.py bunny fast 2.0
python assignment1/main.py fandisk fast 0.05
python assignment1/main.py dragon fast 0.05

# Approximate voxelization for non-watertight meshes
python assignment1/main.py spot_with_hole approx 0.125
python assignment1/main.py bunny_with_hole approx 2.0

# Marching cubes
python assignment1/main.py bunny mc
python assignment1/main.py spot mc
python assignment1/main.py fandisk mc
python assignment1/main.py dragon mc
