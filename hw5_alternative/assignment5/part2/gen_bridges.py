import os
import numpy as np
import subprocess

# Root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create folder for the meshes
meshes_dir = os.path.join(ROOT_DIR, 'data', 'assignment5', 'bridges')
os.makedirs(meshes_dir, mode=0o775, exist_ok=True)

# OpenSCAD file name
scad_file = os.path.join(meshes_dir, 'bridge.scad')

# Generate bridge designs
for radius in np.linspace(3.0, 4.0, 11):
    for offset in np.linspace(-3.0, -2.0, 11):

        # Write OpenSCAD script
        with open(scad_file, 'w') as f:
            f.write(
                f'$fn = 100;\n'
                f'difference() {{\n'
                f'    cube([10,5,5], center=true);\n'
                f'    rotate([90,0,0]) translate([0,{offset:.2f},0]) cylinder(10,{radius:.2f},{radius:.2f}, center=true);\n'
                f'}}\n'
            )

        # output mesh
        mesh_file = os.path.join(meshes_dir, f'bridge_r_{int(radius * 10)}_o_{int(offset * 10)}.stl')
        subprocess.run(f"openscad -o {mesh_file} {scad_file}", shell=True)
