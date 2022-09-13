from voxelizer import Voxelizer

from trimesh.voxel.base import VoxelGrid
from trimesh import Trimesh

import os
import argparse
import time
import numpy as np

# The root folder of the assignment
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_cmd_args():
    '''
    Parse command line arguments (including mesh name)
    '''
    # Set up a command line argument parser
    parser = argparse.ArgumentParser(description='HW1 program.')
    parser.add_argument('mesh_name',
                        choices=[
                            'bunny', 'dragon', 'fandisk', 'spot',
                            'bunny_with_hole', 'spot_with_hole'
                        ],
                        help='Input mesh name (choices are bunny, dragon, fandisk, and spot)')
    parser.add_argument('mode', choices=['bf', 'fast', 'approx', 'mc'],
                        help='Brute-force (bf), accelerated (fast), and approximate (approx) '
                             'voxelization; or marching cubes (mc)')
    parser.add_argument('voxel_size', type=float, default=1.0, nargs='?',
                        help='Voxel size in the voxelized mesh')
    parser.add_argument('-f', '--save-txt-file', action='store_true',
                        help='Save the voxelized mesh in an additional txt file')
    parser.add_argument('-s', '--seed', type=int, default=-1,
                        help='Random seed for reproducibility (negative means none)')
    parser.add_argument('--no-result-check', action='store_true',
                        help='Whether to disable result checking for ray mesh intersection trials')

    # Parse the arguments that the user passed to the program
    args = parser.parse_args()
    return args


def run_marching_cubes(voxel_file_path: str, mesh_name: str):
    '''
    Read a voxel grid from external file and convert it into a triangle mesh using marching cubes.
    '''
    print('=' * 20)
    print(f"Converting voxel grid '{mesh_name}' into triangle mesh ...")

    # Read the voxel grid from the file path
    voxel_data = np.load(voxel_file_path)

    # Start timer
    t_start = time.time()

    # Get the converted mesh
    mesh: Trimesh = VoxelGrid(voxel_data['voxels']).marching_cubes

    # End timer and compute duration
    t_elapsed = time.time() - t_start

    print(f'Marching cubes finished, time elapsed = {t_elapsed:.3f}s')

    # Transform the mesh back to the original scale
    mesh.apply_scale(voxel_data['voxel_size'])
    mesh.apply_translation(voxel_data['voxel_grid_min'])

    # Create the result folder if it doesn't exist, e.g., the output for voxel grid 'bunny'
    # will be stored in '.../hw1_alternative/data/assignment1/results/bunny_mc'
    result_folder = \
        os.path.join(ROOT_DIR, 'data', 'assignment1', 'results', f'{mesh_name}_mc')
    os.makedirs(result_folder, mode=0o775, exist_ok=True)

    # Save the mesh into the result folder
    output_file_path = os.path.join(result_folder, f'{mesh_name}_mc.stl')
    mesh.export(output_file_path)

    print(f"Marching cubes result saved to '{output_file_path}'")


def run_voxelization(voxelizer: Voxelizer, mode: str, check_result: bool=True,
                     save_txt_file: bool=False):
    '''
    Wrapper of voxelization functions.
    '''
    # Retrieve mesh name
    mesh_name = voxelizer.mesh_name

    print('=' * 20)
    print(f"Voxelizing mesh '{mesh_name}' using brute-force method "
          f'(grid size = {voxelizer.voxel_grid_size.tolist()}) ...')

    # Start timer
    t_start = time.time()

    # Run brute-force voxelization
    if mode == 'bf':
        occupancy = voxelizer.run_brute_force()
    # Run accelerated voxelization
    elif mode == 'fast':
        occupancy = voxelizer.run_accelerated(check_result=check_result)
    # Run approximate voxelization
    else:
        occupancy = voxelizer.run_approximate()

    # Stop timer and compute duration
    t_elapsed = time.time() - t_start

    print(f'Voxelization finished, grid occupancy = {occupancy * 100:.3f}%, '
          f'time elapsed = {t_elapsed:.3f}s')

    # Create the result folder if it doesn't exist, e.g., the output for mesh 'bunny'
    # will be stored in '.../hw2_alternative/data/assignment2/results/bunny'
    result_folder_name = mesh_name + '_fast' * (mode == 'fast')
    result_folder = \
        os.path.join(ROOT_DIR, 'data', 'assignment1', 'results', result_folder_name)
    os.makedirs(result_folder, mode=0o775, exist_ok=True)

    # Save the voxel grid as a triangle mesh for visualization
    mesh_file_path = os.path.join(result_folder, f'{mesh_name}_voxel.stl')
    voxelizer.save_mesh(mesh_file_path)

    # Save the voxel grid to a binary file
    binary_file_path = os.path.join(result_folder, f'{mesh_name}_voxel_data.npz')
    voxelizer.save_to_data_file(binary_file_path)

    # Optionally save the voxel grid to a text file if specified by the user
    if save_txt_file:
        txt_file_path = os.path.join(result_folder, f'{mesh_name}_voxel_info.txt')
        voxelizer.save_to_txt_file(txt_file_path)


def main():
    '''
    The main function is just like those in C/C++ programs but actually not required in Python.
    However, it's still recommended to enclose the program body in a 'main'-like function.
    This reduces the cost of maintaining global variables and makes your program more structured.
    '''
    # Commands for individual testing
    # # Brute-force voxelization
    # python assignment1/main.py spot bf 0.125
    # python assignment1/main.py bunny bf 2.0
    # python assignment1/main.py fandisk bf 0.05
    # python assignment1/main.py dragon bf 0.05

    # # Accelerated voxelization
    # python assignment1/main.py spot fast 0.125
    # python assignment1/main.py bunny fast 2.0
    # python assignment1/main.py fandisk fast 0.05
    # python assignment1/main.py dragon fast 0.05

    # # Approximate voxelization for non-watertight meshes
    # python assignment1/main.py spot_with_hole approx 0.125
    # python assignment1/main.py bunny_with_hole approx 2.0

    # # Marching cubes
    # python assignment1/main.py bunny mc
    # python assignment1/main.py spot mc
    # python assignment1/main.py fandisk mc
    # python assignment1/main.py dragon mc

    # Print a welcome message
    print('Welcome to CompFab Alternative Assignment 1')

    # Get the mesh name and voxel size from command line arguments
    args = get_cmd_args()

    mesh_name = args.mesh_name
    voxel_size = args.voxel_size

    # Fix random seed
    if args.seed >= 0:
        np.random.seed(args.seed)

    # Run marching cubes
    if args.mode == 'mc':
        # Set the path to the voxel data file according to mesh name
        voxel_file_path = \
            os.path.join(ROOT_DIR, 'data', 'assignment1', 'references', mesh_name,
                         f'{mesh_name}_voxel_data.npz')

        # Run marching cubes
        run_marching_cubes(voxel_file_path, mesh_name)

    # Run voxelization
    else:
        # Set the path to the STL mesh file according to mesh name
        stl_file_path = os.path.join(ROOT_DIR, 'data', 'assignment1', f'{mesh_name}.stl')

        # Create a voxelizer using the source STL file
        voxelizer = Voxelizer(stl_file_path, voxel_size)

        # Run voxelization
        run_voxelization(
            voxelizer, mode=args.mode,
            check_result=~args.no_result_check,
            save_txt_file=args.save_txt_file
        )


if __name__ == '__main__':
    main()
