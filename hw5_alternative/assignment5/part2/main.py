from voxelizer import Voxelizer
from material import LinearElastic
from tet_mesh import TetMesh
from fem import StaticFEM
from pareto import pareto_front

from typing import Tuple, List

import os
import sys
import numpy as np
import pandas as pd


# Root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pareto_front():
    '''
    Unit test for the Pareto front function.
    '''
    # Set the random seed
    np.random.seed(1)

    # Generate a group of random 2D points
    N = 200000
    points = np.stack((np.arange(1, N + 1), np.arange(N, 0, -1))).astype(np.float64).T
    points += np.random.uniform(low=0.0, high=2.0, size=(N, 2))

    # Compute the Pareto front and sort the Points in ascending X
    pareto = pareto_front(points)
    pareto = pareto[np.argsort(pareto[:, 0])]

    # Create the result folder
    result_dir = os.path.join(ROOT_DIR, 'data', 'assignment5', 'results', 'part2')
    os.makedirs(result_dir, mode=0o775, exist_ok=True)

    # Save the Pareto front as a CSV file
    csv_file_path = os.path.join(result_dir, 'q1_result.csv')
    pd.DataFrame(pareto, columns=['x', 'y']).to_csv(csv_file_path, index=False)

    print(f"Results saved to '{csv_file_path}'")


def set_boundary_conditions(mesh: TetMesh) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Set the boundary conditions for a bridge tetrahedral mesh.

    Params:
        * `mesh: TetMesh` - The tet mesh of a bridge design

    Return values:
        * `f_ext: array`   - (Nxd) External forces, N = #vertices, d = #dimensions
        * `bc_mask: array` - (N) Boundary condition mask (True = fixed, False = free)
    '''
    # Get the vertex array of the tet mesh
    vertices = mesh.vertices

    # Compute the bounding box
    bbox_min = vertices.min(axis=0)     # Bottom-left corner
    bbox_max = vertices.max(axis=0)     # Top-right corner

    # Initialize an empty external force array
    f_ext = np.zeros_like(vertices)

    # Apply -5000N forces in the Z direction at the top layer of vertices
    # --------
    # TODO: Your code here. First calculate the vertex mask, then specify the force to apply.
    f_ext_mask = np.ones(len(vertices), dtype=bool)     # <--
    f_ext[f_ext_mask] = 0                               # <--

    # Obtain the boundary condition mask
    # Note that the vertices on the left and right sides of the bridge are fixed
    # --------
    # TODO: Your code here. Use bit-wise Boolean operations (`&` or `|`) to chain multiple rules.
    bc_mask = np.zeros(len(vertices), dtype=bool)       # <--

    return f_ext, bc_mask


def solve_performance(stl_file: str, save_tet_mesh: str='') -> Tuple[float, float]:
    '''
    Compute the compliance and the mass of a bridge design.

    Params:
        * `stl_file: str`      - Path to the bridge mesh
        * `save_tet_mesh: str` - Path to the saved tet mesh (optional)

    Return values:
        * `compliance: float` - Compliance of the bridge
        * `mass: float`       - Mass of the bridge
    '''
    # Constants
    material = LinearElastic(1e7, 0.45)     # Linear elastic material for FEM
    voxel_size = 0.25                       # Voxel size for voxelization

    # Read and voxelize the input mesh
    voxelizer = Voxelizer(stl_file, voxel_size)
    voxelizer.run_accelerated()

    # Convert the voxel grid into a tetrahedral mesh
    # --------
    # TODO: Your code here. Use a proper method of the Voxelizer class.
    tet_mesh = None         # <--

    # Run static FEM analysis and compute compliance
    # --------
    # TODO: Your code here. Complete the process by filling in the following lines:
    #   1. Create the FEM solver by instantiating a StaticFEM object
    fem = ...               # <--

    #   2. Specify boundary conditions using the `set_bounadry_conditions` function
    f_ext, bc_mask = ..., ...       # <--

    #   3. Solve the FEM problem using the linear solver
    u = ...                 # <--

    #   4. Compute compliance (C = F^T * U)
    # HINT: use `a.ravel()` to flatten a 2D array
    compliance = 0.0        # <--

    # Compute mass (number of voxels in the voxelized mesh)
    # --------
    # TODO: Your code here. Use the `np.count_nonzero` function to count the number of non-zero
    # values in an array.
    mass = 0.0              # <--

    # Optionally save the tetrahedral mesh
    if tet_mesh and save_tet_mesh:
        tet_mesh.write_to_file(save_tet_mesh)

    return compliance, mass


def test_bridge_design():
    '''
    Unit test for a sample bridge design.
    '''
    # Test cases (size, offset)
    test_cases = [(30, -30), (40, -25)]

    # Folder of bridge meshes
    mesh_dir = os.path.join(ROOT_DIR, 'data', 'assignment5', 'bridges')

    # Result folder for saving voxelized meshes
    result_dir = os.path.join(ROOT_DIR, 'data', 'assignment5', 'results', 'part2')

    # Iterate over all test cases
    for i, (size, offset) in enumerate(test_cases):
        print(f'Testing design {i + 1}/{len(test_cases)}')

        # Compute the performance metrics of the current test design
        try:
            mesh_file = os.path.join(mesh_dir, f'bridge_r_{size}_o_{offset}.stl')
            save_mesh_file = os.path.join(result_dir, f'bridge_r_{size}_o_{offset}_voxels.stl')
            compliance, mass = solve_performance(mesh_file, save_tet_mesh=save_mesh_file)
            print(f'Test ({size}, {offset}) - compliance = {compliance}, mass = {mass}')

        except FileNotFoundError:
            print(f"Error - Bridge mesh '{mesh_file}' not found. Did you run gen_bridges.py?")
            quit()


def run_bridges():
    '''
    Evaluate all bridge designs and output the Pareto front.
    '''
    # Test cases (size, offset)
    test_cases = [(size, offset) for size in range(30, 41) for offset in range(-30, -19)]

    # Folder of bridge meshes
    mesh_dir = os.path.join(ROOT_DIR, 'data', 'assignment5', 'bridges')

    # Iterate over all 121 test cases (this is going to take a while ...)
    perf: List[Tuple[float, float]] = []

    for i, (size, offset) in enumerate(test_cases):
        print(f'Testing design {i + 1}/{len(test_cases)}')

        # Compute the performance metrics of the current bridge design
        try:
            mesh_file = os.path.join(mesh_dir, f'bridge_r_{size}_o_{offset}.stl')
            compliance, mass = solve_performance(mesh_file)
            print(f'Bridge ({size}, {offset}) - compliance = {compliance}, mass = {mass}')

            # Record the performance metrics
            perf.append((mass, compliance))

        except FileNotFoundError:
            print(f"Error - Bridge mesh '{mesh_file}' not found. Did you run gen_bridges.py?")
            quit()

    # Compute the Pareto front
    pareto = pareto_front(np.array(perf, dtype=np.float64))

    # Save the Pareto front as a CSV file
    result_dir = os.path.join(ROOT_DIR, 'data', 'assignment5', 'results', 'part2')
    csv_file_path = os.path.join(result_dir, 'q2_result.csv')
    pd.DataFrame(pareto, columns=['x', 'y']).to_csv(csv_file_path, index=False)

    print(f"Pareto front saved to '{csv_file_path}'")

    # Save all performance metrics
    csv_file_path = os.path.join(result_dir, 'q2_all_perfs.csv')
    pd.DataFrame(perf, columns=['x', 'y']).to_csv(csv_file_path, index=False)

    print(f"Performance metrics saved to '{csv_file_path}'")


def main():
    '''
    Main routine.
    '''
    # Display welcome message
    print('Welcome to Alternative Assignment 5')

    # Test mode - Pareto front
    if len(sys.argv) == 2 and sys.argv[1] == 'test_pareto':
        print('Testing the Pareto front function ...')
        test_pareto_front()

    # Test mode - bridge design
    elif len(sys.argv) == 2 and sys.argv[1] == 'test_bridge':
        print('Testing bridge design examples ...')
        test_bridge_design()

    # Run all bridge designs
    else:
        print('Running all bridge designs ...')
        run_bridges()


if __name__ == '__main__':
    main()
