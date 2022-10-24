from tet_mesh import TetMesh, tet_mesh_cuboid, tet_mesh_from_file
from material import Material, LinearElastic, NeoHookean
from fem import StaticFEM

from numpy import ndarray as array
from typing import Tuple

import os
import argparse
import numpy as np


# Root folder
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cube size
cube_size = 0.025

# Material parameters (Young's modulus and Poisson's ratio)
E, nu = 10000000, 0.45


def boundary_conditions(vertices: array, external_force: array,
                        tolerance: float=1e-8) -> Tuple[array, array]:
    '''
    Create default boundary conditions for a tet mesh.

    Params:
        * `vertices: array`       - (Nxd) vertex positions, N = #vertices, d = #dimensions
        * `external_force: array` - (d) external forces at predefined boundary vertices
        * `tolerance: float`      - floating-point tolerance when determining boundary conditions

    Return value:
        * `f_ext: array` - (Nxd) external forces
        * `bc: array`    - (N) the boundary condition mask
    '''
    # Align the bottom-left corner with the origin
    V = vertices - vertices.min(axis=0)

    # Compute bounding box size
    bbox_size = V.max(axis=0)

    # Fix vertices on the left-most side
    bc = V[:, 0] < tolerance

    # Apply downward forces to vertices on the bottom-right side
    # There is no particular reason for such a setting but to match the C++ result
    num_x_units = int(bbox_size[0] / (tolerance * 2) + 1 + 1e-8)
    f_ext_mask = (V[:, 0] > (num_x_units // 2 * 2 - 3) * tolerance) & (V[:, 2] < tolerance)

    f_ext = np.zeros_like(V)
    f_ext[f_ext_mask] = external_force

    return f_ext, bc


def test_fem(mesh: TetMesh, material: Material, external_force: array, name: str):
    '''
    Default FEM test function.
    '''
    # Print test case info
    print(f'------------ Test: {name}, Material: {material.type} ------------')

    # Result folder
    result_dir = os.path.join(ROOT_DIR, 'data', 'assignment4', 'results', name)
    os.makedirs(result_dir, mode=0o775, exist_ok=True)

    # Save the rest mesh
    mesh.write_to_file(os.path.join(result_dir, 'mesh_rest.stl'))

    # Set boundary conditions
    V = mesh.vertices
    f_ext, bc = boundary_conditions(V, external_force, tolerance=cube_size * 0.5)

    # Create the FEM solver
    fem = StaticFEM(mesh, material)

    # Compute the deformation
    if material.type == 'linear':
        U = fem.solve_linear(f_ext, bc)
    elif material.type == 'nonlinear':
        U = fem.solve_newton(f_ext, bc)
    else:
        raise ValueError('Unknown material model type')

    # Save the stiffness matrix for the linear material model (for case 4x2x2 only)
    if V.shape[0] <= 16:
        # Get the stiffness matrix
        K_full = fem.stiffness_matrix(V)
        active_indices = np.nonzero((~bc).repeat(V.shape[1]))[0]
        K = K_full[active_indices][:, active_indices]

        # Write the stiffness matrix into an external file
        stiffness_matrix_file_name = os.path.join(result_dir, f'K_{name}_{material.type}.txt')
        with open(stiffness_matrix_file_name, 'w') as f:
            for row in K.toarray():
                row_str = ' '.join([f'{v:.6f}' for v in row])
                f.write(f'{row_str}\n')

        print(f"Stiffness matrix saved to '{stiffness_matrix_file_name}'")

    # Construct and save the deformed mesh
    mesh_deform = TetMesh(V + U, mesh.elements)
    output_mesh_file_name = os.path.join(result_dir, f'deformed_{material.type}.stl')
    mesh_deform.write_to_file(output_mesh_file_name)

    print(f"Deformed mesh saved to '{output_mesh_file_name}'")


def boundary_conditions_custom(vertices: array, tolerance: float=1e-8) -> Tuple[array, array]:
    '''
    Create custom boundary conditions for a tet mesh.

    Params:
        * `vertices: array`  - (Nxd) vertex positions, N = #vertices, d = #dimensions
        * `tolerance: float` - floating-point tolerance when determining boundary conditions

    Return value:
        * `f_ext: array` - (Nxd) external forces
        * `bc: array`    - (N) the boundary condition mask
    '''
    # Align the bottom-left corner with the origin
    V = vertices - vertices.min(axis=0)

    # Compute bounding box size (bx, by, bz)
    bbox_size = V.max(axis=0)

    # Set the constraints for fixed vertices during deformation
    # --------
    # TODO: Your code here. You are free to define any boundary constraints you want.
    # HINT:
    #   - `V` has been aligned with the origin, meaning that the vertex coordinates will have
    #     a value range from (0, 0, 0) to (bx, by, bz) as referred to by `bbox_size`
    #   - Use vectorized comparison to avoid for loops. For example, the expression below
    #       `V[:, i] < t`
    #     produces a Boolean array of length N that indicates whether the i-th dimension coordinate
    #     of each vertex in the tet mesh is smaller than some `t` value. You can change the
    #     operator to `>`, `<=` or `>=` and use any value for t as you see fit.
    #     We have provided you with a special variable `tolerance` in the input arguments for
    #     floating-point tolerance in such comparison expressions.
    #   - You may combine multiple comparisons logically using Boolean operators `&` (and), `|`
    #     (or), and `^` (xor), etc. For example,
    #       `(V[:, 0] < t1 + tolerance) & (V[:, 1] >= t2 - tolerance)`
    #     generates a Boolean mask for vertices whose X coordinates are smaller than t1 and whose
    #     Y coordinates are no less than t2 (both with some tolerance).
    bc = np.zeros(V.shape[0], dtype=bool)   # <--

    # Set the external forces
    # --------
    # TODO: Your code here. You are free to decide on what forces to apply.
    # HINT:
    #   - Like the previous blank, you will compute a Boolean mask for the vertices to exert forces
    #     at. You could also read the `boundary_condition` function for reference.
    f_ext_mask = np.zeros(V.shape[0], dtype=bool)   # <--
    f_ext = np.zeros_like(V)
    f_ext[f_ext_mask] = [0, 0, 0]       # <--

    print(f'Boundary conditions: {bc.sum()} fixed points, {f_ext_mask.sum()} external forces')

    return f_ext, bc


def test_fem_custom(mesh: TetMesh, material: Material, name: str):
    '''
    Customized FEM test function.
    '''
    # Print test case info
    print(f'------------ Custom test: {name}, Material: {material.type} ------------')

    # Result folder
    result_dir = os.path.join(ROOT_DIR, 'data', 'assignment4', 'results', name)
    os.makedirs(result_dir, mode=0o775, exist_ok=True)

    # Save the rest mesh
    mesh.write_to_file(os.path.join(result_dir, 'mesh_rest.stl'), invert_normal=True)

    # Set boundary conditions
    V = mesh.vertices
    f_ext, bc = boundary_conditions_custom(V)

    # Solve deformation
    fem = StaticFEM(mesh, material)
    U = fem.solve_newton(f_ext, bc)

    # Construct and save the deformed mesh
    mesh_deform = TetMesh(V + U, mesh.elements)
    output_mesh_file_name = os.path.join(result_dir, f'deformed_{material.type}.stl')
    mesh_deform.write_to_file(output_mesh_file_name, invert_normal=True)

    print(f"Deformed mesh saved to '{output_mesh_file_name}'")


def main():
    '''
    Main routine.
    '''
    # Command line argument parser
    parser = argparse.ArgumentParser(description='HW4 - FEM simulation test')
    parser.add_argument('-m', '--mesh', default='',
                        help='Path to an external tet mesh file (in binary format)')
    parser.add_argument('-c', '--test-cuboid-size', default='',
                        help='Dimensions of the cuboid for testing (e.g. 4x2x2)')
    parser.add_argument('-f', '--test-force', default='0,0,-50',
                        help='The external force for testing (e.g., 0,0,-50)')

    # Process arguments
    args = parser.parse_args()

    mesh_name = args.mesh
    test_cuboid_size = args.test_cuboid_size
    test_force = np.array([int(c) for c in args.test_force.split(',')])

    # Material models
    linear_material = LinearElastic(E, nu)
    neohookean_material = NeoHookean(E, nu)

    # Print welcome message
    print('Alternative HW4')

    # Use an external mesh file as input if specified
    if mesh_name:
        # Read tet mesh data from file
        file_name = os.path.join(ROOT_DIR, 'data', 'assignment4', f'{mesh_name}_tetmesh.dat')
        tet_mesh = tet_mesh_from_file(file_name, max_size=cube_size * 10)

        # Test deformation using the specified mesh
        test_fem_custom(tet_mesh, neohookean_material, mesh_name)

    # Use a custom cuboid size
    elif test_cuboid_size:
        # Get the cuboid dimensions
        cuboid_size = [int(c) for c in test_cuboid_size.split('x')]

        # Create a cuboid tet mesh
        tet_mesh = tet_mesh_cuboid(*cuboid_size, cube_size)

        # Test both linear and non-linear materials
        for material in (linear_material, neohookean_material):
            test_fem(tet_mesh, material, test_force, cuboid_size)

    # Perform default testing with cuboids
    else:
        # Test cuboid sizes
        cuboid_sizes = [(4, 2, 2), (20, 8, 8)]

        # Test loop
        for material in (linear_material, neohookean_material):
            for nx, ny, nz in cuboid_sizes:
                # Create a cuboid tet mesh
                tet_mesh = tet_mesh_cuboid(nx, ny, nz, cube_size)

                # Test deformation using the cuboid mesh
                test_name = f'{nx}x{ny}x{nz}'
                test_fem(tet_mesh, material, test_force, test_name)


if __name__ == '__main__':
    main()
