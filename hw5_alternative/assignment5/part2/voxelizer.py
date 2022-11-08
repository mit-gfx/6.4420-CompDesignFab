from intersection import single_ray_mesh_intersection, parallel_ray_mesh_intersection
from tet_mesh import TetMesh

from trimesh.transformations import random_rotation_matrix
from copy import deepcopy

from io import TextIOWrapper
from typing import List

import os
import trimesh
import numpy as np


def write_triangle(f: TextIOWrapper, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                   normal: List[float]):
    '''
    Write a triangle facet to the output file stream 'f'.
    '''
    f.write(
        f"  facet normal {' '.join([f'{val:.6g}' for val in normal])}\n"
        f"    outer loop\n"
        f"      vertex {' '.join([f'{val:.6g}' for val in p1])}\n"
        f"      vertex {' '.join([f'{val:.6g}' for val in p2])}\n"
        f"      vertex {' '.join([f'{val:.6g}' for val in p3])}\n"
        f"    endloop\n"
        f"  endfacet\n"
    )


class Voxelizer:
    '''
    Voxelizer class.
    '''

    def __init__(self, stl_file_path: str, voxel_size: float):
        '''
        Constructor of the Voxelizer class.
        '''
        # Extract mesh name from the STL file path
        mesh_name = os.path.splitext(os.path.split(stl_file_path)[1])[0]

        # Read input mesh in STL file
        mesh = trimesh.load(stl_file_path)

        # Save the local variables as class members so that they can be used in other functions
        self.mesh_name = mesh_name
        self.mesh = mesh
        self.voxel_size = voxel_size

        # Initialize the voxel grid
        self.init_voxels()


    def init_voxels(self):
        '''
        Initialize the voxel grid.
        '''
        # Compute the bounding box of the mesh
        # 'bbox_min' is the bottom left corner of the mesh, while 'bbox_max' is the top right corner
        bbox_min = self.mesh.vertices.min(axis=0)
        bbox_max = self.mesh.vertices.max(axis=0)

        # Allocate a voxel grid slightly bigger than the mesh by padding the bounding box of the mesh
        dx = self.voxel_size
        voxel_grid_min = bbox_min - dx
        voxel_grid_max = bbox_max + dx

        # Compute the number of voxels needed in each dimension
        # np.ceil() rounds the elements in an array up to the smallest integers
        # The results are then cast to integer data type
        voxel_grid_size = np.ceil((voxel_grid_max - voxel_grid_min) / dx)
        voxel_grid_size = voxel_grid_size.astype(np.int64)

        # Allocate an empty voxel grid
        # (using 8-bit unsigned integers saves memory since each voxel stores either 0 or 1)
        voxels = np.zeros(voxel_grid_size, dtype=np.uint8)

        # Save the grid info as class members
        self.voxel_grid_min = voxel_grid_min
        self.voxel_grid_size = voxel_grid_size
        self.voxels = voxels


    def run_brute_force(self) -> float:
        '''
        Run brute-force voxelization.
        '''
        # Read voxel grid dimensions from the class
        nx, ny, nz = self.voxel_grid_size

        # These class members are frequently used so it's best to assign them to local variables
        mesh = self.mesh        # Input mesh
        dx = self.voxel_size    # Voxel size
        voxels = self.voxels    # Voxel grid

        # Compute the center of the bottom-left voxel, which we use to derive the centers of
        # other voxels
        voxel_bottom_left = self.voxel_grid_min + dx * 0.5

        # Loop over all positions in the voxel grid
        # Note that this nested loop is slow and might run for several minutes
        # --------
        # TODO: Your code here. Write a nested loop over all three voxel grid dimensions
        while False:            # <--
            while False:        # <--
                while False:    # <--

                    # Set ray origin as the current voxel center
                    # --------
                    # TODO: Your code here. Compute the voxel center and assign it to ray_origin
                    ray_origin = np.zeros(3)

                    # Set ray direction as positive X direction by default
                    ray_direction = np.array([1.0, 0.0, 0.0])

                    # Intersect the ray with the mesh and get the intersection locations
                    # --------
                    # TODO: Your code here. Invoke the `single_ray_mesh_intersection` function to
                    # get the intersections, which is defined in `intersection.py`.
                    locations = []

                    # Determine whether the voxel at the current grid point is inside the mesh.
                    # Recall from lectures that an odd number of intersections means inside
                    # --------
                    # TODO: Your code here. Set the value the current voxel in integer format
                    pass

            print(f'Completed layer {i + 1} / {nx}')

        # Compute the occupancy of the voxel grid, i.e., the fraction of voxels inside the mesh
        occupancy = np.count_nonzero(voxels) / voxels.size

        return occupancy


    def run_accelerated(self, check_result: bool=True) -> float:
        '''
        Run accelerated voxelization.
        '''
        # Read voxel grid dimensions from the class
        nx, ny, _ = self.voxel_grid_size

        # These class members are frequently used so it's best to assign them to local variables
        mesh = self.mesh                # Input mesh
        dx = self.voxel_size            # Voxel size
        voxels = self.voxels            # Voxel grid

        # Compute the origin of the bottom-left ray, which we use to derive other ray origins
        # Note that all ray origins lie on the Z=0 plane so they will be outside the mesh
        origin_bottom_left = self.voxel_grid_min + np.array([1, 1, 0]) * (dx * 0.5)

        # Precompute ray origins and directions
        num_rays = nx * ny
        ray_origins = origin_bottom_left + dx * \
            np.hstack((
                np.stack(np.mgrid[:nx, :ny], axis=2).reshape(-1, 2),
                np.zeros((num_rays, 1))
            ))
        ray_direction = np.array([0.0, 0.0, 1.0])

        # Clear the voxel grid
        # --------
        # TODO: Your code here. Set the voxel grid to empty in one line of code
        # (hint: slicing indexing).
        pass

        # Intersect the rays with the mesh
        # --------
        # TODO: Your code here. Invoke the `parallel_ray_mesh_intersection` function to get the
        # intersections, which is defined in `intersection.py`. Note that the value some keyword
        # argument should be provided.
        intersections = []

        # Fill the voxels by looping over all rays
        # --------
        # TODO: Your code here. Write a nested loop over each ray.
        while False:        # <--
            while False:    # <--

                # Get the intersections of the current ray
                distances = np.array(intersections[i * ny + j])

                # Only process rays with intersections
                if len(distances):

                    # Convert distances to alternate indices of interval endpoints
                    lower_indices = np.ceil(distances[::2] / dx - 0.5 - 1e-8).astype(np.int64)
                    upper_indices = np.floor(distances[1::2] / dx + 0.5 + 1e-8).astype(np.int64)

                    # Fill voxels within the interval
                    # --------
                    # TODO: Your code here. Write a for loop with one line of code as its body.
                    # The loop handles each segment of voxels marked by the lower and upper indices
                    # computed above (hint: slicing indexing).
                    for _ in []:        # <-- for loop
                        pass            # <-- loop body

        # Compute the occupancy of the voxel grid, i.e., the fraction of voxels inside the mesh
        occupancy = np.count_nonzero(voxels) / voxels.size

        return occupancy


    def run_approximate(self, num_samples: int=20) -> float:
        '''
        Run approximate voxelization on a non-watertight mesh. This method actually doesn't check
        watertight-ness so it can run on any mesh.
        '''
        # Maximum number of samples is 255, otherwise the voxel counters will overflow
        if num_samples > 255:
            raise ValueError('At most 255 samples are supported')

        # Constants
        dx = self.voxel_size

        # Back up the current mesh
        mesh_backup = deepcopy(self.mesh)

        # Read the current voxel grid info
        nx, ny, nz = self.voxel_grid_size
        voxel_grid_size = self.voxel_grid_size
        grid_bottom_left = self.voxel_grid_min

        # Precompute the voxel centers in the current voxel grid
        voxel_centers = grid_bottom_left + dx * 0.5 + \
            np.stack(np.mgrid[:nx, :ny, :nz], axis=3).reshape(-1, 3) * dx

        # Collect an initial sample by running accelerated voxelization
        self.run_accelerated(check_result=False)
        voxels_count = self.voxels

        print(f'Finished 1 / {num_samples} samples')

        # Collect other samples by voxelizing the mesh in random directions
        for i in range(2, num_samples + 1):

            # Create a copy of the original mesh for us to work on
            self.mesh = deepcopy(mesh_backup)

            # Rotate the mesh using a random rotation matrix (with a size of 4x4)
            R = random_rotation_matrix()
            # --------
            # TODO: Your code here. Rotate the mesh by calling the `apply_transform` method of the
            # Trimesh class, defined in trimesh/base.py which is accessible on Github:
            # https://github.com/mikedh/trimesh/blob/master/trimesh/base.py
            pass

            # Voxelize the rotated mesh to obtain a new voxel grid
            self.init_voxels()
            # --------
            # TODO: Your code here. Run accelerated voxelization on the rotated mesh.
            # Note that you should set a proper value for keyword argument(s).
            pass

            # Now we will sample the new voxel grid at the rotated positions of voxel centers
            # in the original voxel grid. First, we compute the rotated coordinates of the original
            # voxel centers
            # --------
            # TODO: Your code here. Rotate the voxel centers from the original coordinates
            # to the new coordinates (where the rotated mesh resides) using matrix multiplication.
            # In NumPy, matrix multiplication can be written as `np.dot(A, B)` where A is MxN and
            # B is NxP. Be aware that in this case `voxel_centers` is an Nx3 array while
            # the rotation matrix is 4x4. What else should be done?
            rotated_voxel_centers = voxel_centers

            # Then, align the rotated voxel centers with the new bottom-left corner
            rotated_voxel_centers -= self.voxel_grid_min

            # Discard voxel centers outside the new voxel grid area
            new_voxel_grid_bound = self.voxel_grid_size * dx
            in_bound_mask = \
                np.all((rotated_voxel_centers >= 0) & \
                       (rotated_voxel_centers < new_voxel_grid_bound), axis=1)
            rotated_voxel_centers = rotated_voxel_centers[in_bound_mask]

            # Round in-bound voxel centers to integer coordinates
            rotated_indices = np.floor(rotated_voxel_centers / dx + 1e-8)
            rotated_indices = rotated_indices.astype(np.int64)

            # Now we extract the sampled values from the new voxel grid
            rid = rotated_indices
            new_voxels = self.voxels[rid[:, 0], rid[:, 1], rid[:, 2]]

            # Add the sampled values to the corresponding voxel counters
            voxels_count.ravel()[in_bound_mask] += new_voxels

            print(f'Finished {i} / {num_samples} samples')

        # Reset mesh and voxel grid info
        self.mesh = mesh_backup
        self.voxel_grid_size = voxel_grid_size
        self.voxel_grid_min = grid_bottom_left

        # Set grid occupany according to voxel counters
        # --------
        # TODO: Your code here. Assign the voxel grid with values according to the majority rule.
        # Please cast the result to `np.uint8` data type according to the default setting in
        # `init_voxels`.
        self.voxels = np.zeros_like(voxels_count)

        # Compute the occupancy of the voxel grid, i.e., the fraction of voxels inside the mesh
        occupancy = np.count_nonzero(self.voxels) / self.voxels.size

        return occupancy


    def save_mesh(self, output_file_path: str):
        '''
        Save the voxel grid as a triangle mesh in STL format.
        '''
        # Read relevant variables from the class
        nx, ny, nz = self.voxel_grid_size
        dx = self.voxel_size
        voxels = self.voxels
        grid_bottom_left = self.voxel_grid_min

        # Precompute all grid point coordinates. Unlike voxel centers, grid points are offset by
        # half of the voxel size.
        grid_indices = np.mgrid[:nx + 1, :ny + 1, :nz + 1]
        grid_indices = np.stack(grid_indices, axis=3)
        grid_points = grid_bottom_left + dx * grid_indices

        # Cache all possible normals
        normals = np.hstack((-np.eye(3), np.eye(3))).reshape(-1, 3)

        # Cache all index slices
        slices = np.array([slice(None, -1), slice(1, None), slice(1, -1)])

        # Start writing to the output file
        with open(output_file_path, 'w') as f:
            # Write the header
            f.write('solid vcg\n')

            # Generate triangles perpendicular to X, Y, and Z direction respectively
            for dim, axis in enumerate('XYZ'):

                # Take the difference between neighboring voxels along the current axis
                # We need to generate a square facet whenever the difference is not zero,
                # which indicates voxelized mesh boundary
                diff = np.diff(voxels, axis=dim)

                # Consider two normal directions along each axis: positive and negative
                # Inside the inner loop, we generate the group of triangles that share the same
                # normal direction
                for positive in (0, 1):

                    # Compute the set of grid points that will be used as triangle vertices
                    grid_mask = diff == 0xff if positive else diff == 1
                    vertices = [
                        grid_points[
                            tuple(slices[np.roll([i // 2, i % 2, 2], dim + 1)].tolist())
                        ][grid_mask] for i in range(4)
                    ]

                    # Get the current normal direction
                    normal = normals[dim * 2 + positive]

                    # According to STL format, the order of vertices in each triangle satisfies the
                    # right-hand rule.
                    for p1, p2, p3, p4 in zip(*vertices):
                        if positive:
                            write_triangle(f, p1, p4, p2, normal)
                            write_triangle(f, p1, p3, p4, normal)
                        else:
                            write_triangle(f, p1, p2, p4, normal)
                            write_triangle(f, p1, p4, p3, normal)

            # Write the footer
            f.write('endsolid\n')

        # Display a finish message
        print(f"Voxelized mesh written to file '{output_file_path}'")


    def save_to_txt_file(self, output_file_path: str):
        '''
        Save the voxelized mesh as a text file of 0-1 strings. This format is intended for
        user inspection.
        '''
        # Read relevant variables from the class
        nx, ny, nz = self.voxel_grid_size
        dx = self.voxel_size
        voxels = self.voxels
        grid_bottom_left = self.voxel_grid_min

        # Open/Create the output file in write mode
        with open(output_file_path, 'w') as f:
            # Write the bottom left position, voxel size, and grid dimensions
            f.write(f'{grid_bottom_left[0]} {grid_bottom_left[1]} {grid_bottom_left[2]} '
                    f'{dx} {nx} {ny} {nz}\n')

            # Write the voxel grid
            for i in range(nx):
                for j in range(ny):
                    line_str = ''.join(voxels[i, j].astype('<U1'))
                    f.write(f'{line_str}\n')

        # Display a finish message
        print(f"Voxelized mesh written to file '{output_file_path}'")


    def load_from_data_file(self, data_file_path: str):
        '''
        Load the voxel grid info from a binary archive file.
        '''
        # Load info from a numpy archive file, including bottom left position, voxel size,
        # voxel grid dimensions, and the voxel grid content
        data = np.load(data_file_path)

        self.voxel_grid_min = data['voxel_grid_min']
        self.voxel_size = data['voxel_size']
        self.voxel_grid_size = data['voxel_grid_size']
        self.voxels[:] = data['voxels']


    def save_to_data_file(self, output_file_path: str):
        '''
        Save the voxelized mesh as a binary archive file. This format is faster to load and
        preferred for grading.
        '''
        # Save info to a numpy archive file, including bottom left position, voxel size,
        # voxel grid dimensions, and the voxel grid content
        np.savez(
            output_file_path,
            voxel_grid_min=self.voxel_grid_min,
            voxel_size=self.voxel_size,
            voxel_grid_size=self.voxel_grid_size,
            voxels=self.voxels
        )

        # Display a finish message
        print(f"Voxelized mesh written to file '{output_file_path}'")


    def convert_to_tet_mesh(self) -> TetMesh:
        '''
        Convert the voxelized mesh to a tetrahedral mesh, where each voxel has 5 tet elements.
        '''
        # Read relevant variables from the class
        nx, ny, nz = self.voxel_grid_size
        dx = self.voxel_size
        voxels = self.voxels.astype(bool)
        grid_bottom_left = self.voxel_grid_min

        # Create a vertex grid and mark the used vertices
        vertices_grid = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
        slices = np.array([slice(None, -1), slice(1, None)])
        for num in range(8):
            bits = [num >> 2, (num >> 1) & 1, num & 1]
            vertices_grid[tuple(slices[bits].tolist())] |= voxels

        # Compute the vertex array
        vertex_indices = np.stack(np.nonzero(vertices_grid), axis=1)
        vertices = vertex_indices * dx + grid_bottom_left

        # Compute a vertex indexing grid for index retrieval
        vertices_id_grid = np.zeros_like(vertices_grid, dtype=np.int32)
        vertices_id_grid[vertices_grid] = np.arange(vertices.shape[0])

        # Get the vertex indices of all hex elements
        hex_elements = []
        for num in range(8):
            bits = [num >> 2, (num >> 1) & 1, num & 1]
            hex_elements.append(vertices_id_grid[tuple(slices[bits].tolist())][voxels])
        hex_elements = np.stack(hex_elements, axis=1)

        # Convert hex elements into tet elements
        tet_indices = [
            [0, 6, 4, 5],
            [0, 1, 3, 5],
            [0, 6, 5, 3],
            [6, 7, 5, 3],
            [0, 3, 2, 6],
        ]
        elements = hex_elements[:, tet_indices].reshape(-1, 4)

        # Return the tet mesh
        return TetMesh(vertices, elements)
