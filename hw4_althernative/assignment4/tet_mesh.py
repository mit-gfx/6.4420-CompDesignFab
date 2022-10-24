from numpy import ndarray as array

from typing import BinaryIO

import struct
import numpy as np


def read_array_from_file(f: BinaryIO, dtype: np.dtype, dim: int=2) -> array:
    '''
    Read a Numpy array from the current position of the input file.
    '''
    assert dim <= 2, 'Only support up to 2D matrices'

    # Read matrix shape
    shape = struct.unpack('q' * dim, f.read(8 * dim))
    n_items = np.prod(shape)

    # Read matrix content
    arr = np.frombuffer(f.read(dtype.itemsize * n_items), dtype=dtype, count=n_items)
    return arr.reshape(*shape)


class TetMesh:
    '''
    Class of a tetrahedral mesh.
    '''
    def __init__(self, vertices: array, elements: array):
        # Check the shape of vertex and element arrays
        assert vertices.ndim == 2 and vertices.shape[1] == 3, \
               f'Invalid shape of the vertex array: {vertices.shape}, should be (N, 3)'
        assert elements.ndim == 2 and elements.shape[1] == 4, \
               f'Invalid shape of the element array: {elements.shape}, should be (M, 4)'

        # Save the tet mesh data
        self.V = vertices
        self.T = elements

    @property
    def vertices(self) -> array:
        '''
        Return the vertex array of the tet mesh.
        '''
        return self.V

    @property
    def elements(self) -> array:
        '''
        Return the element array of the tet mesh.
        '''
        return self.T

    def write_to_file(self, file_name: str, invert_normal: bool=False):
        '''
        Write the tet mesh in STL format.
        '''
        # Store mesh data into local variables (eliminates lookup overhead)
        V, T = self.V, self.T

        # Vertex indices of four triangles in a tet element
        tri_indices = [[0, 1, 2], [1, 3, 2], [1, 0, 3], [0, 2, 3]]

        # Optionally invert normal
        if invert_normal:
            tri_indices = [[i, k, j] for i, j, k in tri_indices]

        with open(file_name, 'w') as f:
            # Write header
            f.write('solid tet_mesh\n')

            # Write tets
            for tet in T:
                for indices in tri_indices:
                    # Get the current triangle
                    tri = V[tet[indices]]

                    # Write the vertices of the current triangle
                    f.write(
                        f'facet normal 0 0 0\n'
                        f'    outer loop\n'
                        f'        vertex {tri[0, 0]} {tri[0, 1]} {tri[0, 2]}\n'
                        f'        vertex {tri[1, 0]} {tri[1, 1]} {tri[1, 2]}\n'
                        f'        vertex {tri[2, 0]} {tri[2, 1]} {tri[2, 2]}\n'
                        f'    endloop\n'
                        f'endfacet\n'
                    )

            # Write footer
            f.write('endsolid')


def tet_mesh_cuboid(nx: int, ny: int, nz: int, cube_size: float) -> TetMesh:
    '''
    Generate the tet mesh of a cuboid. The cuboid has `(nx, ny, nz)` vertices and consists of
    `(nx - 1, ny - 1, nz - 1)` cubes, each with a edge length of `cube_size`.
    '''
    # The grid size must be at least 2 in each dimension
    assert min(nx, ny, nz) >= 2, 'There should be at least 2 vertices in each dimension'

    # Compute vertices
    vertices = np.stack(np.mgrid[:nx, :ny, :nz], axis=3).reshape(-1, 3) * cube_size

    # Predefine the vertex indices of tets in each cube
    tet_indices = np.array([
        [0, 6, 4, 5],
        [0, 1, 3, 5],
        [0, 6, 5, 3],
        [6, 7, 5, 3],
        [0, 3, 2, 6],
    ])

    # Transform vertex indices in a tet into global index offsets
    flatten_indices = lambda a: np.sum(a * np.array([ny * nz, nz, 1]), axis=-1)
    tet_indices_bits = (tet_indices[:, :, None] & np.array([4, 2, 1])) >> np.arange(3)[::-1]
    offsets = flatten_indices(tet_indices_bits)

    # Compute elements
    cube_origins = np.stack(np.mgrid[:(nx - 1), :(ny - 1), :(nz - 1)], axis=3).reshape(-1, 3)
    cube_origin_offsets = flatten_indices(cube_origins)
    elements = (cube_origin_offsets[:, None, None] + offsets).reshape(-1, 4)

    # Construct and return a tet mesh
    return TetMesh(vertices, elements)


def tet_mesh_from_file(file_name: str, max_size: float=0.0) -> TetMesh:
    '''
    Read a tet mesh from binary file (backward compatibility with HW3). Optionally scale the mesh
    so that its longest dimension is equal to `max_size`.
    '''
    # Read mesh data
    with open(file_name, 'rb') as f:
        # Read the vertex array (V)
        V = read_array_from_file(f, np.dtype(np.float64))

        # Read the elements array (T)
        T = read_array_from_file(f, np.dtype(np.int32))

    # Optionally scale the mesh
    if max_size > 0.0:
        bbox_size = V.max(axis=0) - V.min(axis=0)
        V = V * (max_size / bbox_size.max())

    # Construct and return a tet mesh
    return TetMesh(V, T)
