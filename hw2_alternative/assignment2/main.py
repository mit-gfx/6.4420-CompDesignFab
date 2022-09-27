from weights import linear_weights, bounded_biharmonic_weights

from typing import BinaryIO, Tuple, List

import os
import struct
import argparse
import igl
import numpy as np


def read_array_from_file(f: BinaryIO, dtype: np.dtype, dim: int=2) -> np.ndarray:
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


def read_array_from_list(l: List[str], dtype: np.dtype, dim: int=2, offset: int=0) -> np.ndarray:
    '''
    Read a Numpy array from a parsed list of strings.
    '''
    assert dim <= 2, 'Only support up to 2D matrices'

    # Read matrix shape
    shape = tuple(map(int, l[offset: offset + dim]))
    n_items = np.prod(shape)

    # Read matrix content
    offset += dim
    arr = np.array(list(map(dtype, l[offset: offset + n_items])), dtype=dtype)
    return arr.reshape(*shape)


def read_mesh_data(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Read the data of a tetrahedral mesh (vertices and element indices) from a binary file.
    '''
    with open(file_name.format('v'), 'r') as f:
        # Read the vertex array (V)
        V = []
        for line in f.readlines():
            V.append([float(c) for c in line.split()])
        V = np.array(V, dtype=np.float64)

    with open(file_name.format('t'), 'r') as f:
        # Read the elements array (T)
        T = []
        for line in f.readlines():
            T.append([float(c) for c in line.split()])
        T = np.array(T, dtype=np.int32)

    return V, T


def read_handles(file_name: str) -> np.ndarray:
    '''
    Read handles data from a text file.
    '''
    with open(file_name, 'r') as f:
        # Read data and parse it into a list of strings
        l = f.readline().strip().split()

        # Read handles data
        H = read_array_from_list(l, np.float64)
        ind = read_array_from_list(l, np.int64, 1, H.size + 2)
        handles = H[ind]

    return handles


def write_weights_binary(W: np.ndarray, file_name: str):
    '''
    Write a weight matrix to a binary file.
    '''
    with open(file_name, 'wb') as f:
        # Write matrix size
        f.write(struct.pack('qq', *W.shape))

        # Write matrix data
        f.write(W.tobytes())


def write_weights_text(W: np.ndarray, file_name: str):
    '''
    Write a weight matrix to a text file
    '''
    with open(file_name, 'w') as f:
        for row in W:
            f.write(''.join([' '.join([f'{v:.8g}' for v in row]), '\n']))


def main():
    '''
    Main routine.
    '''
    # Command line argument parser
    parser = argparse.ArgumentParser(description='HW2 - linear and BBW weight calculator')
    parser.add_argument('mesh_name', help="Mesh name (must ends with '_voxel' for voxelized mesh)")
    parser.add_argument('-c', '--handles', metavar='FILE', default='', help='Handles data file')

    # Process arguments
    args = parser.parse_args()

    # Printe welcome message
    mesh_name = args.mesh_name
    print('Alternative HW2')
    print(f"Running for test case '{mesh_name}'")

    # Read tetrahedral mesh info
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file_name = os.path.join(ROOT_DIR, 'data/assignment2/BBW/results', f'{mesh_name}_tetmesh_{{}}.txt')
    V, T = read_mesh_data(data_file_name)

    # Read handles
    handles_file_name = args.handles if args.handles else \
        os.path.join(ROOT_DIR, 'data/assignment2/BBW', f'{mesh_name}_handles.txt')
    C = read_handles(handles_file_name)

    # Compute linear weights
    W_lin = linear_weights(V, C)

    # Compute bounded biharmonic weights
    W_bbw = bounded_biharmonic_weights(V, T, C)

    # Compute LBS matrix for BBW
    M_bbw = igl.lbs_matrix(V, W_bbw)

    # Write linear weights and BBW to binary files
    output_file_name = os.path.join(ROOT_DIR, 'data/assignment2/BBW/results', f'{args.mesh_name}_{{}}_weights.dat')
    lin_output_file_name = output_file_name.format('lin')
    write_weights_binary(W_lin, lin_output_file_name)
    print(f"Linear weights saved to file '{lin_output_file_name}'")

    bbw_output_file_name = output_file_name.format('bbw')
    write_weights_binary(W_bbw, bbw_output_file_name)
    print(f"BBW saved to file '{bbw_output_file_name}'")

    # Write LBS matrix for BBW to text file
    lbs_output_file_name = ''.join([os.path.splitext(bbw_output_file_name)[0], '.txt'])
    write_weights_text(M_bbw, lbs_output_file_name)
    print(f"LBS matrix saved to file '{lbs_output_file_name}'")


if __name__ == '__main__':
    main()
