from scipy.sparse import spmatrix, csc_matrix
from numpy import ndarray as array

from typing import List, Tuple, Callable

import igl
import time
import numpy as np
import scipy as sp


def timer(func: Callable[..., array]) -> Callable[..., array]:
    '''
    Function wrapper for timing.
    '''
    # Define the wrapper function
    def timed_func(*args, **kwargs) -> array:
        # Time the wrapped function
        t_start = time.time()
        ret = func(*args, **kwargs)
        t_elapsed = time.time() - t_start

        # Print the time usage
        func_name = func.__name__.replace('_', ' ')
        print(f"Finished computing {func_name} in {t_elapsed:.3f}s")

        return ret

    return timed_func


@timer
def linear_weights(V: array, C: array) -> array:
    '''
    Compute linear weights for vertices on a tetrahedral mesh.

    Params:
        * `V: array` - (Nx3) vertex positions, N = #vertices
        * `C: array` - (Hx3) handle positions, H = #handles

    Return value:
        * `W: array` - (NxH) linear weights
    '''
    # Compute inverse pairwise distances between vertices and handles
    N, H = V.shape[0], C.shape[0]
    W = np.zeros((N, H))

    eps = 1e-14                             # Epsilon
    float_max = np.finfo(np.float64).max    # Infinity

    ## Loop over all vertices and handles
    for i in range(N):
        for j in range(H):

            # Compute the distance between the i-th vertex and the j-th handle
            # --------
            # TODO: Your code here. The Euclidean distance between two 3D points can be computed
            # using the `np.linalg.norm` function. Documentation:
            # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
            d = 0.0     # <-- TODO

            # Compute the inverse distance
            # --------
            # TODO: Your code here. Assign the correct values to W[i, j] in two conditions. Note
            # that you should use variable `float_max` to represent infinity.
            if d < eps:
                ...     # <-- TODO
            else:
                ...     # <-- TODO

    # Normalize weights
    W /= W.sum(axis=1, keepdims=True)
    return W


def cotangent_laplacian(V: array, T: array) -> spmatrix:
    '''
    Compute the cotangent Laplacian matrix for a tetrahedral mesh.

    Params:
        * `V: array` - (Nx3) vertex positions, N = #vertices
        * `T: array` - (Tx4) vertex indices of tet elements, T = #elements

    Return value:
        * `L: spmatrix` - (NxN) Laplacian matrix
    '''
    # Compute edge lengths
    # l: array - (Tx6) edge lengths of tet elements
    # l[i, j] stores the length of edge j in tet element i
    l = igl.edge_lengths(V, T)

    # Compute dihedral angles
    # theta: array - (Tx6) dihedral angles of tet elements
    # theta[i, j] stores the dihedral angle of two triangle faces sharing edge j in tet element i
    # cos_theta[i, j] is the cosine value of theta[i, j]
    theta, cos_theta = igl.dihedral_angles(V, T)

    # Compute weighted cotangents
    # --------
    # TODO: Your code here. Compute the cotangent weight contribution from each edge. These are
    # merely edge-wise components of the Laplacian matrix and you don't have to sum anything up.
    # The construction of the Lapacian matrix is handled later.
    cot_weights = np.zeros_like(theta)      # <--

    # Predefine edge vertex indices in each tet
    # The indices of four vertices in a tet are defined using 0, 1, 2, 3
    # edges[j] stores the pair of indices associated with edge j
    edges = np.array([[1, 2], [2, 0], [0, 1], [3, 0], [3, 1], [3, 2]])

    # Compute elements in the Laplacian matrix in the form of triplets. Each triplet contains the
    # following info:
    #   * r: int, the row index of the element
    #   * c: int, the column index of the element
    #   * val: float, the value of the element
    # Here, we allow multiple triplets for the same element (r, c). The values in these triplets
    # are automatically summed up when constructing the sparse Laplacian matrix.
    num_tets = len(T)
    triplets: List[Tuple[int, int, float]] = []

    # Loop over all tets and edges
    # --------
    # TODO: Your code here. Complete the nested loop.
    for i in []:        # <-- TODO
        for j in []:    # <-- TODO

            # Get the indices of vertices associated with edge j in the entire mesh
            # --------
            # TODO: Your code here. Think about how to obtain vertex indices in the mesh using
            # indices in a tet element.
            src = 0     # <-- TODO
            dst = 0     # <-- TODO

            # Fetch the current weighted cotangent
            weight = cot_weights[i, j]

            # Construct triplets for value contributions to Laplacian matrix elements
            # -------
            # TODO: Your code here. First read the comments above the declaration of `triplets`,
            # then complete the following:
            #   - Add a triplet for the value contribution to L[src, dst]
            #   - Add a triplet for the value contribution to L[dst, src]
            #   - Add a triplet for the value contribution to L[src, src]
            #   - Add a triplet for the value contribution to L[dst, dst]
            ...         # <-- TODO
            ...         # <-- TODO
            ...         # <-- TODO
            ...         # <-- TODO

    # Construct the Laplacian matrix
    if triplets:
        row_inds, col_inds, data = tuple(zip(*triplets))
        L = csc_matrix((data, (row_inds, col_inds)), dtype=np.float64)
    else:
        L = csc_matrix((0, 0))

    return L


def quadratic_optimization(Q: spmatrix, b: array, bc: array, num_iters: int=100) -> array:
    '''
    Solve the quadratic optimization problem where the result minimizes `0.5 * W^T * Q * W`.

    Params:
        * `Q: spmatrix` - (NxN) coefficient matrix, N = #vertices
        * `b: array` - (Hx1) indices of constrained vertices
        * `bc: array` - (HxH) weights at constrained vertices w.r.t. each handle

    Return value:
        * `W: array` - (NxH) optimization result
    '''
    # Get #vertices and #handles
    N, H = Q.shape[0], bc.shape[1]

    # Functions for creating placeholder variables
    empty_spmatrix = lambda: csc_matrix((0, N))
    empty_vector = lambda: np.zeros((0, 1))

    # Placeholder variables
    B = np.zeros((N, H))
    Bi = np.zeros(N)
    Aeq, Aieq = empty_spmatrix(), empty_spmatrix()
    Beq, Bieq = empty_vector(), empty_vector()

    # Variable bounds
    lx = np.zeros(N)
    ux = np.ones(N)

    # Initial solve for preconditioning (iter 0)
    _, W = igl.min_quad_with_fixed(Q, B, b, bc, Aeq, Beq, True)

    # Compute weights for all handles
    for i in range(H):
        print(f'BBW: computing weights for handle {i + 1} / {H} ...')

        # Get the constraints and initial guess for handle i
        bci = np.ascontiguousarray(bc[:, i])
        Wi = np.ascontiguousarray(W[:, i])

        # Solve the quadratic optimization problem for the current handle
        _, Wi = igl.active_set(Q, Bi, b, bci, Aeq, Beq, Aieq, Bieq, lx, ux, Wi,
                               Auu_pd=True, max_iter=num_iters - 1)

        # Write results to the corresponding column of W
        W[:, i] = Wi

    return W


@timer
def bounded_biharmonic_weights(V: array, T: array, C: array) -> array:
    '''
    Compute bounded biharmonic weights for vertices on a tetrahedral mesh.

    Params:
        * `V: array` - (Nx3) vertex positions, N = #vertices
        * `T: array` - (Tx4) vertex indices of tet elements, T = #elements
        * `C: array` - (Hx3) handle positions, H = #handles

    Return value:
        * `W: array` - (NxH) BBW matrix
    '''
    # Compute the Laplacian matrix
    L = cotangent_laplacian(V, T)

    # Compute the mass matrix and its inverse
    M = igl.massmatrix(V, T)
    M_inv = sp.sparse.diags(1.0 / M.diagonal())

    # Compute the Q matrix
    # --------
    # TODO: Your code here. Derive Q according to Equation (4) in the write-up.
    # Hint: Muplication of matrix A and B can be written as `A @ B` or `np.dot(A, B)`
    Q = ...     # <-- TODO

    # Get boundary conditions
    P = np.arange(len(C), dtype=np.int32)
    BE = np.zeros((0, 2), dtype=np.int32)
    CE = np.zeros((0, 2), dtype=np.int32)
    _, b, bc = igl.boundary_conditions(V, T, C, P, BE, CE)

    # Perform quadratic optimization
    # --------
    # TODO: Your code here. Invoke the `quadratic_optimization` function. All required
    # arguments are available above. Please ignore the `num_iters` argument and leave it
    # to the default value.
    W = np.zeros((len(V), len(C)))      # <-- TODO

    # Normalize weights
    W /= W.sum(axis=1, keepdims=True)
    return W

