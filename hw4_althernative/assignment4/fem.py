from tet_mesh import TetMesh
from material import Material

from numpy import ndarray as array
from scipy.sparse import csc_matrix, spmatrix
from scipy.sparse.linalg import cg
from functools import partial
from typing import Type, List

import numpy as np

# CG solver
conjugate_gradient = partial(cg, tol=1e-5)


class StaticFEM:
    '''
    Static analysis using the finite element method (FEM).
    '''
    def __init__(self, mesh: TetMesh, material: Type[Material]):
        '''
        The constructor takes as input a tet mesh (`mesh`) and a material model (`material`).
        '''
        # Extract vertices and tet elements from the input argument
        V, T = mesh.vertices, mesh.elements

        # Constants
        dim, dim2 = V.shape[1], V.shape[1] ** 2

        # Recall that F = Ds * Dm^(-1), where
        #   - Ds = [x2 - x1, x3 - x1, x4 - x1] is the bases after deformation
        #   - Dm = [X2 - X1, X3 - X1, X4 - X1] is the bases before deformation
        # Here we precompute Dm^(-1) for all tet elements
        tet_vertices = V[T.ravel()].reshape(-1, T.shape[1], dim)
        Dm = (tet_vertices[:, 1:] - tet_vertices[:, [0]]).transpose(0, 2, 1)
        Dm_inv = np.linalg.inv(Dm)

        # Precompute dF/dx for all tet elements
        # The formula is dF/dx = d(Ds * Dm^(-1))/dx = d(Ds)/dx * Dm^(-1)
        ## Compute dF/dx1
        dF_dx1 = np.eye(dim) * -Dm_inv.sum(axis=1).reshape(-1, dim, 1, 1)
        dF_dx1 = dF_dx1.reshape(-1, dim2, dim)

        ## Compute dF/d(x2, x3, ..., xm), where m is the simplex size (4 for a tet mesh)
        dF_dx_others = np.eye(dim) * Dm_inv.transpose(0, 2, 1).reshape(-1, dim, dim, 1, 1)
        dF_dx_others = dF_dx_others.transpose(0, 1, 3, 2, 4).reshape(-1, dim2, dim2)
        dF_dx = np.dstack((dF_dx1, dF_dx_others))

        # Precompute tet volumes
        volumes = np.abs(np.linalg.det(Dm)) * (1 / 6)

        # Save the input arguments
        self.mesh = mesh
        self.material = material

        # Save the precomputed values
        self.Dm_inv = Dm_inv
        self.dF_dx = dF_dx
        self.volumes = volumes

    def elastic_force(self, vertices: array) -> array:
        '''
        Compute the internal elastic force at current vertex positions.

        Params:
            * `vertices: array` - (Nxd) current vertex positions, N = #vertices, d = #dimensions

        Return value:
            * `f: spmatrix` - (Nxd) the elastic force matrix
        '''
        # Store class member data into local variables
        V = self.mesh.vertices      # (Nx3), N = #vertices
        T = self.mesh.elements      # (Tx4), T = #elements

        material = self.material    # Material model

        Dm_inv = self.Dm_inv        # (Tx3x3), Dm^(-1)
        dF_dx = self.dF_dx          # (Tx9x12), dF/dx
        volumes = self.volumes      # (T), volumes of tet elements

        # Check input validity
        assert vertices.shape == V.shape, \
            f'The passed-in vertices must match the shape of tet vertices. Expected {V.shape} ' \
            f'but got {vertices.shape} instead'

        # Constants
        num_tets = T.shape[0]
        dim = V.shape[1]

        # Initialize the force matrix
        f = np.zeros_like(V)

        # Iterate over all tet elements
        for t in range(num_tets):
            # Compute the deformation gradient F
            tet_indices = T[t]
            tet_vertices = vertices[tet_indices]
            F = (tet_vertices[1:] - tet_vertices[0]).T @ Dm_inv[t]

            # Compute dE/dx
            P = material.stress_tensor(F)
            dE_dx = volumes[t] * P.T.ravel() @ dF_dx[t]

            # Add the nodal forces to the force matrix f
            f[tet_indices] -= dE_dx.reshape(-1, dim)

        # Suppress negative zeroes
        f = np.where(np.abs(f) < 1e-8, 0, f)
        return f

    def stiffness_matrix(self, vertices: array) -> spmatrix:
        '''
        Compute the stiffness matrix given the current vertex positions.

        Params:
            * `vertices: array` - (Nxd) current vertex positions, N = #vertices, d = #dimensions

        Return value:
            * `K: spmatrix` - (Nd x Nd) the sparse stiffness matrix
        '''
        # Store class member data into local variables
        V = self.mesh.vertices      # (Nx3), N = #vertices
        T = self.mesh.elements      # (Tx4), T = #elements

        material = self.material    # Material model

        Dm_inv = self.Dm_inv        # (Tx3x3), Dm^(-1)
        dF_dx = self.dF_dx          # (Tx9x12), dF/dx
        volumes = self.volumes      # (T), volumes of tet elements

        # Check input validity
        assert vertices.shape == V.shape, \
            f'The passed-in vertices must match the shape of tet vertices. Expected {V.shape} ' \
            f'but got {vertices.shape} instead'

        # Constants
        num_vertices, num_tets = V.shape[0], T.shape[0]
        dim, simplex_size = V.shape[1], T.shape[1]

        # Initialize an empty list of triplets for the output sparse matrix
        triplets: List[int, int, float] = [[0, 0, 0.0]]

        # Iterate over all tet elements
        for t in range(num_tets):
            # Compute the deformation gradient F
            # Formula: F = Ds * Dm^(-1), where
            #   - Dm = [x2 - x1, x3 - x1, x4 - x1] is the bases after deformation
            #   - Ds = [X2 - X1, X3 - X1, X4 - X1] is the bases before deformation
            # We have precomputed Dm^(-1) in the constructor, now we only care about Ds
            # --------
            # TODO: Your code here. The steps for computing F are as follows:
            #   1. Get the vertices of the current tet element, dubbed as `tet_vertices`
            #   2. Compute Ds
            #   3. Compute F using the formula
            # HINT:
            #   - Consider how to index or slice the matrices to perform vectorized calculation,
            #     i.e., conducting multiple concurrent math operations using a single Python
            #     operator. For example,
            #       * The addtion of two vectors `a` and `b` is simply written as `c = a + b`,
            #         where c[i] = a[i] + b[i]
            #       * The row-wise addtion of matrix `A` and vector `b` is written as `C = A + b`,
            #         where C[i, j] = a[i, j] + b[j]
            #   - Use `A.T` or `np.transpose(A)` to transpose a 2D array A
            #   - Use the `@` operator for matrix multiplication
            tet_vertices = vertices[T[t]]
            Ds = ...    # <--
            F = ...     # <--

            # Compute tet element t's contribution to the stiffness matrix K
            # Formula: Kt = d^2(Et)/d(xt)^2, where
            #   - Et (scalar) is the strain energy of t
            #   - xt (4x3) is the vertex positions of t, represented by the varaible `tet_vertices`
            #   - Kt (12x12) is the contribtion from t to K
            # If we apply the chain rule (the subscript t is omitted for simplicity):
            #   dE/dx = volume * dW/dx
            #         = volume * dW/dF * dF/dx
            #         = volume * P * dF/dx
            #   d^E/dx^2 = volume * d(P * dF/dx)/dx
            #            = volume * dP/dx * dF/dx            (d^2F/dx^2 is zero, can you see why?)
            #            = volume * (dP/dF * dF/dx) * dF/dx

            # Let's first prepare the operands
            # --------
            # TODO: Your code here. Get the volume of t, dP/dF and dF/dx.
            vol = ...       # <--
            dP_dF = ...     # <--
            dF_dxt = ...    # <--

            # Compute Kt using the formula above
            # --------
            # TODO: Your code here. Compute Kt in two steps:
            #   1. Compute dP/dx = dP/dF * dF/dx
            #   2. Compute Kt = volume * dP/dx * dF/dx
            # HINT:
            #   - Use the `@` operator for matrix multiplication
            #   - Some matrix should be transposed in Step 2
            dP_dxt = ...    # <--
            Kt = np.zeros((simplex_size * dim, simplex_size * dim))     # <--

            # Suppress negative zeroes
            Kt = np.where(np.abs(Kt) < 1e-8, 0, Kt)

            # Assign the elements of Kt to their right positions in the global matrix K
            #
            # Just like HW3, K is represented by a list of triplets (row_ind, col_ind, val) which
            # refers to the row index, column index, and the value of each element. Note that
            # triplets associated with the same element index (row_ind, col_ind) are summed up
            # automatically when creating the sparse matrix.
            #
            # Therefore, we can simply create m^2 triplets for every Kt assuming that Kt is a mxm
            # matrix. To avoid writing deep nested loops in Python (very slow), let's adopt
            # the following strategy:
            #   1. Figure out an index mapping from Kt to K, dubbed as `index_map`. `index_map` is
            #      a array of length m and maps the i-th row/column in Kt to the index_map[i]-th
            #      row/column in K. Mathematically, m = `simplex_size * dim`.
            #   2. Iterate over all elements in Kt to create cooresponding triplets for K. This
            #      step only requires a double nested loop.

            # Construct the index mapping from Kt to K
            # --------
            # TODO: Your code here. Implement step 1.
            m = Kt.shape[0]
            index_map = np.zeros(m, dtype=np.int64)
            for i in []:        # <--
                for j in []:    # <--
                    ...         # <--

            # Create m^2 triplets for K
            # --------
            # TODO: Your code here. Implement step 2.
            for i in []:        # <--
                for j in []:    # <--
                    ...         # <--

        # Construct the sparse matrix K
        row_inds, col_inds, vals = list(zip(*triplets))
        K = csc_matrix((vals, (row_inds, col_inds)),
                       shape=(num_vertices * dim, num_vertices * dim))
        return K

    def solve_linear(self, external_forces: array, boundary_conditions: array) -> array:
        '''
        Solve mesh deformation from the linear equations K * U = f_ext.

        Params:
            * `external_forces: array`     - (Nxd), external forces, N = #vertices, d = #dimensions
            * `boundary_conditions: array` - (N), a boolean mask array over the vertices. Those
                masked by True are assumed to be fixed and excluded from the solver.

        Return Value:
            * `U: array` - (Nxd), deformation matrix
        '''
        # Store class member data into local variables
        V = self.mesh.vertices      # (Nx3), N = #vertices
        dim = V.shape[1]            # d = #dimensions

        # Compute the stiffness matrix
        K_full = self.stiffness_matrix(V)

        # Apply boundary conditions by removing fixed points
        active_mask = (~boundary_conditions).repeat(dim)    # The mask of unconstrained coordinates
        active_indices = np.nonzero(active_mask)[0]         # Indices of unconstrained coordinates

        K = K_full[active_indices][:, active_indices]       # The actual stiffness matrix we use
        f_ext = external_forces.ravel()[active_mask]        # The actual external forces we use

        # Solve the linear equation using the conjugate gradient method
        U, stat = conjugate_gradient(K, f_ext)
        if stat != 0:
            print('Warning - CG solver failed with status', stat)
        else:
            print('CG solver finished')

        # Obtain the full-size deformation matrix
        U_full = np.zeros_like(V)
        U_full.ravel()[active_mask] = U
        return U_full

    def solve_newton(self, external_forces: array, boundary_conditions: array,
                     max_iters: int=1000, max_line_search_iters: int=20) -> array:
        '''
        Solve mesh deformation using Newton's method. Instead of solving K * U = f_ext, Newton's
        method iteratively solves the following equation:
            `K(Ui) * (U - Ui) = f_ext + f_el`
        where Ui is the deformation matrix in the previous iteration, and f_el is the elastic forces
        in the previous iteration.

        Params:
            * `external_forces: array`     - (Nxd), external forces, N = #vertices, d = #dimensions
            * `boundary_conditions: array` - (N), a boolean mask array over the vertices. Those
                masked by True are assumed to be fixed and excluded from the solver.
            * `max_iters: int`             - maximum iterations for the Newton's method
            * `max_line_search_iters: int` - maximum iterations of the line search algorithm

        Return Value:
            * `U: array` - (Nxd), deformation matrix
        '''
        # Check input validity
        assert max_iters >= 1 and max_line_search_iters >= 1, \
            'The iteration budgets must be at least 1'

        # Store class member data into local variables
        V = self.mesh.vertices     # (Nx3), N = #vertices
        dim = V.shape[1]           # d = #dimensions

        # Apply boundary conditions to external forces
        active_mask = (~boundary_conditions).repeat(dim)    # The mask of unconstrained coordinates
        active_indices = np.nonzero(active_mask)[0]         # Indices of unconstrained coordinates
        f_ext = external_forces.ravel()[active_mask]        # The reduced external force vector

        # Initialize the stiffness matrix
        K_full = self.stiffness_matrix(V)
        K = K_full[active_indices][:, active_indices]       # The reduced stiffness matrix

        # Initialize the elastic force matrix and the solution
        f_el = np.zeros_like(f_ext)
        Ui = np.zeros_like(f_ext)

        # Our solver has a predefined budget of `max_iters` iterations
        for it in range(max_iters):

            # Solve the linear equation
            #    K(Ui) * (U - Ui) = f_ext + f_el
            # for the update direction of U. Here, we also refer to the right hand side as the
            # residual forces `f_res`:
            #    f_res = f_ext + f_el
            # The update direction is dubbed as `dU`, where dU = U - Ui.
            # --------
            # TODO: Your code here. Compute f_res and dU.
            # HINT: use the function `conjugate_gradient` to solve linear equations A * x = b.
            # The usage is `x, stat = conjugate_gradient(A, b)`, where `stat` indicates the
            # status of the CG solver.
            f_res = np.zeros_like(f_ext)            # <--
            dU, stat = np.zeros_like(f_ext), 0      # <--
            if stat != 0:
                print('Warning - CG solver failed with status', stat)

            # Perform line search to find a feasible step size for updating Ui
            #
            # The idea behind line search is very simple. We start from an initial step size `l`
            # and test if U = Ui + dU * l is a better solution to K * U = f_ext. A better solution
            # means that the residual forces at U are smaller than Ui (smaller residual error).
            #
            # We have given you the skeleton code of the line search algorithm, and now you will
            # fill in the blank lines according to the comments

            # Initialize line search step size
            l = 1.0

            # Precompute the norm of `f_res` as the residual error
            f_res_norm = np.linalg.norm(f_res)

            # Line search algorithm loop
            for _ in range(max_line_search_iters):

                # Compute the current U using the step size l
                # --------
                # TODO: Your code here. Compute U.
                U = np.zeros_like(Ui)       # <--

                # Get the vertex coordinates `V_l` given the deformation matrix U
                V_l = V.copy()
                V_l.ravel()[active_mask] += U

                # Computing the residual forces at U breaks down into two steps:
                #   1. Compute the reduced elastic forces f_el
                #   2. Compute f_res
                # --------
                # TODO: Your code here. Compute f_el.
                f_el_full = np.zeros_like(V)    # <--
                f_el[:] = f_el_full.ravel()[active_mask]

                # --------
                # TODO: Your code here. Compute f_res at step size l.
                f_res_l = np.zeros_like(f_ext)      # <--

                # Exit the loop if `f_res_l` has a smaller norm than `f_res`
                # --------
                # TODO: Your code here. Implement the if-condition.
                # HINT:
                #   - The `np.linalg.norm` function computes the norm of a vector.
                #   - The norm of f_res has been precomputed and stored in `f_res_norm`
                f_res_l_norm = 0.0      # <--
                if True:                # <--
                    break

                # Halve the step size
                l *= 0.5

            # Print the residual error after line search
            print(f'Iteration {it + 1}: residual error = {f_res_l_norm}')

            # Exit the loop if the residual error is sufficiently small
            if f_res_l_norm < 1e-4:
                print(f"Newton's method converged in {it + 1} iterations")
                break

            # Update Ui using the U value after line search
            Ui[:] = U

            # Update the reduced stiffness matrix at Ui
            # --------
            # TODO: Your code here.
            # HINT: You will need the deformed vertex positions to compute the stiffness matrix.
            # However, it's actually ready in an existing variable. Which one is it?
            K_full = csc_matrix((V.shape[0] * dim, V.shape[0] * dim))   # <--
            K = K_full[active_indices][:, active_indices]

        # Obtain the full-size deformation matrix U
        U_full = np.zeros_like(V)
        U_full.ravel()[active_mask] = U
        return U_full
