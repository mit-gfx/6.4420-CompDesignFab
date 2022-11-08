from scipy.signal import convolve2d
from PIL import Image

from typing import List, Tuple
from numpy import ndarray as array

import os
import argparse
import numpy as np


class TopologyOptimization:
    '''
    2D topology optimization algorithm
    '''
    def __init__(self, grid_size_x: int, grid_size_y: int, bc_type: str):
        '''
        Constructor function of the topology optimization class.

        Params:
            * `grid_size_x: int`    - width of the FEM grid
            * `grid_size_y: int`    - height of the FEM grid
            * `bc_type: str`        - boundary condition type: 'mbb' for MBB beam; 'cantilever' for
                                      cantilever beam; 'bridge' for bridge
        '''
        # Initialize the density, displacement, and force fields
        self.density = np.zeros((grid_size_x, grid_size_y))
        self.x = np.zeros((grid_size_x + 1, grid_size_y + 1, 2))
        self.f = np.zeros((grid_size_x + 1, grid_size_y + 1, 2))

        # Material parameters
        self.E = 1.0        # Young's modulus
        self.nu = 0.3       # Poisson's ratio

        # Initialize boundary conditions
        self.initialize_boundary_conditions(bc_type)

        # Initialize the stiffness matrix for each quad element
        self.initialize_element_stiffness_matrix(self.E, self.nu)


    def initialize_boundary_conditions(self, bc_type: str):
        '''
        Initialize boundary conditions (forces, fixed dimensions)

        Params:
            * `bc_type: str` - boundary condition type: 'mbb' for MBB beam; 'cantilever' for
                               cantilever beam; 'bridge' for bridge.
        '''
        # List of constrained dimensions. Each triplet has the following info:
        #   1. the X coordinate (0 to grid_size_x)
        #   2. the Y coordinate (0 to grid_size_y)
        #   3. the dimension (0 for x-direction and 1 for y-direction)
        bc: List[Tuple[int, int, int]] = []

        # Grid dimensions
        grid_size_x, grid_size_y = self.density.shape

        ## MBB beam
        if bc_type == 'mbb':
            # Force: downward, at the top center of the beam
            self.f[0, grid_size_y, 1] = -1

            # Fixed dimensions:
            #   1. center of the beam, x-direction
            #   2. bottom right corner of the beam, y-direction
            bc.extend([[0, j, 0] for j in range(grid_size_y + 1)])
            bc.append([grid_size_x, 0, 1])

        ## Cantilever beam
        elif bc_type == 'cantilever':
            # Force: downward, 1N, at the midpoint of the right side of the beam
            # --------
            # TODO: Your code here. You only need to set one element in the f array.
            self.f[0, 0, 0] = 0

            # Fixed dimensions: top-left and bottom-left corners, both dimensions
            # --------
            # TODO: Your code here. Use `bc.extend([[i, j, 0], [i, j, 1]])` to add both dimensions
            # of a node (i, j) to boundary constraints
            ...     # <--
            ...     # <--

        ## Bridge
        elif bc_type == 'bridge':
            # Force: downward, 1N, along the top side of the beam
            # --------
            # TODO: Your code here. You should set a slice of elements in the f array.
            self.f[0, 0, 0] = 0

            # Fixed dimensions: bottom-left and bottom-right corners, both dimensions
            # --------
            # TODO: Your code here.
            ...     # <--
            ...     # <--

        else:
            raise ValueError(f"Unrecognized boundary condition type '{bc_type}'")

        # Save the boundary conditions
        self.bc = tuple(zip(*bc))


    def initialize_element_stiffness_matrix(self, E: float, nu: float):
        '''
        Initialize the 8x8 stiffness matrix for each quad element. This is a replication of
        Yuanming's implementation in Taichi C++.

        Params:
            * `E: float`  - Young's modulus of the material
            * `nu: float` - Poisson's ratio of the material
        '''
        # Distinct values of the stiffness matrix Ke
        Ke_entries = np.array([
            1.0 / 2.0 - nu / 6.0, 1 / 8.0 + nu / 8.0,
            -1 / 4.0 - nu / 12.0, -1 / 8.0 + 3 * nu / 8.0,
            -1 / 4.0 + nu / 12.0, -1 / 8.0 - nu / 8.0,
            nu / 6.0, 1 / 8.0 - 3.0 * nu / 8.0,
        ])
        Ke_entries *= E / (1 - nu ** 2)

        # Indices to the Ke_entries array
        indices = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 7, 6, 5, 4, 3, 2],
            [2, 7, 0, 5, 6, 3, 4, 1], [3, 6, 5, 0, 7, 2, 1, 4],
            [4, 5, 6, 7, 0, 1, 2, 3], [5, 4, 3, 2, 1, 0, 7, 6],
            [6, 3, 4, 1, 2, 7, 0, 5], [7, 2, 1, 4, 3, 6, 5, 0],
        ])

        # Obtain the Ke matrix
        maps = [0, 1, 6, 7, 2, 3, 4, 5]
        self.Ke = np.ascontiguousarray(Ke_entries[indices[maps][:, maps]])


    def solve_fem(self, penalty: int, cg_max_iters: int=10 ** 5,
                  cg_tolerance: float=1e-4) -> array:
        '''
        Solve the FEM problem and compute sensitivities.

        Params:
            * `penalty: int`        - the penalty exponent
            * `cg_max_iters: int`   - Max. iterations of the Conjugate Gradient solver
                                      (default to 10^5)
            * `cg_tolerance: float` - Relative error tolerance of the CG solver (default to 1e-4)

        Return value:
            * `s: array` - (N, M), sentivity values
        '''
        # Save class members as local variables
        density = self.density      # Density field
        x = self.x                  # Displacement field
        f = self.f                  # Force field
        bc = self.bc                # Boundary conditions
        Ke = self.Ke                # Element stiffness matrix

        # Save NumPy functions as local variables
        dstack = np.dstack
        expand_dims = np.expand_dims
        zeros_like = np.zeros_like

        # Helper functions
        def apply_K(x: array) -> array:
            '''
            Compute the matrix-vector product of the stiffness matrix `K` and a vector `x`.
            Both input and output are shaped as vector fields rather than flattened vectors.

            Params:
                * `x: array` - (N + 1, M + 1, 2), the input vector field

            Return value:
                * `Kx: array` - (N, M, 2), the output vector field after left-multiplication by K
            '''
            # Unroll x into (N, M, 8, 1)
            x_unroll = dstack((x[:-1, :-1], x[:-1, 1:], x[1:, :-1], x[1:, 1:]))

            # Compute the unrolled Kx in the shape of (N, M, 8)
            Kx_unroll = (Ke @ expand_dims(x_unroll, 3)).squeeze(3)
            Kx_unroll *= expand_dims(density ** penalty, 2)

            # Compute the output Kx (N, M, 2)
            Kx = zeros_like(x)
            Kx[:-1, :-1] += Kx_unroll[:, :, :2]
            Kx[:-1, 1:] += Kx_unroll[:, :, 2: 4]
            Kx[1:, :-1] += Kx_unroll[:, :, 4: 6]
            Kx[1:, 1:] += Kx_unroll[:, :, 6:]

            return Kx

        def get_sensitivity(x: array) -> array:
            '''
            Compute the sensitivity matrix for a vector field x using the formula
            s = p * d^(p-1) * (xe^T * Ke * xe)

            Params:
                * `x: array` - (N + 1, M + 1, 2), the input vector field

            Return value:
                * `s: array` - (N, M), the sensitivity matrix
            '''
            # Unroll x into (N, M, 8)
            x_unroll = dstack((x[:-1, :-1], x[:-1, 1:], x[1:, :-1], x[1:, 1:]))

            # Compute Kx in the shape of (N, M, 8)
            Kx_unroll = (Ke @ expand_dims(x_unroll, 3)).squeeze(3)

            # Compute sensitivities
            s = (expand_dims(x_unroll, 2) @ expand_dims(Kx_unroll, 3)).squeeze((2, 3))
            s *= (density ** (penalty - 1)) * penalty
            s = np.maximum(s, 0)

            return s

        # Error tolerance value
        f_tol = np.abs(f).max() * cg_tolerance

        # Compute initial residual forces r = f - Kx and the conjugate vector p = r
        Kx = apply_K(x)
        r = f - Kx
        r[bc] = 0
        p = r.copy()

        # Conjugate gradient loop
        print('CG solver start')

        for it in range(cg_max_iters):
            # Compute Kp
            Kp = apply_K(p)
            Kp[bc] = 0

            # Compute the step size
            rr = r.ravel()
            r_dot = rr @ rr
            alpha = r_dot / (p.ravel() @ Kp.ravel() + 1e-100)

            # Update x and r
            x += alpha * p
            r -= alpha * Kp

            # Exit the loop if the error tolerance is met
            r_max = abs(r).max()
            if not it % 1000:
                print(f'  iter {it}, r = {r_max:.6f}')
            if r_max < f_tol:
                break

            # Compute the step size
            beta = (rr @ rr) / (r_dot + 1e-100)

            # Update p
            p[:] = r + beta * p

        # Check CG success
        if r_max < f_tol:
            print(f'CG converged in {it + 1} iterations')
        else:
            print(f'Warning - CG did not converge')

        # Compute sensitivities
        s = get_sensitivity(x)
        return s


    def optimality_criteria(self, s: array, fraction: float, change_limit: float):
        '''
        Apply optimality criteria to update the density field.

        Params:
            * `s: array`            - (N, M), the sensitivity values
            * `fraction: float`     - the target volume fraction
            * `change_limit: float` - maximum density change in each step
        '''
        # Save class members as local variables
        d = self.density        # Density field

        # Find lambda using binary search
        l, r = 0.0, 1e15

        while l * (1 + 1e-15) < r:
            # Compute the current lambda as the midpoint of the search interval
            m = (l + r) * 0.5

            # Scale the density field using the formula
            #   d' = d * sqrt(sensitivity / lambda)
            # --------
            # TODO: Your code here
            d_new = d       # <--

            # Clamp the new density values within the change limit
            # --------
            # TODO: Your code here.
            # HINT: use the 'np.clip' function for element-wise clamping. The usage is
            #   x_clipped = np.clip(x, lb, ub)
            # where `lb` and `ub` can be scalars or arrays. If they are arrays, their shapes should
            # match the shape of `x` exactly or within the broadcasting range (you don't have to
            # consider broadcasting here, though).
            d_new = d_new       # <--

            # Clamp the new density values within the valid density range
            d_new = np.clip(d_new, 1e-2, 1.0)

            # Halve the search interval according to the current total volume
            # --------
            # TODO: Your code here. Fill in the condition and two branches.
            # HINT: Think about this question - if the total volume under the current lambda is
            # smaller than the target volume, should we increase or decrease lambda?
            if True:        # <--
                r = 0.0     # <--
            else:
                l = 0.0     # <--

        # Update the density field using the new-found lambda (`l` rather than `m`)
        # --------
        # TODO: Your code here. Apply your solution from the lines above.
        d_new = d           # <--
        d_new = d_new       # <--
        d_new = np.clip(d_new, 1e-2, 1.0)
        self.density[:] = d_new


    def sensitivity_filtering(self, s: array, radius: float) -> array:
        '''
        Apply filtering to the sensitivity values.

        Params:
            * `s: array`      - (N, M), the sensitivity values
            * `radius: float` - the filter radius
        '''
        # Save class members as local variables
        d = self.density        # Density field

        # The essence of sensitivity filtering is smoothening the sensitivity field to eliminate
        # the checkerboard effect, under the control of a filtering radius. This is very similar to
        # a convolution operation in image processing that averages the adjacent pixel values of
        # an image (https://en.wikipedia.org/wiki/Kernel_(image_processing)).
        #
        # While we could just code the entire filtering process using nested for loops in C++
        # without worrying about any performance issue, it's hard to say the same with Python.
        # After completing the previous assignments, you probably noticed that a better strategy is
        # to utilize existing libraries like NumPy or SciPy to speed things up. This time you will
        # use the convolution operation in SciPy `scipy.signal.convolve2d` to compute the filtered
        # sensitivities. Please read the derivation below before proceeding with implementation.
        #
        # Sensitivity filtering is defined using the formula below:
        #
        #        SUM_j w_ij * s_j * nu_j    <-- numerator
        # s'_i = -----------------------
        #           SUM_j w_ij * nu_j       <-- denominator
        #
        # where w_ij = max(r - euclidean_dist(i, j), 0). For each pixel i, the corresponding j's
        # with non-zero weights will always be within a circle with a radius of r. Furthermore,
        # w_ij is neither affected by the sensitivity nor the density values at i or j, and stays
        # constant even though we shift i and j by the same amount. Thus, both the numerator and
        # the denominator of s'_i can be formulated as 2D convolutions.
        #   * For the numerator, the convolution kernel is all w_ij's within a radius of r.
        #     The image to convolve upon is an element-wise multiplication of the sensitivity field
        #     and the density field (s * nu).
        #   * For the denominator, we have the same convolution kernel. However, the image here is
        #     simply the density field.
        #
        # In the coding section below, you will implement the aforementioned algorithm by building
        # the convolution kernel and computing s' using the imported `convolve2d` function.

        # Construct the convolution kernel (the w matrix)
        # Create an empty w matrix of a proper size
        w_size = int(np.ceil(radius)) * 2 - 1
        w = np.zeros((w_size, w_size))

        # Compute the weights inside the w matrix using the formula
        # w[i, j] = max(0, dist((i, j), (ci, cj)))
        # where (ci, cj) is the coordinates of the center voxel
        # --------
        # TODO: Your code here.
        ci, cj = w_size // 2, w_size // 2
        for i in range(w_size):
            for j in range(w_size):
                dist = 0.0          # <--
                w[i, j] = 0.0       # <--

        # Compute convolution for the denominator
        # -------
        # TODO: Your code here.
        # HINT: The usage of `convolve2d` is
        #   a_conv = convolve2d(a, kernel, mode='same')
        # setting `mode='same'` adds zero padding and makes sure the returned array has the same
        # shape as the input array.
        d_conv = d      # <--

        # Compute convolution for the numerator
        # --------
        # TODO: Your code here.
        ds_conv = d     # <--

        # Compute filtered sensitivities
        # --------
        # TODO: Your code here.
        s_filtered = s      # <--

        return s_filtered


    def run(self, fraction: float, penalty: int=3, radius: float=1.5, threshold: float=0.005,
            change_limit: float=0.2):
        '''
        Run topology optimization.

        Params:
            * `fraction: float`     - target volume fraction
            * `penalty: int`        - penalty exponent (default to 3)
            * `radius: float`       - radius for sensitivity filtering (default to 1.5)
            * `threshold: float`    - termination threshold for density changes (default to 0.005)
            * `change_limit: float` - maximum density change in each OC step (default to 0.2)
        '''
        # Check input validity
        assert fraction > 0 and fraction < 1, 'target fraction must be between 0 and 1'
        assert penalty > 0, 'penalty exponent must be positive'
        assert radius >= 1, 'sensitivity filtering radius must be at least 1'

        # Save class members as local variables
        d = self.density        # Density field
        x = self.x              # Displacement field

        # Initialization
        d[:] = fraction
        x[:] = 0

        # Main loop
        d_last = d.copy()       # density from the last iteration
        it = 1                  # iteration counter

        while True:
            # Get sensitivities from the FEM solver
            s = self.solve_fem(penalty)

            # Perform sensitivity filtering
            s_filtered = self.sensitivity_filtering(s, radius)

            # Update the density field by optimality criteria
            self.optimality_criteria(s_filtered, fraction, change_limit)

            # Calculate the change in density
            d_change = abs(d_last - d).max()
            print(f'Iter {it}: density change = {d_change:.6g}')

            # Exit the loop if it is sufficiently small
            if d_change < threshold:
                print(f'Topology optimization finished in {it} iterations')
                break

            # Update density from the last iteration
            d_last[:] = d
            it += 1


    def save_image(self, file_name: str, scale=16):
        '''
        Save the current density grid to an image.

        Params:
            * `file_name: str` - file name of the image
        '''
        # Create an image using the density field
        img_arr = (255 * (1 - self.density.T)).astype(np.uint8)
        img_arr = img_arr.repeat(scale, axis=0).repeat(scale, axis=1)

        # Save the image
        Image.fromarray(img_arr[::-1]).save(file_name)


def main():
    '''
    Main routine.
    '''
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='2D topology optimization for alternative HW5')
    parser.add_argument('bc_type', metavar='TYPE', choices=['mbb', 'cantilever', 'bridge'],
                        help='Boundary condition type')
    parser.add_argument('-s', '--size', type=int, default=50, help='Grid size (x by x/2)')
    parser.add_argument('-f', '--fraction', metavar='FRAC', type=float, default=0.5,
                        help='Target volume fraction')
    parser.add_argument('-p', '--penalty', metavar='NUM', type=int, default=3,
                        help='SIMP penalty exponent')
    parser.add_argument('-r', '--radius', metavar='RAD', type=float, default=1.5,
                        help='Sensitivity filtering radius')
    parser.add_argument('-m', '--change-limit', metavar='LIM', type=float, default=0.2,
                        help='Max. change limit per OC step')

    args = parser.parse_args()

    # Run topology optimization
    opt = TopologyOptimization(args.size, args.size // 2, args.bc_type)
    opt.run(args.fraction, args.penalty, args.radius, change_limit=args.change_limit)

    # Create the result folder
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result_dir = os.path.join(ROOT_DIR, 'data', 'assignment5', 'results', 'part1')
    os.makedirs(result_dir, mode=0o775, exist_ok=True)

    # Save the final density field to the result folder
    opt.save_image(os.path.join(result_dir, f'topo_{args.bc_type}.png'))


if __name__ == '__main__':
    main()
