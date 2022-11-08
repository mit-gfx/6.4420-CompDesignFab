import numpy as np


def pareto_front(points: np.ndarray) -> np.ndarray:
    '''
    Compute the Pareto front of a set of 2D points. Minimization is assumed for all properties.
    '''
    # Check input validity
    assert points.ndim == 2 and points.shape[-1] == 2 and points.shape[0] > 0, \
        'The input array must represent a set of 2D points'

    # Sort the points by both properties
    # --------
    # TODO: Your code here.
    # HINT:
    #   1. The function you will use here is called `np.lexsort`, whose documentation is at
    #      https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html. `np.lexsort`
    #      supports sorting by multiple columns in a user-specified order. You can decide whichever
    #      order you want here.
    #   2. Indexing a NumPy array using the output of `np.lexsort` returns the sorted array.
    points = points     # <--

    pareto_indices = [0]        # List of indices to Pareto-optimal points (in the sorted array)
    pareto_x = points[0, 0]     # X value of the last Pareto-optimal point
    pareto_y = points[0, 1]     # Y value of the last Pareto-optimal point

    # Traverse the sorted array to figure out Pareto-optimal points
    for i in range(points.shape[0]):

        # Add this point to the Pareto front if it isn't dominated by the last Pareto-optimal point
        # --------
        # TODO: You code here.
        if False:       # <--
            ...         # <--

            # Update the last Pareto-optimal point using this point
            # --------
            # TODO: Your code here.
            pareto_x = ...      # <--
            pareto_y = ...      # <--

    # Return the Pareto front
    pareto_front = points[pareto_indices]

    return pareto_front


# Unit test
if __name__ == '__main__':
    # Set number of points
    num_points = 1000

    # Generate random numbers
    points = np.random.uniform(size=(num_points, 2))

    # Get the Pareto front by brute-force
    dom = (points[:, None] >= points[None]).all(axis=2)
    dom[np.arange(num_points), np.arange(num_points)] = False
    pareto_gt = points[~dom.any(axis=1)]
    pareto_gt = pareto_gt[np.lexsort((pareto_gt[:, 1], pareto_gt[:, 0]))]

    # Get the Pareto front by the user function
    pareto_user = pareto_front(points)

    # Check correctness
    print(np.allclose(pareto_gt, pareto_user))
