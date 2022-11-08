from trimesh import Trimesh
from trimesh.util import unitize
from trimesh.intersections import planes_lines
from trimesh.ray.ray_pyembree import _EmbreeWrap

from typing import Tuple, List

import numpy as np


def single_ray_mesh_intersection(mesh: Trimesh, ray_origin: np.ndarray, ray_direction: np.ndarray,
                                 **kwargs) -> List[float]:
    '''
    Intersect one ray with a mesh and return the intersection locations.
    '''
    # Invoke the ray mesh intersection function
    locations = ray_mesh_intersection(mesh, ray_origin, ray_direction, **kwargs)

    return locations[0]


def parallel_ray_mesh_intersection(mesh: Trimesh, ray_origins: np.ndarray, ray_direction: np.ndarray,
                                   origins_outside: bool=True, perturbation: float=0.01,
                                   **kwargs) -> List[List[float]]:
    '''
    Intersect a group of parallel rays with a mesh and obtain the intersection locations.

    This function supports an optional assumption that all ray origins are outside the mesh using
    the `origins_outside` parameter. Parity checking will be conducted on the number of
    intersections if the assumption is enabled.
    '''
    # Back up the ray origins array to avoid altering the input
    ray_origins_backup = np.atleast_2d(np.asarray(ray_origins, dtype=np.float64))
    ray_origins = ray_origins_backup.copy()

    # Expand ray direction into a 2D array
    num_rays = len(ray_origins)
    ray_directions = \
        np.atleast_2d(np.asarray(ray_direction, dtype=np.float64)).repeat(num_rays, axis=0)

    # Initialize the array of intersection locations
    intersections = [[] for _ in range(num_rays)]

    # Initialize the indices of unfinished rays
    current_indices = np.arange(num_rays)

    # Trial loop
    # To speed up ray mesh intersection, we batch all rays together and submit a group query
    # per trial
    trial_no = 0

    while True:
        # Print current trial info
        num_current_rays = len(current_indices)

        print(f'Trial {trial_no}: {num_current_rays} rays')

        # Perform batch ray mesh intersection by calling the backend function
        locations = \
            ray_mesh_intersection(
                mesh, ray_origins[current_indices],
                ray_directions[:num_current_rays], **kwargs
            )

        # Check if the results contain an even number of intersections (since the origins are
        # guaranteed to be outside the mesh)
        if origins_outside:
            num_intersections = np.array([len(loc) for loc in locations])
            valid = (num_intersections & 1) == 0
        else:
            valid = np.ones(num_current_rays, dtype=np.bool)

        # Save the intersections if they passed the check
        locations_valid = [loc for loc, flag in zip(locations, valid) if flag]
        for ind, loc in zip(current_indices[valid], locations_valid):
            intersections[ind] = loc

        # Only proceed to the next trial with invalid rays
        current_indices = current_indices[~valid]

        # Exit the loop when all rays obtained valid results
        num_current_rays = len(current_indices)
        if num_current_rays == 0:
            break

        # Generate the next group of rays by jittering the origins
        perturbations = \
            np.hstack((
                np.zeros((num_current_rays, 1)),
                np.random.uniform(-perturbation, perturbation, size=(num_current_rays, 2))
            ))
        ray_origins[current_indices] = ray_origins_backup[current_indices] + perturbations

        # Increment the trial number
        trial_no += 1

    return intersections


def ray_mesh_intersection(mesh: Trimesh, ray_origins: np.ndarray, ray_directions: np.ndarray,
                          ray_offset: float=1e-6, ray_offset_multiplier: float=10.0) \
                          -> Tuple[List[List[float]], np.ndarray]:
    '''
    Intersect a group of rays with a mesh and return the intersection locations.

    This is an optimized implementation from `trimesh.ray.ray_pyembree.intersects_id()`. It
    calls the PyEmbree wrapper directly and computes distances only.
    '''
    # Convert ray input to 2D numpy arrays and unitize the directions
    ray_origins = np.atleast_2d(np.asarray(ray_origins, dtype=np.float64).copy())
    ray_directions = np.atleast_2d(np.asarray(ray_directions, dtype=np.float64))
    ray_directions = unitize(ray_directions)

    # Represent triangles in the mesh as planes (for computing intersection distances)
    plane_origins = mesh.triangles[:, 0, :]
    plane_normals = mesh.face_normals

    # Get the PyEmbree scene object from mesh to bypass trimesh implementation
    scene: _EmbreeWrap = mesh.ray._scene

    # Constants
    num_rays = len(ray_origins)     # Number of rays in total
    NO_HIT = -1                     # Indicator for not hitting any triangle

    # Information of last hits
    last_hit_triangles = np.full(num_rays, NO_HIT)          # Indices of triangles that were hit
    last_hit_distances = np.full(num_rays, -ray_offset)     # Distances from ray origins

    # Offsets for creating new rays after intersections
    ray_offsets = np.full(num_rays, ray_offset)

    # Distance records of all intersections
    locations = [[] for _ in range(num_rays)]

    # Indices of currently active rays
    current_indices = np.arange(num_rays)

    # Main loop - find rays with valid intersections and spawn new rays to find
    # subsequent intersections
    while True:

        # Get the intersections from PyEmbree
        # 'query' stores indices of triangles that the rays intersect with
        query = scene.run(
            ray_origins[current_indices],
            ray_directions[current_indices]
        )

        # Select hitting rays
        hit_mask = query != NO_HIT
        hit_indices = current_indices[hit_mask]
        hit_triangles = query[hit_mask]

        # Quit the loop if no more hits are detected
        if not len(hit_indices):
            break

        # Validate ray hits by computing distances to intersection points
        # An intersection is invalid when ray and triangle are almost coplanar
        new_origins, valid_mask, distances = planes_lines(
            plane_origins[hit_triangles],
            plane_normals[hit_triangles],
            ray_origins[hit_indices],
            ray_directions[hit_indices],
            return_distance=True
        )

        # Raise error upon invalid intersections. Normally it should not happen
        if not valid_mask.all():
            raise RuntimeError('Invalid intersections encountered')

        # A ray might have been stuck in an infinite loop if the current hit is identical to
        # the previous hit. We need to check that and increase the offset distance
        duplicate_mask = last_hit_triangles[hit_indices] == hit_triangles

        # Record the current hit locations if they are not duplicates
        record_mask = ~duplicate_mask
        record_indices = hit_indices[record_mask]
        current_distances = \
            last_hit_distances[record_indices] + distances[record_mask] + ray_offsets[record_indices]
        for i, dist in zip(record_indices, current_distances):
            locations[i].append(dist)

        # Update the triangle indices at last hits
        last_hit_triangles[hit_indices] = hit_triangles

        # Update the distances at last hits
        last_hit_distances[record_indices] = current_distances

        # Update the ray offsets
        ray_offsets[hit_indices[duplicate_mask]] *= ray_offset_multiplier
        ray_offsets[record_indices] = ray_offset

        # Create new rays by translating the intersection points
        ray_origins[hit_indices] = new_origins + \
            ray_directions[hit_indices] * ray_offsets[hit_indices, None]

        # Mark the new rays as active
        current_indices = hit_indices

    return locations
