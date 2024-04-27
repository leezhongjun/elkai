import numpy as np

def path_distance(path: np.ndarray, distance_matrix: np.ndarray):
    """Returns the distance of a path given a distance matrix."""
    dist = 0
    last_idx = path[0]
    tour = path[1:]

    if len(path) != len(distance_matrix[0]) + 1:
        tour = np.append(tour, path[0])

    for idx in tour:
        dist += distance_matrix[last_idx][idx]
        last_idx = idx

    return dist

def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    """Flattens a 2D matrix into a 1D array."""
    N = len(matrix)
    if N <= 2:
        raise TypeError("Argument must be a N*N matrix with N > 2")
    flattened = matrix.flatten()
    return flattened

def is_2d_matrix(obj: np.ndarray) -> bool:
    """Checks if obj is a 2d matrix of ints/floats."""
    if obj.ndim != 2:
        return False
    if not np.issubdtype(obj.dtype, np.number):
        return False
    return True

def is_symmetric_matrix(obj: np.ndarray) -> bool:
    """Returns True if obj is a symmetric matrix, False otherwise."""
    return np.allclose(obj, obj.T)
