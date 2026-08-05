"""
Description:

    Includes utility functions that used throughout PyGenAlgo.

Author:
    Michail D. Vrettas, PhD

Email:
    michail.vrettas@gmail.com

Metadata:
    License: GPL-3
"""

from typing import Callable, Union
from functools import wraps, partial

import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator

# Public interface.
__all__ = ["cost_function", "np_cdist", "two_indices_fast",
           "np_pareto_front", "clamp",  "np_pareto_front_index"]

# Declare a union type.
Number = Union[int, float]

def clamp(x: Number,
          x_lower: Number,
          x_upper: Number) -> Number:
    """
    Clamps a value within a specified range.

    :param x: value to clamp.

    :param x_lower: lower bound.

    :param x_upper: upper bound.

    :return: clamped value.
    """
    return min(max(x, x_lower), x_upper)
# _end_def_

def _dominance_batch(sample_points: NDArray, eps_arr: NDArray) -> NDArray:
    """
    Uses the dominance condition on the sample points to find
    those that are on the Pareto front. It's using vectorized
    (numpy optimized) code to compute the dominance.

    WARNING: This step is O(N^2 x D) in memory allocation.

    :param sample_points: The points we want to find the Pareto
                          front.
    :param eps_arr: Threshold tolerance values (per objective).

    :return: an array with the indices of the pareto samples.
    """
    # Subtract all points (from all other points).
    diff: NDArray = sample_points[:, None, :] - sample_points[None, :, :]

    # Covert eps to [1, 1, n_dims].
    eps_batch = eps_arr[None, None, :]

    # This condition is for maximization problems.
    strictly_better: NDArray = np.all(diff >= -eps_batch, axis=-1) & \
                               np.any(diff > eps_batch, axis=-1)
    # Get the pareto points mask.
    return ~np.any(strictly_better, axis=0)
# _end_def_

def _dominance_loop(sample_points: NDArray, eps_arr: NDArray) -> NDArray:
    """
    Uses the dominance condition on the sample points to find
    those that are on the Pareto front. It is using a loop to
    do that, and it's called when the size of the input array
    exceeds 500MB in memory. This approach is safer in terms
    of memory usage, but slower due to the loop.

    NOTE: This function is O(N x D) in memory allocation.

    :param sample_points: The points we want to find the Pareto
                          front.
    :param eps_arr: Threshold tolerance values (per objective).

    :return: an array with the indices of the pareto samples.
    """
    # Get the number of sample points.
    n_points = sample_points.shape[0]

    # For the loop branch eps [1, n_dims].
    eps_loop: NDArray = eps_arr[None, :]

    # Initialize all sample points as Pareto optimal.
    is_pareto: NDArray = np.ones(n_points, dtype=bool)

    # Check all pairs once, bidirectionally.
    for i in range(n_points):

        # If i-th point is already dominated.
        if not is_pareto[i]:
            # Skip.
            continue

        # Compare i against all later points (j > i).
        for j in range(i + 1, n_points):

            # If j is already dominated.
            if not is_pareto[j]:
                # Skip.
                continue

            # Compute difference once (reuse for both checks).
            diff: NDArray = sample_points[i] - sample_points[j]

            # Check if i dominates j.
            i_dominates_j: NDArray = (np.all(diff >= -eps_loop) &
                                      np.any(diff > eps_loop))

            # Check if j dominates i (reverse the comparison).
            j_dominates_i: NDArray = (np.all(-diff >= -eps_loop) &
                                      np.any(-diff > eps_loop))
            if i_dominates_j:
                # Flag for skipping.
                is_pareto[j] = False

            elif j_dominates_i:
                # Flag for skipping.
                is_pareto[i] = False
                break
    # _end_for_

    return is_pareto
# _end_def_

def np_pareto_front_index(points: NDArray,
                          mode: str = "max",
                          rel_eps: float = 1.0e-6,
                          exclude_duplicates: bool = True) -> NDArray:
    """
    Fast (numpy - vectorized) function that calculates
    the Pareto optimal front from a given input points
    numpy array, but returns their indices instead  of
    their actual values.

    :param points: array of points [(fx1, fx2, ..., fxn),
                                    (fy1, fy2, ..., fyn),
                                    ....................,
                                    (fk1, fk2, ..., fkn)]

    :param mode: "max" (maximize all objective) or
                 "min" (minimize all objective).

    :param rel_eps: relative eps threshold tolerance.

    :param exclude_duplicates: whether to exclude duplicate
                               points.

    :return: array of indexes (from the points that lie on
             the Pareto front).
    """
    # Sanity check.
    if points.ndim != 2:
        raise RuntimeError("Points must be a 2-D array.")
    # _end_if_

    # Sanity check.
    if mode not in ("max", "min"):
        raise ValueError("Mode must be either 'max' or 'min'.")
    # _end_if_

    # Normalize to a single convention. Here we
    # maximize in all objectives function values.
    x_points = points if mode == "max" else -points

    # Check for deduplicates.
    if exclude_duplicates:

        # Remove duplicate points to speed up the
        # routine and keep track of their indices.
        sample_points, sample_indices = np.unique(
            x_points, axis=0, return_index=True
        )
    else:
        # Keep all points in the Pareto calculation.
        sample_points = x_points
        sample_indices = np.arange(x_points.shape[0])
    # _end_if_

    # Get the dimensions of unique points.
    n_points, n_dims = sample_points.shape

    # Compute an eps value per objective: (n_dims,).
    eps_arr: NDArray = rel_eps * np.max(np.abs(sample_points), axis=0)

    # Rough calculation of sample_points memory.
    memory_bytes: int = (n_points * n_points * n_dims *
                         sample_points.dtype.itemsize)

    # Compare it against ~500MB.
    if memory_bytes <= 500_000_000:
        # Return the pareto front indices (batch mode).
        return sample_indices[_dominance_batch(sample_points, eps_arr)]

    # Safeguard with loop mode.
    return sample_indices[_dominance_loop(sample_points, eps_arr)]
# _end_def_

def np_pareto_front(points: NDArray, mode: str = "max",
                    rel_eps: float = 1.0e-6,
                    exclude_duplicates: bool = True) -> NDArray:
    """
    Fast (numpy - vectorized) function that calculates
    the Pareto optimal front points from a given input
    points numpy array.

    :param points: array of points [(fx1, fx2, ..., fxn),
                                    (fy1, fy2, ..., fyn),
                                    ....................,
                                    (fk1, fk2, ..., fkn)]

    :param mode: "max" (maximize all objective) or
                 "min" (minimize all objective).

    :param rel_eps: relative eps threshold tolerance.

    :param exclude_duplicates: whether to exclude duplicate
                               points.

    NOTE: Its memory complexity grows quadratically
    with the number of points in the NDArray: O(N^2)!

    :return: array of points that lie on the Pareto front.
    """

    # Get the indices of the points that lie on the pareto front.
    idx = np_pareto_front_index(points, mode=mode, rel_eps=rel_eps,
                                exclude_duplicates=exclude_duplicates)

    # Return the actual points.
    return points[idx]
# _end_def_

def cost_function(func: Callable = None, minimize: bool = False):
    """
    Decorator for the function that we want to optimize.
    The default setting is maximization.

    :param func: the function to be optimized.

    :param minimize: if 'True' it will return the negative function
                     value to allow for the minimization. Default is
                     set to 'False'.

    :return: the 'function_wrapper' method.
    """
    # This allows the decorator to be called with
    # parenthesis and using the default parameters.
    if func is None:
        return partial(cost_function, minimize=minimize)
    # _end_if_

    @wraps(func)
    def function_wrapper(*args, **kwargs) -> dict:
        """
        Internal function wrapper.

        :param args: function positional arguments.

        :param kwargs: function keywords arguments.

        :return: a dictionary with two key-values.
        """

        # Run the function we want to optimize.
        result = func(*args, **kwargs)

        # Check if the function returns a tuple, with two values
        # or a single output parameter. In the former the second
        # value should be bool to signal that the solution meets
        # the termination requirements.
        if isinstance(result, tuple) and len(result) == 2 and\
                isinstance(result[1], (bool, np.bool_)):

            f_value, solution_is_found = result
        else:

            f_value, solution_is_found = result, False
        # _end_if_

        # Multi-objective functions return a tuple
        # with all the objective function values.
        if isinstance(f_value, tuple):

            if minimize:
                # Reverse the sign of the objectives.
                f_value = tuple(-fx for fx in f_value)

            return {"f_value": f_value,
                    "solution_is_found": solution_is_found}
        # _end_if_

        # Standard return statement.
        return {"f_value": -f_value if minimize else f_value,
                "solution_is_found": solution_is_found}
    # _end_def_

    return function_wrapper
# _end_def_

def np_cdist(x_pos: NDArray, scaled: bool = False) -> NDArray:
    """
    This is equivalent to the scipy.spatial.distance.cdist method with Euclidean
    distance metric. It is a tailored version for the purposes of the multimodal
    operation mode.

    :param x_pos: a ndarray of positions. The dimensions of the input array should be
                  [n_rows, n_cols], where n_rows is the number of particles and n_cols
                  are the number of positions. In special 3D cases the input can have
                  a shape of [n_sample, n_rows, n_cols], in these cases the input will
                  be first reshaped to [n_sample, (n_rows*n_cols)] before we continue.

    :param scaled: boolean flag that allows the input array to be scaled, using the
                   MaxAbsScaler, before computing the distances.

    :return: a square [n_rows, n_rows] ndarray of distances.
    """
    # Check if the input array is 3D.
    if x_pos.ndim == 3:
        # Get the original shape of the matrix.
        n_samples, n_rows, n_cols = x_pos.shape

        # Reshape it by combining the n_rows and n_cols.
        x_pos = x_pos.reshape((n_samples, n_rows * n_cols), copy=False)
    # _end_if_

    # Get the number of rows/cols.
    n_rows, n_cols = x_pos.shape

    # Check if we want the input data to be scaled.
    if scaled:
        # Scale with the MaxAbsScaler.
        x_pos /= np.max(np.abs(x_pos), axis=0)
    # _end_if_

    # Create a square matrix with zeros.
    dist_x = np.zeros((n_rows, n_rows), dtype=float)

    # Iterate through all vectors.
    for i in range(n_rows):
        # Compute the Euclidean norm of the 'i-th' element with the rest of them.
        dist_x[i, i + 1:] = np.sqrt(np.sum((x_pos[i] - x_pos[i + 1:, :]) ** 2, axis=1))

        # Since the array is symmetric store the result in the 'i-th' column too.
        dist_x[:, i] = dist_x[i, :]
    # _end_for_
    return dist_x
# _end_def_

def two_indices_fast(rng: Generator, num: int,
                     in_order: bool = False) -> tuple[int, int]:
    """
    Select two distinct random indices in the range [0, num)
    without allocating, or shuffling a np.arange(num) array.

    :param rng: Random number generator used to draw uniform
                integers.

    :param num: Exclusive upper bound of the index range.
                Must be greater than 1.

    :param in_order: Boolean flag that allows the output
                     values to be sorted in ascending order.

    :return: Two distinct integer values from the range [0, num).
    """
    # Sanity check.
    if num <= 1:
        raise ValueError("'num' must be more than one.")

    # Pick a random 'i' in [0, num).
    i: int = rng.integers(num, dtype=int)

    # Pick another random 'k' in [0, num-1).
    k: int = rng.integers(num-1, dtype=int)

    # If the flag is set to True.
    if in_order:
        # Return in ascending order.
        return (k, i) if k < i else (i, k + 1)

    # Default is random order.
    # Exclude 'i' from the second index via a mapped draw.
    return i, k if k < i else k + 1
# _end_def_
