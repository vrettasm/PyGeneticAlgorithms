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

from numbers import Real
from typing import Callable, Union
from functools import wraps, partial
from collections.abc import Sequence

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

def np_pareto_front_slow(points: NDArray,
                         mode: str = "max") -> NDArray:
    """
    Simple function that calculates the Pareto optimal
    front points from a given input points numpy array.

    :param points: array of points [(fx1, fx2, ..., fxn),
                                    (fy1, fy2, ..., fyn),
                                    ....................,
                                    (fk1, fk2, ..., fkn)]

    :param mode: "max" (maximize all objective) or
                 "min" (minimize all objective).

    NOTE:  Its space (memory) complexity grows linearly
    with the number of points in the NDArray: O(N), but
    it is much slower than its numpy vectorized version.

    :return: array of points that lie on the Pareto front.
    """
    # Sanity check.
    if points.ndim != 2:
        raise RuntimeError("Points must be a 2-D array.")
    # _end_if_

    # Sanity check.
    if mode not in ("max", "min"):
        raise ValueError("Mode must be either 'max' or 'min'.")
    # _end_if_

    # First remove the duplicate points to speed up the loop.
    unique_points = np.unique(points, axis=0)

    # Get the number of unique points.
    n_points: int = unique_points.shape[0]

    # Normalize to a single convention:
    # maximize in all objectives function values.
    x_points = unique_points if mode == "max" else -unique_points

    # Create an array of boolean to track Pareto optimal points.
    is_pareto_optimal: NDArray = np.ones(n_points, dtype=bool)

    for i, point_i in enumerate(x_points):

        # Condition 1:
        # Greater than or equal in all objectives.
        ge_all: NDArray = np.all(x_points >= point_i, axis=1)

        # Condition 2:
        # Greater than at least in one objective.
        gt_any: NDArray = np.any(x_points > point_i, axis=1)

        # Combine the two conditions.
        is_dominated: NDArray = ge_all & gt_any

        # Explicit self-exclusion.
        is_dominated[i] = False

        # Set the i-th flag appropriately.
        is_pareto_optimal[i] = not np.any(is_dominated)
    # _end_for_

    # Return only the unique Pareto points.
    return unique_points[is_pareto_optimal]
# _end_def_

def np_pareto_front_index(points: NDArray,
                          mode: str = "max") -> NDArray:
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

    :return: array of indexes (from the points that lie on
             the Pareto front).
    """
    # Sanity check.
    if points.ndim != 2:
        raise RuntimeError("Points must be a 2-D array.")
    # _end_if_

    # Check objective.
    if minimize:
        points = -points
    # _end_if_

    # Remove duplicate points to speed up the routine.
    _, unique_indices = np.unique(points, axis=0,
                                  return_index=True)

    # Extract the unique points from the set.
    unique_points = points[unique_indices]

    # Subtract all points (from all other points).
    # WARNING: This step is O(N^2) in memory allocation.
    diff = unique_points[None, :, :] - unique_points[:, None, :]

    # This condition is for maximization problems.
    strictly_better = np.all(diff >= 0.0, axis=-1) & \
                      np.any(diff >  0.0, axis=-1)

    # Get the pareto points mask.
    is_pareto_unique = ~np.any(strictly_better, axis=0)

    # Return the indexes.
    return unique_indices[is_pareto_unique]
# _end_def_

def np_pareto_front(points: NDArray,
                    minimize: bool = False) -> NDArray:
    """
    Fast (numpy - vectorized) function that calculates
    the Pareto optimal front points from a given input
    points numpy array.

    :param points: array of points [(fx1, fx2, ..., fxn),
                                    (fy1, fy2, ..., fyn),
                                    ....................,
                                    (fk1, fk2, ..., fkn)]

    :param minimize: whether we are solving minimization
                     or maximization problem.

    NOTE: Its memory complexity grows quadratically
    with the number of points in the NDArray: O(N^2)!

    :return: array of points that lie on the Pareto front.
    """

    # First get the indexes of the pareto front points,
    # using the helper function.
    idx = np_pareto_front_index(points, minimize=minimize)

    # Then return the actual points.
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
                isinstance(result[1], bool):

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
