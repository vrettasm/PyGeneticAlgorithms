"""
Description:

    Includes utility functions that used throughout pygenalgo.

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
__all__ = ["pareto_front", "cost_function", "np_cdist", "clamp",
           "pareto_dominance", "np_pareto_front", "two_indices_fast"]

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

def pareto_dominance(point_a: tuple | list,
                     point_b: tuple | list) -> bool:
    """
    Implements a shortcut version of the Pareto dominance condition:

    all(p <= q for p, q in zip(point_i, point_j)) &&
    any(p < q for p, q in zip(point_i, point_j))

    NOTE: It is assumed that both points have the same size (length).

    :param point_a: the  first point (as tuple or list).

    :param point_b: the second point (as tuple or list).

    :return: if the condition is satisfied.
    """
    # Second condition.
    strictly_better: bool = False

    # Scan both points elementwise.
    for p, q in zip(point_a, point_b):
        # Return immediately.
        if p < q:
            return False
        # _end_if_

        # Check if the second condition
        # is satisfied.
        if not strictly_better and p > q:
            # Switch the flag value
            # to avoid re-setting it.
            strictly_better = True
    # _end_for_

    # If we got here return only the second condition.
    return strictly_better
# _end_def_

def pareto_front(points: list) -> list:
    """
    Simple function that calculates the Pareto (optimal)
    front points from a given input points list.

    NOTE: The function is working directly with lists,
    even though it can be optimized using numpy arrays.

    :param points: list of points [(fx1, fx2, ..., fxn),
                                   (fy1, fy2, ..., fyn),
                                   ....................,
                                   (fk1, fk2, ..., fkn)]

    :return: List of points that lie on the Pareto front.
    """
    # Get the size of the first point.
    n_size = len(points[0]) if points else 0

    # Sanity check.
    if n_size == 0 or any(len(p) != n_size for p in points[1:]):
        raise RuntimeError("All points must have the same size.")
    # _end_if_

    # Create a set that will hold
    # all the Pareto front points.
    pareto_points = set()

    # Iterate through every point in the list.
    for i, point_i in enumerate(points):

        # Set the pareto optimal flag to True.
        is_pareto_optimal: bool = True

        # Compare it against every other point.
        for j, point_j in enumerate(points):

            # Check if "dominance" condition is satisfied.
            if i != j and pareto_dominance(point_i, point_j):
                # We swap the flag value.
                is_pareto_optimal = False

                # Break the internal loop and
                # continue to the next point.
                break
        # _end_for_

        # If we get here and the flag hasn't changed
        # it means that 'point_i' is on the frontier.
        if is_pareto_optimal:
            pareto_points.add(point_i)
    # _end_for_

    # Return the points as list.
    return list(pareto_points)
# _end_def_

def np_pareto_front(points: NDArray) -> NDArray:
    """
    Simple function that calculates the Pareto (optimal)
    front points from a given input points numpy array.

    :param points: array of points [(fx1, fx2, ..., fxn),
                                    (fy1, fy2, ..., fyn),
                                    ....................,
                                    (fk1, fk2, ..., fkn)]

    :return: array of points that lie on the Pareto front.
    """
    # Sanity check.
    if points.ndim != 2:
        raise RuntimeError("Points must be a 2-D array.")
    # _end_if_

    # Get the number of points.
    num_points: int = points.shape[0]

    # Create an array of boolean to track Pareto optimal points.
    is_pareto_optimal: NDArray = np.ones(num_points, dtype=bool)

    for i, point_i in enumerate(points):
        # Condition 1:
        # Less than or equal to all objectives.
        le_all: NDArray = np.all(points <= point_i, axis=1)

        # Condition 2:
        # Less than at least in one objective.
        lt_any: NDArray = np.any(points < point_i, axis=1)

        # Combine the two conditions.
        dominates_i: NDArray = le_all & lt_any

        # Explicit self-exclusion.
        dominates_i[i] = False

        # Set the i-th flag appropriately.
        is_pareto_optimal[i] = not np.any(dominates_i)
    # _end_for_

    # Return only the unique Pareto optimal points.
    return np.unique(points[is_pareto_optimal], axis=0)
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

        # Check if the function returns a tuple (with two values)
        # or a single output parameter. In the former, the second
        # value should be boolean to signal that the solution meets
        # the termination requirements.
        if isinstance(result, tuple) and len(result) == 2:

            f_value, solution_is_found = result[0], bool(result[1])
        else:

            f_value, solution_is_found = result, False
        # _end_if_

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

def two_indices_fast(rng: Generator, num: int) -> tuple[int, int]:
    """
    Select two distinct random indices in the range [0, num)
    without allocating, or shuffling a np.arange(num) array.

    :param rng: Random number generator used to draw uniform
                integers.

    :param num: Exclusive upper bound of the index range.
                Must be greater than 1.

    :return: two distinct integer values 'i' and 'j' from the
             range [0, num-1].
    """
    # Sanity check.
    if num <= 1:
        raise ValueError("'num' must be more than one.")

    # Pick a random 'i' in [0, num).
    i: int = rng.integers(num, dtype=int)

    # Pick another random 'j' in [0, num-1).
    j: int = rng.integers(num-1, dtype=int)

    # Set 'j' by excluding 'i' via a mapped draw.
    return i, j if j < i else j + 1
# _end_def_
