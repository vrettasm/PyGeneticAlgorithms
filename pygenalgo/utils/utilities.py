import numpy as np
from typing import Callable
from functools import wraps, partial

# Public interface.
__all__ = ["pareto_front", "cost_function",
           "pareto_dominance", "np_pareto_front"]

def pareto_dominance(point_a: tuple | list,
                     point_b: tuple | list) -> bool:
    """
    Implements a shortcut version of the pareto dominance condition:

        all(p <= q for p, q in zip(point_i, point_j))
                            and
        any(p < q for p, q in zip(point_i, point_j))

    NOTE: It is assumed that both points have the same size (length).

    :param point_a: the  first point (as tuple or list).

    :param point_b: the second point (as tuple or list).

    :return: if the condition is satisfied.
    """
    # Second condition.
    at_least_one_greater = False

    # Scan both points elementwise.
    for p, q in zip(point_a, point_b):
        # Return immediately.
        if p < q:
            return False
        # _end_if_

        # Check if the second condition is satisfied.
        if not at_least_one_greater and p > q:
            # Switch the flag value to avoid re-setting it.
            at_least_one_greater = True
        # _end_if_
    # _end_for_

    # If we got here return only the second condition.
    return at_least_one_greater
# _end_def_

def pareto_front(points: list) -> list:
    """
    Simple function that calculates the pareto (optimal)
    front points from a given input points list.

    NOTE: The function is working directly with lists,
    even though it can be optimized using numpy arrays.

    :param points: list of points [(fx1, fx2, ..., fxn),
                                   (fy1, fy2, ..., fyn),
                                   ....................,
                                   (fk1, fk2, ..., fkn)]

    :return: List of points that lie on the pareto front.
    """
    # Create a set with all the points sizes.
    check_size = {len(point) for point in points}

    # Sanity check.
    if len(check_size) > 1:
        raise RuntimeError("All point must have the same size.")
    # _end_if_

    # Create a set that will hold
    # all the Pareto front points.
    pareto_points = set()

    # Iterate through every point in the list.
    for i, point_i in enumerate(points):

        # Set the pareto optimal flag value to True.
        is_pareto_optimal = True

        # Compare it against every other point.
        for j, point_j in enumerate(points):

            # Check if "dominance" condition is satisfied.
            if i != j and pareto_dominance(point_i, point_j):
                # We swap the flag value.
                is_pareto_optimal = False

                # Break the internal loop and
                # continue to the next point.
                break
            # _end_if_

        # _end_for_

        # If we get here and the flag hasn't changed
        # it means that 'point_i' is on the frontier.
        if is_pareto_optimal:
            pareto_points.add(point_i)
        # _end_if_
    # _end_for_

    # Return the points as list.
    return list(pareto_points)
# _end_def_

def np_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Simple function that calculates the pareto (optimal)
    front points from a given input points numpy array.

    :param points: array of points [(fx1, fx2, ..., fxn),
                                    (fy1, fy2, ..., fyn),
                                    ....................,
                                    (fk1, fk2, ..., fkn)]

    :return: array of points that lie on the pareto front.
    """
    # Sanity check.
    if points.ndim != 2:
        raise RuntimeError("Points must be a 2-D array.")
    # _end_if_

    # Get the number of points.
    num_points = points.shape[0]

    # Create a boolean array to track Pareto optimal points.
    is_pareto_optimal = np.ones(num_points, dtype=bool)

    for i, point_i in enumerate(points):
        # Compare point i-th with all other points.
        is_dominated = np.any(np.all(points <= point_i, axis=1) &
                              np.any(points < point_i, axis=1))
        # Set the flag appropriately.
        is_pareto_optimal[i] = not is_dominated
    # _end_for_

    # Return only the unique Pareto optimal points.
    return np.unique(points[is_pareto_optimal], axis=0)
# _end_def_

def cost_function(func: Callable = None, minimize: bool = False):
    """
    Decorator for the function that we want to optimize.
    The default setting is maximization.

    :param func: the function to be optimized.

    :param minimize: if 'True' it will return the negative
    function value to allow for the minimization.
    Default is 'False'.

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
