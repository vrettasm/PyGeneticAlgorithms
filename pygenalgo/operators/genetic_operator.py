from typing import Any
from threading import Lock
from functools import wraps
from pygenalgo.utils.utilities import clamp
from numpy.random import default_rng, Generator

# Public interface.
__all__ = ["GeneticOperator", "increase_counter"]


def increase_counter(method):
    """
    Decorator function that is used in the derived
    classes main operation to increase the counter
    by one.

    :param method: that we wrap its functionality.

    :return: the wrapper function.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        """
        NOTE: We do not do any kind of error handling in here
        because if the method fails (exit with an error) then
        the counter value will be useless.
        """
        # Run the wrapped method.
        result = method(self, *args, **kwargs)

        # Increase the counter.
        self.inc_counter()

        # Return the output.
        return result
    # _end_def_
    return wrapper
# _end_def_


class GeneticOperator(object):
    """
    Description:

        Provides the base class (interface) for a Genetic Operator.
        This class includes the common variables (such as the probability
        and the application counter) along with access to them.

        All genetic operators (Selection, Crossover, Mutation, Migration)
        should inherit this class.
    """

    # Create a random number generator.
    _rng: Generator = default_rng()

    # Initialize the iteration value.
    _iteration: int = 0

    # Object variables.
    __slots__ = ("_probability", "_counter", "_lock", "_items")

    def __init__(self, probability: float) -> None:
        """
        Construct a 'GeneticOperator' object with
        a given probability value.

        :param probability: (float) in [0, 1].
        """
        # Ensure the default entry value is within range [0, 1].
        self._probability = clamp(float(probability), 0.0, 1.0)

        # Initialize the application counter to zero.
        self._counter = 0

        # Initialize a thread lock.
        self._lock = Lock()

        # Place holder.
        self._items = None
    # _end_def_

    @classmethod
    def set_seed(cls, new_seed=None) -> None:
        """
        Sets a new seed for the random number generator.

        :param new_seed: New seed value (default=None).

        :return: None.
        """
        # Re-initialize the class variable.
        cls._rng = default_rng(seed=new_seed)
    # _end_def_

    @property
    def iteration(self) -> int:
        """
        Accessor (getter) of the iteration parameter.

        :return: the iteration value.
        """
        return self._iteration
    # _end_def_

    @classmethod
    def set_iteration(cls, value: int) -> None:
        """
        Accessor (setter) of the iteration value.

        :param value: (int).
        """
        # Check for correct type and allow only the positive values.
        if not isinstance(value, int) or value < 0:
            raise RuntimeError(f"{cls.__class__.__name__}: "
                               f"Iteration value should be positive int: {type(value)}.")
        # _end_if_

        # Update the iteration value.
        cls._iteration = value
    # _end_def_

    @property
    def items(self) -> Any:
        """
        Accessor (getter) of the _items container.

        :return: _items (if any).
        """
        return self._items
    # _end_def_

    @property
    def counter(self) -> int:
        """
        Accessor (getter) of the application counter.

        :return: the int value of the counter variable.
        """
        return self._counter
    # _end_def_

    def reset_counter(self) -> None:
        """
        Sets the counter value to zero.

        :return: None.
        """
        # Protect operator counter.
        with self._lock:
            self._counter = 0
    # _end_def_

    def inc_counter(self) -> None:
        """
        Increase the counter value by one. This is applied
        after each application of the genetic operator.

        :return: None.
        """
        # Protect operator counter.
        with self._lock:
            self._counter += 1
    # _end_def_

    @property
    def probability(self) -> float:
        """
        Accessor (getter) of the probability.

        :return: the float value of the probability.
        """
        return self._probability
    # _end_def_

    @probability.setter
    def probability(self, new_value: float) -> None:
        """
        Accessor (setter) of the probability.

        :param new_value: (float) in [0, 1].
        """
        # Check for the correct type.
        if not isinstance(new_value, float):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Probability should be float: {type(new_value)}.")
        # _end_if_

        # Ensure the correct range.
        if not (0.0 <= new_value <= 1.0):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Probability should be in [0, 1].")
        # _end_if_

        # Update the probability value.
        self._probability = new_value
    # _end_def_

    @property
    def rng(self):
        """
        Get access of the Class variable (_rng).

        :return: the random number generator.
        """
        return self._rng
    # _end_def_

    def is_operator_applicable(self) -> bool:
        """
        Since to apply a genetic operator we have to check
        it probabilistically, we set the condition in here
        so that the objects inheriting from this class can
        call only this function.

        If the genetic probability is higher than a uniformly
        random value, apply the operator's changes.

        :return: (bool) the output of the: probability > U(0,1).
        """
        return self._probability > self._rng.random()
    # _end_def_

    def __str__(self) -> str:
        """
        Description:
            Override to print a readable string presentation of the
            genetic operator object, using the selected parameters.
        """
        # Initialize the string with the class name and its ID.
        str_self = f" {self.__class__.__name__}: ({id(self)})\n"

        # Add all the fields with their values.
        for s in self.__slots__:
            str_self += (" " + s + ": " +
                         str(self.__getattribute__(s)) + "\n")
        # _end_for_

        # Return the string.
        return str_self
    # _end_def_

    def __repr__(self) -> str:
        """
        Description:
            Override to provide a simple string that’s a valid Python expression
            which could be used to recreate the object: ClassName(_probability).
        """
        return f"{self.__class__.__name__}({self._probability})"
    # _end_def_

    def __getstate__(self) -> dict:
        """
        This method is used when "pickling" the object during the parallel execution.
        For multiprocessing backends like 'loky' or 'multiprocessing', the Lock() in
        this object causes problems, since it's not 'pickleable'. Therefore, we have
        to implement our own getstate method to exclude the '_lock' feature.
        """
        return {
            attr: getattr(self, attr) for attr in self.__slots__
            if attr not in ("_lock",)
        }
    # _end_def_

    def __setstate__(self, state: dict) -> None:
        """
        This method works in tandem with the __getstate__() and used to unpickle the
        object. Since the threading.Lock() is not stored in the 'pickle', we need to
        add a new one upon creation of the new object.
        """
        for attr, value in state.items():
            setattr(self, attr, value)
        # _end_for_

        # Add a new lock.
        self._lock = Lock()
    # _end_def_

# _end_class_
