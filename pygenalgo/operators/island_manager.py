""" Island manager module. """
from pygenalgo.operators.genetic_operator import GeneticOperator


class IslandManager:
    """
    TBD:
    """

    __slots__ = ["_operators", "_idx"]

    def __init__(self, operators: list[GeneticOperator]) -> None:
        """
        Construct a 'IslandManager' object.

        :param operators: a list of genetic operators.

        :return: None.
        """
        # Sanity check.
        if operators is None or len(operators) == 0:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"'operators' list is missing or empty.")
        # _end_if_

        # Copy the operators list.
        self._operators: list[GeneticOperator] = operators

        # Initialize the index.
        self._idx: int = 0
    # _end_def_

    @property
    def operator(self) -> list:
        """
        TBD

        :return:
        """
        return self._operators
    # _end_def_

    @property
    def idx(self) -> int:
        """
        TBD
        :return:
        """
        return self._idx
    # _end_def_

    @idx.setter
    def idx(self, new_idx: int) -> None:
        """
        Set the pointer to the genetic operator that
        we want to execute.

        :param idx: the index of the genetic operator.

        :return: None.
        """
        # Ensure correct type.
        new_idx: int = int(new_idx)

        # Sanity check.
        if new_idx < 0 or new_idx >= len(self._operators):
            raise IndexError(f"{self.__class__.__name__}: "
                             f"selected index is out of range.")
        # _end_if_

        # Update the index.
        self._idx = new_idx
    # _end_def_

    def get_all_counters(self) -> dict:
        """
        Accessor (getter) of the application counter from all
        the internal mutators. This is mostly to verify that
        everything is working as expected.

        :return: a dictionary with the counter calls for all
                 mutator methods.
        """
        return {
            f"{n}-{gen_op.__class__.__name__}": gen_op.counter
            for n, gen_op in enumerate(self._operators)
        }
    # _end_def_

    def reset_island_counters(self) -> None:
        """
        Resets the internal genetic operators counters.

        :return: None.
        """
        # Clear all the counters.
        for op in self._operators:
            op.reset_counter()
    # _end_def_

# _end_class_
