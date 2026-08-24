"""  Island-selector module. """

# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class IslandSelector(SelectionOperator):
    """
    Description:

        Island-selector performs the selection of individuals (parents) for each
        island. The object holds a list of SelectionOperator objects along with a
        pointer that is unique for each island.

        Once the pointer is set, the selection is performed using the predefined
        SelectionOperator that corresponds to the specific island.
    """

    def __init__(self, select_probability: float = 1.0,
                 select_ops: list[SelectionOperator] = None) -> None:
        """
        Construct an 'IslandSelector' object with a predefined probability
        value and a list of SelectionOperator objects.

        :param select_probability: (float).
        :param select_ops: a list of SelectionOperator objects.

        :return: None.
        """
        # Call the super constructor with the provided initial value.
        super().__init__(selection_probability=select_probability)

        # Sanity check.
        if select_ops is None or len(select_ops) == 0:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"'select_ops' is missing or empty.")
        # _end_if_

        # Sanity check.
        if any(not isinstance(op, SelectionOperator) for op in select_ops):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"'select_ops' items must be of type SelectionOperator.")
        # _end_if_

        # Copy the variables locally.
        self._items = {"operators": select_ops, "idx": 0}
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Select the individuals, from the input population, that will be
        passed on to the next genetic operations of crossover and mutation
        to form the new population of solutions.

        :param population: a list of chromosomes to select the parents from.

        :return: the selected parents population (as list of chromosomes).
        """
        # Get the pointer value.
        idx: int = self._items["idx"]

        # Get the list of selectors.
        select_op: list[SelectionOperator] = self._items["operators"]

        # Apply the specific select method.
        return select_op[idx].select(population)
    # _end_def_

    def set_pointer(self, idx: int) -> None:
        """
        Set the pointer to the selection operator.

        :param idx: the index of the selection operator.

        :return: None.
        """
        # Sanity check.
        if idx < 0 or idx >= len(self._items["operators"]):
            raise IndexError(f"{self.__class__.__name__}: "
                             f"selected index out of range.")
        # _end_if_

        # Update the index in the dict.
        self._items["idx"] = idx
    # _end_def_

    @property
    def all_counters(self) -> dict:
        """
        Accessor (get) of the application counter from all the internal
        selectors. This is mostly to verify that everything is working
        as expected.

        :return: a dictionary with the counter calls for all selectors.
        """
        return {
            sel_op.__class__.__name__: sel_op.counter
            for sel_op in self._items["operators"]
        }
    # _end_def_

    def reset_counter(self) -> None:
        """
        Sets ALL the counters to zero. We have to override the
        super().reset_counter() method because we have to call
        explicitly the reset_counter on all the internal operators.

        :return: None.
        """
        # First call the super() to reset the self internal counter.
        super().reset_counter()

        # Here call explicitly the reset on each of the internal selectors.
        for op in self._items["operators"]:
            op.reset_counter()
    # _end_def_

# _end_class_
