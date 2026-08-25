"""  Island-selector module. """

# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.island_manager import IslandManager
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
        # Call the super() constructor with an initial probability.
        super().__init__(selection_probability=select_probability)

        # Create an IslandOperator.
        self._items: IslandManager = IslandManager(operators=select_ops)

        # Sanity check.
        if any(not isinstance(op, SelectionOperator) for op in select_ops):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"'select_ops' items must be of type SelectionOperator.")
    # _end_def_

    @property
    def probability(self) -> float:
        """
        Accessor (getter) of the _items probability.

        :return: the float value of the probability.
        """
        return self._items.operator.probability
    # _end_def_

    @probability.setter
    def probability(self, new_value: float) -> None:
        """
        Accessor (setter) of the _items probability.

        :param new_value: (float) in [0, 1].
        """
        # Assign the new probability to the composed
        # _items operator.
        self._items.operator.probability = new_value
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
        # Get the selected (selection) operator.
        select_op: SelectionOperator = self._items.operator

        # Apply the specific select method.
        return select_op.select(population)
    # _end_def_

    def reset_counter(self) -> None:
        """
        Sets ALL the counters to zero. We have to override the
        super().reset_counter() method because we have to call
        explicitly the reset_counter on all the internal operators.

        :return: None.
        """
        # First call the super() to reset the self counter.
        super().reset_counter()

        # Then call the _items to reset the island counter.
        self._items.reset_island_counters()
    # _end_def_

# _end_class_
