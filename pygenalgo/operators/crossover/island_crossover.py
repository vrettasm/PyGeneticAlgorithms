""" Island-crossover module. """
# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.island_manager import IslandManager
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offsprings)


class IslandCrossover(CrossoverOperator):
    """
    Description:

        Island-crossover performs the crossover of individuals (offspring) for
        each island. The object holds a list of CrossoverOperator objects along
        with a pointer that is unique for each island.

        Once the pointer is set, the crossover is performed using the predefined
        CrossoverOperator that corresponds to the specific island.
    """

    def __init__(self, crossover_probability: float = 0.9,
                 crossx_ops: list[CrossoverOperator] = None) -> None:
        """
        Construct an 'IslandCrossover' object with a predefined
        probability value and a list of CrossoverOperator objects.

        :param crossover_probability: (float).
        :param crossx_ops: a list of CrossoverOperator objects.

        :return: None.
        """
        # Call the super() constructor with an initial probability.
        # Note: This probability is never used. We use directly the
        # probability values of the genetic operators in the _items.
        super().__init__(crossover_probability=crossover_probability)

        # Create an IslandManager to handle the operators.
        self._items: IslandManager = IslandManager(operators=crossx_ops)

        # Sanity check: correct Type.
        if any(not isinstance(op, CrossoverOperator) for op in crossx_ops):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"'crossx_ops' items must be of type CrossoverOperator.")
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Offsprings:
        """
        Perform the crossover operation on the two input parent
        chromosomes, by selecting randomly a predefined method.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """
        # Get the selected (crossover) operator.
        crossx_op: CrossoverOperator = self._items.operator

        # Call its crossover method.
        return crossx_op.crossover(parent1, parent2)
    # _end_def_

    def reset_counter(self) -> None:
        """
        Sets ALL the counters to 'zero'. We have to override the
        super().reset_counter() method,  because we have to call
        explicitly the reset_counter on all the internal operators.

        :return: None.
        """
        # First call the super() to reset the self counter.
        super().reset_counter()

        # Then call the _items to reset the island counter.
        self._items.reset_island_counters()
    # _end_def_

# _end_class_
