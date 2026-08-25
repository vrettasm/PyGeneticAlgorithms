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
        Construct an 'IslandCrossover' object with a predefined probability
        value and a list of CrossoverOperator objects.

        :param crossover_probability: (float).
        :param crossx_ops: a list of CrossoverOperator objects.

        :return: None.
        """
        # Call the super constructors with the provided initial value.
        super().__init__(crossover_probability=crossover_probability)

        # Sanity check.
        if crossx_ops is None or any(not isinstance(op, CrossoverOperator)
                                     for op in crossx_ops):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"'crossx_ops' items must be of type CrossoverOperator.")
        # _end_if_

        # Create an IslandOperator.
        self._items: IslandOperator = IslandOperator(operators=crossx_ops)
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Offsprings:
        """
        Perform the crossover operation on the two input parent
        chromosomes, by selecting randomly a predefined method.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """
        # If the crossover probability is higher than a uniformly
        # random value and the parents aren't identical apply the
        # changes.
        if (parent1 != parent2) and self.is_operator_applicable():
            # Local reference of island operator.
            island_op = self._items

            # Get the selected operator.
            crossx_op: CrossoverOperator = island_op.operator[island_op.idx]

            # Increase the crossover counter.
            self.inc_counter()

            # Call its crossover method.
            return crossx_op.crossover(parent1, parent2)
        # _end_if_

        # Return two cloned offsprings.
        return parent1.clone(), parent2.clone()
    # _end_def_

    def reset_counter(self) -> None:
        """
        Sets ALL the counters to 'zero'. We have to override the
        super().reset_counter() method,  because we have to call
        explicitly the reset_counter on all the internal operators.

        :return: None.
        """
        # First call the super() to reset
        # the self internal counter.
        super().reset_counter()

        # Then clear all the island counters.
        for op in self._items.operator:
            op.reset_counter()
    # _end_def_

    def all_counters(self) -> dict:
        """
        Accessor (getter) of the application counter from all
        the internal crossovers. This is mostly to verify that
        everything is working as expected.

        :return: a dictionary with the counter calls for all
                 crossover methods.
        """
        return {
            f"{n}-{gen_op.__class__.__name__}": gen_op.counter
            for n, gen_op in enumerate(self._items.operator)
        }
    # _end_def_

# _end_class_
