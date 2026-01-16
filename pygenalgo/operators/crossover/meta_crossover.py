from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator
from pygenalgo.operators.crossover.uniform_crossover import UniformCrossover
from pygenalgo.operators.crossover.multi_point_crossover import MultiPointCrossover
from pygenalgo.operators.crossover.single_point_crossover import SinglePointCrossover


class MetaCrossover(CrossoverOperator):
    """
    Description:

        Meta-crossover, crosses the chromosomes by applying randomly
        all other crossovers (one at a time), with equal probability.
    """

    def __init__(self, crossover_probability: float = 0.9) -> None:
        """
        Construct a 'MetaCrossover' object with a given probability value.

        :param crossover_probability: (float).
        """
        # Call the super constructor with the provided
        # probability value.
        super().__init__(crossover_probability)

        # NOTE: In here the crossover probabilities for
        # each operator are set to 1.0.
        self._items = (UniformCrossover(1.0),
                       MultiPointCrossover(1.0),
                       SinglePointCrossover(1.0))
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
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

            # Get the number of available crossovers.
            n_operators = len(self.items)

            # Select randomly, with equal probability (but this can be changed),
            # a crossover operator and call its crossover method.
            child1, child2 = self.items[self.rng.integers(n_operators,
                                                          dtype=int)].crossover(parent1, parent2)
            # Increase the crossover counter.
            self.inc_counter()
        else:
            # Otherwise each child points to a clone of a single parent.
            child1 = parent1.clone()
            child2 = parent2.clone()
        # _end_if_

        # Return the two offsprings.
        return child1, child2
    # _end_def_

    @property
    def all_counters(self) -> dict:
        """
        Accessor (getter) of the application counter from all the internal crossovers.
        This is mostly to verify that everything is working as expected.

        :return: a dictionary with the counter calls for all crossover methods.
        """
        return {cross_op.__class__.__name__: cross_op.counter for cross_op in self.items}
    # _end_def_

    def reset_counter(self) -> None:
        """
        Sets ALL the counters to 'zero'. We have to override the super().reset_counter()
        method, because we have to call explicitly the reset_counter on all the internal
        operators.

        :return: None.
        """
        # First call the super() to reset the self internal counter.
        super().reset_counter()

        # Here call explicitly the reset on each of the internal cross operators.
        for op in self.items:
            op.reset_counter()
    # _end_def_

# _end_class_
