# Standard imports.
from functools import lru_cache

# Third party imports.
from numpy import arange
from numpy.typing import NDArray

# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class SwapMutator(MutationOperator):
    """
    Description:

        Swap mutator mutates the chromosome by swapping the gene
        values between two randomly selected gene positions.
    """

    def __init__(self, mutate_probability: float = 0.1) -> None:
        """
        Construct a 'SwapMutator' object with a given probability value.

        :param mutate_probability: (float).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(mutation_probability=mutate_probability)
    # _end_def_

    @staticmethod
    @lru_cache(maxsize=64)
    def _cached_range(num: int) -> NDArray:
        """
        Cached range function.

        :param num: max range value.

        :return: NDArray (0, 1, ..., num-1).
        """
        return arange(num, dtype=int)
    # _end_def_

    def mutate(self, individual: Chromosome) -> None:
        """
        Perform the mutation operation by swapping
        the genes at two random positions.

        :param individual: (Chromosome).

        :return: None.
        """
        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.is_operator_applicable():

            # Make an array with all the indexes.
            indexes: NDArray = SwapMutator._cached_range(len(individual))

            # Shuffle them in place.
            self.rng.shuffle(indexes)

            # Select the first two values.
            i, j = indexes[0:2]

            # Swap in place between the two positions.
            individual[i], individual[j] = individual[j], individual[i]

            # Set the fitness to NaN.
            individual.invalidate_fitness()

            # Increase the mutator counter.
            self.inc_counter()
    # _end_def_

# _end_class_
