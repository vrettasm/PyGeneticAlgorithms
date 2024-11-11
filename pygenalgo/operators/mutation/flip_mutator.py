from numpy import nan as np_nan
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class FlipMutator(MutationOperator):
    """
    Description:

        Flip mutator, mutates the chromosome by selecting randomly
        a position and flip its Gene value (0 -> 1, or 1 -> 0).
    """

    def __init__(self, mutate_probability: float = 0.1):
        """
        Construct a 'FlipMutator' object with a given probability value.

        :param mutate_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(mutate_probability)

    # _end_def_

    def mutate(self, individual: Chromosome) -> None:
        """
        Perform the mutation operation by randomly flipping a gene.

        :param individual: (Chromosome).

        :return: None.
        """

        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.probability > self.rng.random():

            # Select randomly the mutation point.
            locus = self.rng.integers(low=0, high=len(individual))

            # Flip the old gene value.
            individual[locus].flip()

            # Invalidate the fitness of the chromosome.
            individual.fitness = np_nan

            # Increase the mutator counter.
            self.inc_counter()
        # _end_if_

    # _end_def_

# _end_class_
