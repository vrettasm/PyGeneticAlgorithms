from src.genome.chromosome import Chromosome
from src.operators.mutation.swap_mutator import SwapMutator
from src.operators.mutation.random_mutator import RandomMutator
from src.operators.mutation.shuffle_mutator import ShuffleMutator
from src.operators.mutation.mutate_operator import MutationOperator


class SuperMutator(MutationOperator):
    """
    Description:

        Super mutator, mutates the chromosome by applying randomly
        all other mutators (one at a time), with equal probability.
    """

    # Tuple with all the other mutators. Note that in here the mutation
    # probabilities for each mutator are set to 1.0.
    mutator = (SwapMutator(1.0), RandomMutator(1.0), ShuffleMutator(1.0))

    def __init__(self, mutate_probability: float = 0.1):
        """
        Construct a 'SuperMutator' object with a given
        probability value.

        :param mutate_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(mutate_probability)

    # _end_def_

    def mutate(self, individual: Chromosome):
        """
        Perform the mutation operation by randomly applying another mutator.

        :param individual: (Chromosome).

        :return: None.
        """

        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.probability >= self.rng.random():

            # Select randomly the mutation method.
            index = self.rng.integers(len(self.mutator))

            # Call the selected mutator.
            self.mutator[index].mutate(individual)

            # Increase the mutator counter.
            self.inc_counter()
        # _end_if_

    # _end_def_

# _end_class_
