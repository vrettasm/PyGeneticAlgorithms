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

            # Select randomly, with equal probability
            # (but this can be changed) a mutator and
            # call its mutation method.
            self.rng.choice(self.mutator).mutate(individual)

            # Increase the mutator counter.
            self.inc_counter()
        # _end_if_

    # _end_def_

    @property
    def all_counters(self):
        """
        Accessor (getter) of the application counter from all the internal mutators.
        This is mostly to verify that everything is working as expected.

        :return: a dictionary with the counter calls for all mutator methods.
        """
        return {mut_op.__class__.__name__: mut_op.counter for mut_op in self.mutator}
    # _end_def_

    def reset_counter(self):
        """
        Sets ALL the counters to 'zero'. We have to override the super().reset_counter()
        method, because we have to call explicitly the reset_counter on all the internal
        operators.

        :return: None.
        """

        # First call the super() to reset the self internal counter.
        super().reset_counter()

        # Here call explicitly the reset on each of the mutator operators.
        for op in self.mutator:
            op.reset_counter()
        # _end_for_

    # _end_def_

# _end_class_
