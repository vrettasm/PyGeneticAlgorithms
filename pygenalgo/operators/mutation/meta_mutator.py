from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.swap_mutator import SwapMutator
from pygenalgo.operators.mutation.random_mutator import RandomMutator
from pygenalgo.operators.mutation.shuffle_mutator import ShuffleMutator
from pygenalgo.operators.mutation.inverse_mutator import InverseMutator
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class MetaMutator(MutationOperator):
    """
    Description:

        Meta-mutator, mutates the chromosome by applying randomly
        all other mutators (one at a time), with equal probability.

        NOTE: In the future the equal probabilities can be amended.
    """

    def __init__(self, mutate_probability: float = 0.1):
        """
        Construct a 'MetaMutator' object with a predefined probability value.

        :param mutate_probability: (float).
        """
        # Call the super constructor with the provided probability value.
        super().__init__(mutate_probability)

        # NOTE: In here the mutation probabilities for each mutator are set to 1.0.
        self._items = (SwapMutator(1.0), RandomMutator(1.0), ShuffleMutator(1.0), InverseMutator(1.0))
    # _end_def_

    def mutate(self, individual: Chromosome) -> None:
        """
        Perform the mutation operation by randomly applying another mutator.

        :param individual: (Chromosome).

        :return: None.
        """
        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.is_operator_applicable():

            # Get the number of available mutators.
            n_operators = len(self.items)

            # Select randomly with equal probability
            # a mutator and call its mutation method.
            self.items[self.rng.integers(n_operators)].mutate(individual)

            # Increase the mutator counter.
            self.inc_counter()
    # _end_def_

    @property
    def all_counters(self) -> dict:
        """
        Accessor (getter) of the application counter from all the internal mutators.
        This is mostly to verify that everything is working as expected.

        :return: a dictionary with the counter calls for all mutator methods.
        """
        return {mut_op.__class__.__name__: mut_op.counter for mut_op in self.items}
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

        # Here call explicitly the reset on each of the internal mutators.
        for op in self.items:
            op.reset_counter()
    # _end_def_

# _end_class_
