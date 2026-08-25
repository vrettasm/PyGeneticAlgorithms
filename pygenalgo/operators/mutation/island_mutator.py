""" Island-mutator module. """

# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.island_manager import IslandManager
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class IslandMutator(MutationOperator):
    """
    Description:

        Island-mutator performs the mutation of individuals (offspring) for each
        island. The object holds a list of MutationOperator objects along with a
        pointer (int) that is unique for each island.

        Once the pointer is set, the mutation is performed using the predefined
        MutationOperator that corresponds to the specific island.
    """

    def __init__(self, mutate_probability: float = 0.1,
                 mutate_ops: list[MutationOperator] = None) -> None:
        """
        Construct an 'IslandMutator' object with a predefined probability
        value and a list of MutationOperator objects.

        :param mutate_probability: (float) for compatibility only.
        :param mutate_ops: a list of MutationOperator objects.

        :return: None.
        """
        # Call the super() constructor with an initial probability.
        # Note: This probability is never used. We use directly the
        # probability values of the genetic operators in the _items.
        super().__init__(mutation_probability=mutate_probability)

        # Sanity check.
        if mutate_ops is None or any(not isinstance(op, MutationOperator)
                                     for op in mutate_ops):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"'mutate_ops' items must be of type MutationOperator.")
        # _end_if_

        # Create an IslandOperator.
        self._items: IslandOperator = IslandOperator(operators=mutate_ops)
    # _end_def_

    def mutate(self, individual: Chromosome) -> None:
        """
        Perform the mutation operation.

        :param individual: (Chromosome).

        :return: None.
        """
        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.is_operator_applicable():
            # Local reference of island operator.
            island_op = self._items

            # Get the selected operator.
            mutate_op: MutationOperator = island_op.operator[island_op.idx]

            # Call its mutation method.
            mutate_op.mutate(individual)

            # Increase the mutator counter.
            self.inc_counter()
    # _end_def_

    def reset_counter(self) -> None:
        """
        Sets ALL the counters to zero. We have to override
        the super().reset_counter() method because we have
        to call explicitly the reset_counter on all the
        internal operators.

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
