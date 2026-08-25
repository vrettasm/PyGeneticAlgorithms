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

        # Create an IslandManager to handle the operators.
        self._items: IslandManager = IslandManager(operators=mutate_ops)

        # Sanity check: correct Type.
        if any(not isinstance(op, MutationOperator) for op in mutate_ops):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"'mutate_ops' items must be of type MutationOperator.")
    # _end_def_

    def mutate(self, individual: Chromosome) -> None:
        """
        Perform the mutation operation.

        :param individual: (Chromosome).

        :return: None.
        """
        # Get the selected (mutation) operator.
        mutate_op: MutationOperator = self._items.operator()

        # Call its mutation method.
        mutate_op.mutate(individual)
    # _end_def_

    def reset_counter(self) -> None:
        """
        Sets ALL the counters to zero. We have to override
        the super().reset_counter() method because we have
        to call explicitly the reset_counter on all the
        internal operators.

        :return: None.
        """
        # First call the super() to reset the self counter.
        super().reset_counter()

        # Then call the _items to reset the island counter.
        self._items.reset_island_counters()
    # _end_def_

# _end_class_
