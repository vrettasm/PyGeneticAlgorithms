""" Island-mutator module. """
# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class IslandMutator(MutationOperator):
    """
    Description:

        Meta-mutator, mutates the chromosome by applying randomly
        all other mutators (one at a time), with equal probability.

        NOTE: In the future the equal probabilities can be amended.
    """

    def __init__(self, mutate_probability: float = 0.1,
                 mutate_ops: list[MutationOperator] = None) -> None:
        """
        Construct an 'IslandMutator' object with a predefined probability
        value and a list of MutationOperator objects.

        :param mutate_probability: (float).
        :param mutate_ops: a list of MutationOperator objects.

        :return: None.
        """
        # Call the super constructor with the provided initial value.
        super().__init__(mutation_probability=mutate_probability)

        # Sanity check.
        if mutate_ops is None or len(mutate_ops) == 0:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"'mutate_ops' is missing or empty.")
        # _end_if_

        # Sanity check.
        if any(not isinstance(op, MutationOperator) for op in mutate_ops):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"'mutate_ops' items must be of type MutationOperator.")
        # _end_if_

        # Copy the variables locally.
        self._items = {"operators": mutate_ops, "idx": 0}
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
            # Get the pointer value.
            idx: int = self._items["idx"]

            # Get the list of mutators.
            mutate_op: list[MutationOperator] = self._items["operators"]

            # Call its mutation method.
            mutate_op[idx].mutate(individual)

            # Increase the mutator counter.
            self.inc_counter()
    # _end_def_

    @property
    def all_counters(self) -> dict:
        """
        Accessor (getter) of the application counter from all
        the internal mutators. This is mostly to verify that
        everything is working as expected.

        :return: a dictionary with the counter calls for all
                 mutator methods.
        """
        return {
            mut_op.__class__.__name__: mut_op.counter
            for mut_op in self._items["operators"]
        }
    # _end_def_

    def reset_counter(self) -> None:
        """
        Sets ALL the counters to zero. We have to override
        the super().reset_counter() method because we have
        to call explicitly the reset_counter on all the
        internal operators.

        :return: None.
        """
        # First call the super() to reset the self internal counter.
        super().reset_counter()

        # Here call explicitly the reset on each of the internal mutators.
        for op in self._items["operators"]:
            op.reset_counter()
    # _end_def_

# _end_class_
