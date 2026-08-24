""" Island-crossover module. """
# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
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
        # Call the super constructor with the provided initial value.
        super().__init__(crossover_probability=crossover_probability)

        # Sanity check.
        if crossx_ops is None or len(crossx_ops) == 0:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"'crossx_ops' is missing or empty.")
        # _end_if_

        # Sanity check.
        if any(not isinstance(op, CrossoverOperator) for op in crossx_ops):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"'crossx_ops' items must be of type CrossoverOperator.")
        # _end_if_

        # Copy the variables locally.
        self._items = {"operators": crossx_ops, "idx": 0}
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
            # Get the pointer value.
            idx: int = self._items["idx"]

            # Get the list of mutators.
            crossx_op: list[CrossoverOperator] = self._items["operators"]

            # Increase the crossover counter.
            self.inc_counter()

            # Call its crossover method.
            return crossx_op[idx].crossover(parent1, parent2)
        # _end_if_

        # Return the two offsprings.
        return parent1.clone(), parent2.clone()
    # _end_def_

    def set_pointer(self, idx: int) -> None:
        """
        Set the pointer to the crossover operator.

        :param idx: the index of the crossover operator.

        :return: None.
        """
        # Sanity check.
        if idx < 0 or idx >= len(self._items["operators"]):
            raise IndexError(f"{self.__class__.__name__}: "
                             f"selected index out of range.")
        # _end_if_

        # Update the index in the dict.
        self._items["idx"] = idx
    # _end_def_

    @property
    def all_counters(self) -> dict:
        """
        Accessor (getter) of the application counter from all
        the internal crossovers. This is mostly to verify that
        everything is working as expected.

        :return: a dictionary with the counter calls for all
                 crossover methods.
        """
        return {
            f"{n}-{crossx_op.__class__.__name__}": crossx_op.counter
            for n, crossx_op in enumerate(self._items["operators"])
        }
    # _end_def_

    def reset_counter(self) -> None:
        """
        Sets ALL the counters to 'zero'. We have to override the
        super().reset_counter() method,  because we have to call
        explicitly the reset_counter on all the internal operators.

        :return: None.
        """
        # First call the super() to reset the self internal counter.
        super().reset_counter()

        # Here call explicitly the reset on each of the internal crossx operators.
        for op in self._items["operators"]:
            op.reset_counter()
    # _end_def_

# _end_class_
