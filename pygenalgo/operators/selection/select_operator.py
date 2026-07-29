from math import fabs, isclose
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import GeneticOperator

# Public interface.
__all__ = ["SelectionOperator", "ensure_positive_fitness"]


def _shift_up_values(population: list[Chromosome]) -> list[float]:
    """
    Ensures that the fitness value of each chromosome is a positive number.

    This is useful because some of the selection methods require a positive
    fitness to operate.

    :param population: (list) of chromosomes.

    :return: (list) of positive fitness values.
    """
    # Extract all the fitness values.
    all_fitness: list[float] = [
        p.fitness for p in population
    ]

    # Sanity check.
    if not all_fitness:
        return []

    # If there are negative values we perform a shift
    # transformation where all the values are shifted
    # so that the minimum fitness is going to be one.
    if any(fit_value <= 0.0 for fit_value in all_fitness):
        # Compute the shift value.
        shift_value: float = fabs(min(all_fitness)) + 1.0

        # Shift all fitness values so that the minimum is '1'.
        all_fitness = [f + shift_value for f in all_fitness]
    # _end_if_

    return all_fitness
# _end_def_

def _linear_scaled_values(population: list[Chromosome]) -> list[float]:
    """
    Ensures fitness values are positive for maximization selection methods.
    The worst individual is assigned a baseline fitness of 1.0, and others
    are scaled relative to it.

    :param population: (list) of chromosomes.

    :return: (list) of positive fitness values
    """
    # Extract all the fitness values.
    all_fitness: list[float] = [
        p.fitness for p in population
    ]

    # Sanity check.
    if not all_fitness:
        return []

    # Extract min/max values.
    min_fitness = min(all_fitness)
    max_fitness = max(all_fitness)

    # If all individuals have the exact same
    # fitness, give them equal weight.
    if isclose(min_fitness, max_fitness):
        return [1.0] * len(population)

    # Shift everything so the worst individual is always 1.0.
    # This works whether values are positive, negative, zero.
    return [
        f - min_fitness + 1.0 for f in all_fitness
    ]
# _end_def_

def ensure_positive_fitness(population: list[Chromosome],
                            mode: str = "shift_up") -> list[float]:
    """
    Ensures fitness values are positive, using a predefined
    selected method.

    :param population: (list) of chromosomes.

    :param mode: (str) method of calculating fitness.

    :return: (list) of fitness values.
    """

    if mode == "shift_up":
        return _shift_up_values(population)

    if mode == "linear_scaled":
        return _linear_scaled_values(population)

    raise ValueError(f"{mode} is not implemented.")
# _end_if_


class SelectionOperator(GeneticOperator):
    """
    Description:

        Provides the base class (interface) for a Selection Operator.
        Note that even though the operator accepts a probability value,
        for the moment this operator is applied with 100% 'probability'.
    """

    def __init__(self, selection_probability: float) -> None:
        """
        Construct a 'SelectionOperator' object with a
        given probability value.

        :param selection_probability: (float).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(probability=selection_probability)
    # _end_def_

    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Abstract method that "reminds" the user that if they want to
        create a Selection Class that inherits from here they should
        implement a select method.

        :param population: is a list, with the chromosomes, to select
                           he parents for the next generation

        :return: Nothing but raising an error.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")
    # _end_def_

    def __call__(self, *args, **kwargs) -> list[Chromosome]:
        """
        This is only a wrapper of the "select" method.
        """
        return self.select(*args, **kwargs)
    # _end_def_

# _end_class_
