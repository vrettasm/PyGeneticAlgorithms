""" Exponential rank selector module. """
from math import fsum
from typing import Callable
from operator import attrgetter
from functools import lru_cache

# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class ExponentialRankSelector(SelectionOperator):
    """
    Description:

        The ExponentialRankSelector Selector is an evolutionary selection operator.
        This operator sorts the population by individual fitness scores and assigns
        selection probabilities based on an exponentially decreasing function of
        their rank. By utilizing relative ranking instead of raw fitness values, it
        avoids scale-dependency issues (such as negative fitness values) while maintaining
        strong selective pressure on top-performing individuals to accelerate convergence.
    """

    def __init__(self, select_probability: float = 1.0, c_base: float = 0.95) -> None:
        """
        Construct a 'ExponentialRankSelector' object with a given probability
        value.

        :param select_probability: (float) in [0, 1].
        :param c_base: (float) exponential base parameter, typically in (0, 1).
                        Values closer to '0' increase selective pressure on top
                        individuals.
        """
        # Call the super constructor with the provided initial value.
        super().__init__(selection_probability=select_probability)

        # Sanity check (correct type).
        if not isinstance(c_base, float):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Exponential base must be a float.")
        # _end_if_

        # Sanity check (correct range).
        if not (0.0 < c_base < 1.0):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Exponential base must be in (0.0, 1.0). ")
        # _end_if_

        # Store the exponential base parameter.
        self._items: float = c_base
    # _end_def_

    @staticmethod
    @lru_cache(maxsize=64)
    def probabilities(pop_size: int, c_base: float) -> list[float]:
        """
        Calculate the probabilities for the population using the exponential
        rank formula. The function is lru_cached so that repeated calls with
        the same input should not recompute the same array, since the population
        size of the chromosomes is not expected to change dynamically.

        Formula for rank index 'idx' (0 to pop_size-1):
        Weight = c^(pop_size - 1 - idx)

        This gives the best individual (idx = pop_size-1) a weight of c^0 = 1,
        and the worst individual (idx = 0) a weight of c^(pop_size-1).

        :param pop_size: (int) population size.
        :param c_base: (float) exponential base parameter.

        :return: (list) probabilities in ascending order.
        """
        # Sanity check.
        if pop_size <= 0:
            raise ValueError(f"Population size {pop_size} must be > 0.")
        # _end_if_

        # Handle edge case where population size is 1.
        if pop_size == 1:
            return [1.0]
        # _end_if_

        # Calculate the weights for each rank.
        weights: list[float] = [
            c_base ** (pop_size - 1 - idx) for idx in range(pop_size)
        ]

        # Sum all the weights to compute the normalization constant.
        total_weight: float = fsum(weights)

        # Normalize weights to return a true probability distribution.
        return [w / total_weight for w in weights]
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Select the individuals, from the input population, that will be
        passed on to the next genetic operations of crossover and mutation
        to form the new population of solutions.

        :param population: a list of chromosomes to select the parents from.

        :return: the selected parents population (as list of chromosomes).
        """
        # Get the population size.
        pop_size: int = len(population)

        # Quick exit for empty or single-individual populations.
        if pop_size <= 1:
            return list(population)

        # Extract exponential base.
        c_base: float = self._items

        # Get the (cached) probability distribution.
        selection_probs: list[float] = ExponentialRankSelector.probabilities(pop_size, c_base)

        # Define the key.
        key_sort: Callable = attrgetter("fitness")

        # Sort the population in ascending order using fitness.
        # - Worst individual is at index 0 (rank 1).
        # - Best individual is at index N-1 (rank N).
        sorted_population = sorted(population, key=key_sort)

        # Select the new individuals (indexes).
        index = self.rng.choice(pop_size, size=pop_size, p=selection_probs,
                                replace=True, shuffle=False)

        # Return the new parents (individuals).
        return [sorted_population[i] for i in index]
    # _end_def_

# _end_class_
