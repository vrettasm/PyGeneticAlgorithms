from typing import Generator
from math import fsum, isclose
from bisect import bisect_left
from itertools import accumulate

from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import (SelectionOperator,
                                                           ensure_positive_fitness)


class StochasticUniversalSelector(SelectionOperator):
    """
    Description:

        Stochastic Universal Selector is an extension of fitness proportionate selection
        (i.e. RouletteWheelSelection) which exhibits no bias and minimal spread. Where RWS
        chooses several solutions from the population by repeated random sampling, SUS uses
        a single random value to sample all the solutions by choosing them at evenly spaced
        intervals. This gives weaker members of the population (according to their fitness)
        a chance to be chosen.
    """

    def __init__(self, select_probability: float = 1.0) -> None:
        """
        Construct a 'StochasticUniversalSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].
        """
        # Call the super constructor with the provided initial value.
        super().__init__(selection_probability=select_probability)
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
        # Extract the (positive) fitness values from the chromosomes.
        all_fitness: list[float] = ensure_positive_fitness(population)

        # Get the population size.
        pop_size: int = len(population)

        # Calculate sum of all fitness.
        sum_fitness: float = fsum(all_fitness)

        # If total fitness is zero (or effectively zero),
        # fall back to uniform random selection so every
        # individual has equal chance.
        if isclose(sum_fitness, 0.0):
            # Select the new individuals with equal probability.
            safe_index = self.rng.choice(pop_size,
                                         size=pop_size, replace=True)

            # Return the new parents to a list.
            return [population[i] for i in safe_index]
        # _end_if_

        # Distance between pointers.
        dist_p: float = sum_fitness / pop_size

        # Get a random number between 0 and dist_p.
        start_0: float = dist_p * self.rng.random()

        # Create a generator to calculate the pointers at
        # equal distances 'dist_p' starting from 'start_0'.
        pointers: Generator[float, None, None] = (
            start_0 + i*dist_p for i in range(pop_size)
        )

        # Compute the cumulative sum of the fitness values.
        cum_sum_fit: list[float] = list(accumulate(all_fitness))

        # Use optimized C-level binary search to extract individuals.
        new_parents: list[Chromosome] = [
            population[min(bisect_left(cum_sum_fit, p), pop_size - 1)]
            for p in pointers
        ]

        # Return the new parents (individuals).
        return new_parents
    # _end_def_

# _end_class_
