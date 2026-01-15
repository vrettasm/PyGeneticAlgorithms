from math import fsum
from collections import deque
from itertools import accumulate
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


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
        # Call the super constructor with the provided probability value.
        super().__init__(select_probability)
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
        all_fitness = self.ensure_positive_fitness(population)

        # Get the population size.
        pop_size = len(population)

        # Compute the distance between pointers.
        dist_p = fsum(all_fitness) / pop_size

        # Get a random number between 0 and dist_p.
        start_0 = dist_p * self.rng.random()

        # Calculate the pointers at equal distances 'dist_p'
        # starting from 'start_0'.
        pointers = (start_0 + i*dist_p for i in range(pop_size))

        # Create a list that will contain the new parents.
        new_parents = deque(maxlen=pop_size)

        # Compute the cumulative sum of the fitness values.
        cum_sum_fit = list(accumulate(all_fitness))

        # Collect the new parents.
        for p in pointers:

            # Reset the index to '0'.
            i = 0

            # Find the cumulative value smaller than 'p'.
            while cum_sum_fit[i] < p:
                i += 1
            # _end_while_

            # Add the individual at position 'i' in the new parents pool.
            new_parents.append(population[i])
        # _end_for_

        # Return the new parents (individuals).
        return list(new_parents)
    # _end_def_

# _end_class_
