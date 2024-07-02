from math import fsum
from numpy import cumsum
from src.genome.chromosome import Chromosome
from src.operators.selection.select_operator import SelectionOperator

class StochasticUniversalSelector(SelectionOperator):
    """
    Description:

        Stochastic Universal Selector is an extension of fitness proportionate selection (i.e. Roulette Wheel Selection)
        which exhibits no bias and minimal spread. Where RWS chooses several solutions from the population by repeated
        random sampling, SUS uses a single random value to sample all the solutions by choosing them at evenly spaced
        intervals. This gives weaker members of the population (according to their fitness) a chance to be chosen.
    """

    def __init__(self, select_probability: float = 1.0):
        """
        Construct a 'StochasticUniversalSelector' object with a given probability value.

        :param select_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(select_probability)
    # _end_def_

    def select(self, population: list[Chromosome]):
        """
        Select the new individuals from the population that will be passed on to the next genetic
        operations of crossover and mutation to form the new population of solutions.

        :param population:

        :return: a new population (list of chromosomes).
        """

        # Extract the fitness value of each chromosome.
        # This assumes that the fitness values are all
        # positive.
        all_fitness = [p.fitness for p in population]

        # Get the size of the population.
        N = len(population)

        # Compute the distance between pointers.
        dist_p = fsum(all_fitness) / N

        # Get a random number between 0 and dist_p.
        start_0 = dist_p * self.rng.random()

        # Calculate the pointers at equal distances 'dist_p'
        # starting from 'start_0.
        pointers = [start_0 + i*dist_p for i in range(0, N)]

        # Create a list that will contain the new parents.
        new_parents = []

        # Get the list append method locally.
        new_parents_append = new_parents.append

        # Compute the cumulative sum of the fitness values.
        cum_sum_fit = cumsum(all_fitness)

        # Collect the new parents.
        for p in pointers:

            # Reset the index to '0'.
            i = 0

            # Find the cumulative value smaller than 'p'.
            while cum_sum_fit[i] < p:
                i += 1
            # _end_while_

            # Add the individual at position 'i' in the new parents pool.
            new_parents_append(population[i])
        # _end_for_

        # Increase the selection counter.
        self.inc_counter()

        # Return the (new) selected individuals.
        return self.rng.shuffle(new_parents)
    # _end_def_

# _end_class_
