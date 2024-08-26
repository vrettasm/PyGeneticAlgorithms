from math import fsum
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.selection.select_operator import SelectionOperator


class RouletteWheelSelector(SelectionOperator):
    """
    Description:

        Roulette Wheel Selector implements 'fitness proportional selection'. Each member of the population
        is assigned a probability value that is directly proportional to its fitness value (compared to the
        rest of the population).

        Individuals with higher fitness value are more likely to be selected for parents when forming the
        new generation of individuals (offsprings).
    """

    def __init__(self, select_probability: float = 1.0):
        """
        Construct a 'RouletteWheelSelector' object with a given probability value.

        :param select_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(select_probability)
    # _end_def_

    def select(self, population: list[Chromosome]):
        """
        Select the new individuals from the population that will be passed on to the next
        genetic operations of crossover and mutation to form the new population of solutions.

        :param population:

        :return: a new population (list of chromosomes).
        """

        # Get the length of the population list.
        N = len(population)

        # Extract the fitness value of each chromosome.
        # This assumes that the fitness values are all
        # positive.
        all_fitness = [p.fitness for p in population]

        # Calculate sum of all fitness.
        sum_fitness = fsum(all_fitness)

        # Calculate the "selection probabilities", of each member
        # in the population.
        selection_probs = [f / sum_fitness for f in all_fitness]

        # Select 'N' new individuals (indexes).
        index = self.rng.choice(N, size=N, p=selection_probs, replace=True, shuffle=False)

        # Increase the selection counter.
        self.inc_counter()

        # Return the (new) selected individuals.
        return [population[i] for i in index]

    # _end_def_

# _end_class_
