from math import fsum
from src.genome.chromosome import Chromosome
from src.operators.selection.select_operator import SelectionOperator


class RouletteWheelSelector(SelectionOperator):
    """
    Description:

        Roulette Wheel Selector implements 'fitness proportional selection'. Its member of the population is assigned a
        probability value that is directly proportional to its fitness value (compared to the rest of the population).

        Individuals with higher fitness value are more likely to be selected for parents when forming the new generation
        of individuals (offsprings).
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
        Select the new individuals from the population that will be passed on to the next genetic operations of
        crossover and mutation to form the new population of solutions.

        :param population:

        :return: a new population (list of chromosomes).
        """

        # Extract the fitness value of each chromosome making
        # sure it is a positive value.
        abs_fitness = [abs(p._fitness) for p in population]

        # Calculate sum of all fitness.
        sum_fitness = fsum(abs_fitness)

        # Calculate the "selection probabilities", of each member
        # in the population.
        selection_probs = [f / sum_fitness for f in abs_fitness]

        # Return the (new) selected individuals.
        return population[self.rng.choice(len(population), p=selection_probs, shuffle=False)]
    # _end_def_

# _end_class_
