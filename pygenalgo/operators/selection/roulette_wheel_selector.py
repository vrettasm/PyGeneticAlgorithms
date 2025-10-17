from math import fsum
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class RouletteWheelSelector(SelectionOperator):
    """
    Description:

        Roulette Wheel Selector implements 'fitness proportional selection'. Each member
        of the population is assigned a probability value that is directly proportional
        to its fitness value (compared to the rest of the population).

        Individuals with higher fitness value are more likely to be selected for parents
        when forming the new generation of individuals (offsprings).
    """

    def __init__(self, select_probability: float = 1.0):
        """
        Construct a 'RouletteWheelSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].
        """

        # Call the super constructor with the provided probability value.
        super().__init__(select_probability)
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]):
        """
        Select the individuals, from the input population, that will be passed on
        to the next genetic operations of crossover and mutation to form the new
        population of solutions.

        :param population: a list of chromosomes to select the parents from.

        :return: the selected parents population (as list of chromosomes).
        """

        # Get the population size.
        pop_size = len(population)

        # Extract the fitness value of each chromosome.
        # This assumes that the fitness values are all
        # positive.
        all_fitness = [p.fitness for p in population]

        # Calculate sum of all fitness.
        sum_fitness = fsum(all_fitness)

        # Calculate the "selection probabilities" of each member
        # in the population.
        selection_probs = [f / sum_fitness for f in all_fitness]

        # Select the new individuals (indexes).
        index = self.rng.choice(pop_size, size=pop_size, p=selection_probs,
                                replace=True, shuffle=False)

        # Return the new parents (individuals).
        return [population[i] for i in index]
    # _end_def_

# _end_class_
