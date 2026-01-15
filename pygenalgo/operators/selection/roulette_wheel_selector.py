from math import fsum
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class RouletteWheelSelector(SelectionOperator):
    """
    Description:

        Roulette Wheel Selection employs a probabilistic mechanism where individuals are
        selected based on their fitness relative to the entire population. Each individual
        receives a slice of a wheel proportional to its fitness, similar to a casino roulette,
        where a random number determines the selected individual. This method encourages selection
        of fitter individuals while allowing lower-fitness individuals a chance to contribute.
        While effective, it can lead to premature convergence if the population's fitness is skewed,
        often referred to as the "selection pressure."

    """

    def __init__(self, select_probability: float = 1.0) -> None:
        """
        Construct a 'RouletteWheelSelector' object with a given probability value.

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
