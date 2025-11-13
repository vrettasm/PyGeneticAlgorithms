from math import fsum
from numpy import exp as np_exp
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class BoltzmannSelector(SelectionOperator):
    """
    Description:

        Boltzmann Selector implements an object that performs selection by choosing an
        individual from a set of individuals by sampling solutions from a Boltzmann
        distribution depending on their fitness's.
    """

    def __init__(self, select_probability: float = 1.0, k: float = 100.0):
        """
        Construct a 'BoltzmannSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].

        :param k: constant of the Boltzmann distribution.
        """

        # Call the super constructor with the provided probability value.
        super().__init__(select_probability)

        # Make sure 'k' is float and at least 50.
        self._items = max(float(k), 50.0)
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Select the individuals, from the input population, that will be
        passed on to the next genetic operations of crossover and mutation
        to form the new population of solutions.

        NOTE: the Boltzmann constant is held in the '_items' variable.

        :param population: a list of chromosomes to select the parents from.

        :return: the selected parents population (as list of chromosomes).
        """
        # Compute the Temperature.
        temperature = max(0.1, np_exp(-self.iteration / self._items))

        # Get the population size.
        pop_size = len(population)

        # Extract the fitness value of each chromosome.
        # This assumes that the fitness values are all positive.
        exp_fitness = np_exp([-p.fitness/temperature for p in population]).tolist()

        # Calculate sum of all fitness.
        sum_fitness = fsum(exp_fitness)

        # Calculate the selection probabilities of each member
        # in the population (Boltzmann distribution).
        selection_probs = [f/sum_fitness for f in exp_fitness]

        # Select the new individuals indexes.
        index = self.rng.choice(pop_size, size=pop_size, p=selection_probs,
                                replace=True, shuffle=False)

        # Return the new parents (individuals).
        return [population[i] for i in index]
    # _end_def_

# _end_class_
