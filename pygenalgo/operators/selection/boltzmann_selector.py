import numpy as np
from math import fsum
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.selection.select_operator import SelectionOperator


class BoltzmannSelector(SelectionOperator):
    """
    Description:

        Boltzmann Selector implements an object that performs selection by choosing an individual
        from a set of individuals by sampling solutions from a Boltzmann distribution depending on
        their fitnesses.
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
        self._k = float(k) if k > 50.0 else 50.0
    # _end_def_

    def select(self, population: list[Chromosome]):
        """
        Select the individuals, from the input population, that will be passed on to the next
        genetic operations of crossover and mutation to form the new population of solutions.

        :param population: a list of chromosomes to select the parents from.

        :return: the selected parents population (as list of chromosomes).
        """
        # Compute the 'T'emperature.
        T = max(0.1, np.exp(-self.iter/self._k))

        # Get the length of the population list.
        N = len(population)

        # Extract the fitness value of each chromosome.
        # This assumes that the fitness values are all
        # positive.
        exp_fitness = [np.exp(-p.fitness/T) for p in population]

        # Calculate sum of all fitness.
        sum_fitness = fsum(exp_fitness)

        # Calculate the "selection probabilities", of each member
        # in the population.
        selection_probs = [f / sum_fitness for f in exp_fitness]

        # Select 'N' new individuals (indexes).
        index = self.rng.choice(N, size=N, p=selection_probs, replace=True, shuffle=False)

        # Increase the selection counter.
        self.inc_counter()

        # Return the new parents (individuals).
        return [population[i] for i in index]
    # _end_def_

# _end_class_