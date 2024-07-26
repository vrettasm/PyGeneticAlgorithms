from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.selection.select_operator import SelectionOperator

class TruncationSelector(SelectionOperator):
    """
    Description:

        Truncation selector, creates a new population using a pre-defined proportion of the old population.
        When this method is called, it sorts the individuals of the OLD population using their fitness and
        then using a predefined value (e.g. p=0.3 or 30%) selects repeatedly new individuals from the top
        0.3 percent of the old population, until we reach the required size of the NEW population.
    """

    def __init__(self, select_probability: float = 1.0, p: float = 0.3):
        """
        Construct a 'TruncationSelector' object with a given probability value.

        :param select_probability: (float).

        :param p: proportion of the population that will reproduce (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(select_probability)

        # The proportion value should be in [0.1, 0.9].
        self._p = max(min(float(p), 0.9), 0.1)
    # _end_def_

    def select(self, population: list[Chromosome]):
        """
        Select the new individuals from the population that will be passed on to the next
        genetic operations of crossover and mutation to form the new population of solutions.

        :param population: List of Chromosomes.

        :return: a new population (list of chromosomes).
        """

        # Get the length of the population list.
        N = len(population)

        # Sort the population in descending order using their fitness value.
        sorted_population = sorted(population, key=lambda p: p.fitness, reverse=True)

        # Select 'N' using only the higher 'p%' of the (old) population (indexes).
        index = self.rng.choice(int(N*self._p), size=N, replace=True, shuffle=True)

        # Increase the selection counter.
        self.inc_counter()

        # Return the (new) selected individuals.
        return [sorted_population[i].clone() for i in index]
    # _end_def_

# _end_class_
