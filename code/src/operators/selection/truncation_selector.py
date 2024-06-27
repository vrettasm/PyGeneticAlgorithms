from src.genome.chromosome import Chromosome
from src.operators.selection.select_operator import SelectionOperator

class TruncationSelector(SelectionOperator):
    """
    Description:
        TBD
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
        Select the new individuals from the population that will be passed on to the next genetic
        operations of crossover and mutation to form the new population of solutions.

        :param population:

        :return: a new population (list of chromosomes).
        """

        # Get the length of the population list.
        N = len(population)

        # Sort the population in descending order using their fitness value.
        sorted_population = sorted(population, key=lambda p: p._fitness, reverse=True)

        # Increase the selection counter.
        self.inc_counter()

        # Return the (new) selected individuals, using only the higher 'p%'
        # of the (old) population.
        return self.rng.choice(sorted_population[0:int(N*self._p)],
                               size=N, replace=True, shuffle=True).tolist()
    # _end_def_

# _end_class_
