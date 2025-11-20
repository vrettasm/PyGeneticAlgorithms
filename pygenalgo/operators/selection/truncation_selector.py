from operator import attrgetter
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator

class TruncationSelector(SelectionOperator):
    """
    Description:

        Truncation Selection is a straightforward method where only a predetermined percentage of the best
        individuals are selected based on fitness, while the rest are discarded. This ensures a consistent
        evolution of stronger solutions, accelerating convergence toward optimal fitness levels. While
        effective in honing in on high-quality individuals, truncation selection can create a loss of genetic
        diversity, risking premature convergence if the population becomes too homogeneous. The selection
        proportion can be adjusted to control the pressure, making this method adaptable for various scenarios
        but necessitating monitoring to retain essential diversity.

    """

    def __init__(self, select_probability: float = 1.0, p: float = 0.3):
        """
        Construct a 'TruncationSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].

        :param p: proportion of the population that will reproduce (float).
        """
        # Call the super constructor with the provided probability value.
        super().__init__(select_probability)

        # The proportion value should be in [0.1, 0.9].
        self._items = max(min(float(p), 0.9), 0.1)
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
        # Get the population size.
        pop_size = len(population)

        # Sort the population in descending order using their fitness value.
        sorted_population = sorted(population, key=attrgetter("fitness"), reverse=True)

        # Select tne new parents using only the higher percentage '%' of
        # the old population (indexes).
        index = self.rng.choice(int(pop_size*self._items), size=pop_size,
                                replace=True, shuffle=True)

        # Return the new parents (individuals).
        return [sorted_population[i] for i in index]
    # _end_def_

# _end_class_
