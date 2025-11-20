from operator import attrgetter
from functools import lru_cache
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class LinearRankSelector(SelectionOperator):
    """
    Description:

        Linear Rank Selector implements an object that performs selection using ranking.
        The individuals first are sorted according to their fitness values. The rank 'N'
        is assigned to the best individual and the rank 1 to the worst individual.

        After that the selection process is similar to the one of RouletteWheelSelector.
    """

    def __init__(self, select_probability: float = 1.0):
        """
        Construct a 'LinearRankSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].
        """
        # Call the super constructor with the provided probability value.
        super().__init__(select_probability)
    # _end_def_

    @staticmethod
    @lru_cache(maxsize=32)
    def probabilities(p_size: int) -> list[float]:
        """
        Calculate the rank probability distribution over the population size.
        The function is lru_cached so that repeated calls with the same input
        should not recompute the same array, since the population size of the
        chromosomes is not expected to change dynamically.

        NOTE: Probabilities are returned in ascending order.

        :param p_size: (int) population size.

        :return: (list) rank probability distribution in ascending order.
        """
        # Sanity check.
        if p_size <= 0:
            raise ValueError("Population size 'p_size' must be > 0.")
        # _end_if_

        # Calculate the sum of '1 + 2 + 3 + ... + N'.
        # We know that this is equal to: N * (N+1)/2.
        sum_ranked_values = float(0.5 * p_size * (p_size + 1))

        # Return the probability values (ascending order).
        return [n / sum_ranked_values for n in range(1, p_size + 1)]
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

        # Calculate the selection probabilities of each member
        # in the population, using their ranking position.
        selection_probs = LinearRankSelector.probabilities(pop_size)

        # Sort the population in ascending order using their fitness value.
        sorted_population = sorted(population, key=attrgetter("fitness"))

        # Select the new individuals (indexes).
        index = self.rng.choice(pop_size, size=pop_size, p=selection_probs,
                                replace=True, shuffle=False)

        # Return the new parents (individuals).
        return [sorted_population[i] for i in index]
    # _end_def_

# _end_class_
