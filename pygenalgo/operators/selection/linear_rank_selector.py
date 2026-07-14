from math import fsum
from operator import attrgetter
from functools import lru_cache
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class LinearRankSelector(SelectionOperator):
    """
    Description:

        The LinearRank Selector is a selection mechanism in genetic algorithms
        that ranks individuals based on their fitness values. It assigns selection
        probabilities to candidates in a linear fashion, ensuring that higher-ranking
        individuals are more likely to be chosen for reproduction. The operator helps
        maintain diversity in the population by avoiding premature convergence, allowing
        lower-ranked individuals a chance to contribute. This method is particularly
        effective in scenarios with varying fitness levels, as it balances selection
        pressure while fostering exploration of the solution space. Overall, it enhances
        the optimization process in evolutionary computation models.

    """

    def __init__(self, select_probability: float = 1.0, eta: float = 2.0) -> None:
        """
        Construct a 'LinearRankSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].
        :param eta: (float) pressure adjustment factor [1, 2]. A value of 1
                    provides the lowest selection pressure (uniform random)
                    whilst a value of 2, will provide the highest selection
                    pressure.
        """
        # Call the super constructor with the provided initial value.
        super().__init__(selection_probability=select_probability)

        # Sanity check for selection pressure parameter.
        if not (1.0 <= eta <= 2.0):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Selection pressure {eta} must be between 1.0 and 2.0.")
        # _end_if_

        # Store the pressure adjustable parameter.
        self._items: float = float(eta)
    # _end_def_

    @staticmethod
    @lru_cache(maxsize=64)
    def probabilities(pop_size: int, eta: float) -> list[float]:
        """
        Calculate the rank probability distribution over the population size.
        The function is lru_cached so that repeated calls with the same input
        should not recompute the same array, since the population size of the
        chromosomes is not expected to change dynamically. The eta parameter
        provides a means to adjust the pressure from total random (eta = 1.0)
        to the highest possible rank (eta = 2.0).

        NOTE: Probabilities are returned in ascending order.

        :param pop_size: (int) population size.
        :param eta: (float) pressure adjustment factor.

        :return: (list) rank probability distribution in ascending order.
        """
        # Sanity check.
        if pop_size <= 0:
            raise ValueError(f"Population size {pop_size} must be > 0.")
        # _end_if_

        # Handle edge case where population size is 1.
        if pop_size == 1:
            return [1.0]
        # _end_if_

        # Precompute constant invariants out of the
        # loop to eliminate repetitive division.
        base: float = (2.0 - eta) / pop_size
        step: float = (2.0 * (eta - 1.0)) / (pop_size * (pop_size - 1))

        # Calculate the probabilities.
        probabilities: list[float] = [
            base + (i * step) for i in range(pop_size)
        ]

        # Calculate the sum of probabilities.
        total: float = fsum(probabilities)

        # Normalize for minor numerical safety. The
        # probability values are in ascending order.
        return [p / total for p in probabilities]
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
        pop_size: int = len(population)

        # Calculate the selection probabilities of each member
        # in the population, using their ranking position. Use
        # the pressure adjustment parameter 'eta' (as _items).
        selection_probs = LinearRankSelector.probabilities(pop_size,
                                                           self._items)

        # Sort the population in ascending order using their fitness value.
        sorted_population = sorted(population, key=attrgetter("fitness"))

        # Select the new individuals (indexes).
        index = self.rng.choice(pop_size, size=pop_size, p=selection_probs,
                                replace=True, shuffle=False)

        # Return the new parents (individuals).
        return [sorted_population[i] for i in index]
    # _end_def_

# _end_class_
