""" Pareto tournament selector module. """
# Third party imports.
import numpy as np
from numpy.typing import NDArray

# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.utilities import np_pareto_front_index
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class ParetoTournamentSelector(SelectionOperator):
    """
    Description:
        This selector follows the same logic of the TournamentSelector
        but instead of choosing the one with the optimal fitness value
        it passes the contestants to the pareto_front_index function,
        and selects a random one that lies on the Pareto front.
    """

    def __init__(self, select_probability: float = 1.0,
                 n_contestants: int = 2) -> None:
        """
        Construct a 'ParetoTournamentSelector' object with a
        given probability value.

        :param select_probability: (float) in [0, 1].

        :param n_contestants: the number of participants in
                              the tournament (int).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(selection_probability=select_probability)

        # Set the value of the contestants in the placeholder _items.
        self._items: int = max(2, int(n_contestants))
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Select the individuals from the population that will be passed
        on to the next genetic operations of crossover and mutation to
        form the new population of solutions.
        """
        # Build a 2D array from the fitness
        # tuples (optimization objectives).
        fitness_array: NDArray = np.array([
            p.fitness for p in population
        ], dtype=float)

        # Total size of the population.
        n_size: int = len(population)

        # Local copy of random choice.
        choose_randomly = self.rng.choice

        # Local number of contestants. Ensure that this
        # number is not higher than the population size.
        n_contestants: int = min(self._items, n_size)

        # Select the contestants in one call.
        contestants: NDArray = np.array([
            # Set 'replace=False' to avoid duplicates.
            choose_randomly(n_size, size=n_contestants,
                            replace=False, shuffle=False)
            for _ in range(n_size)
        ], dtype=int)

        # Preallocate the chosen list.
        chosen: list[int] = n_size * [None]

        # Select the new indices iteratively.
        for i, row in enumerate(contestants):

            # Get the indexes on the Pareto front.
            pf_idx = np_pareto_front_index(fitness_array[row])

            # If more than one, choose at random.
            idx = choose_randomly(pf_idx) if pf_idx.size > 1 else pf_idx[0]

            # Dereference the position in the
            # population through "row" vector.
            chosen[i] = row[idx]
        # _end_for_

        return [
            # Ensure 'k' is used as integer.
            population[int(k)] for k in chosen
        ]
    # _end_def_

# _end_class_
