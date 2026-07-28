import numpy as np
from numpy.typing import NDArray

from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.utilities import np_pareto_front_index
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class ParetoTournamentSelector(SelectionOperator):
    """
    Description:

        TBD ...

    """

    def __init__(self, select_probability: float = 1.0,
                 n_contestants: int = 5) -> None:
        """
        Construct a 'ParetoTournamentSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].

        :param n_contestants: the number of participants in the tournament (int).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(selection_probability=select_probability)

        # Set the value of the contestants in the placeholder _items.
        self._items: int = max(5, int(n_contestants))
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Select the individuals, from the input population that will be passed on
        to the next genetic operations of crossover and mutation to form the new
        population of solutions.
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

        # Local number of contestants.
        n_contestants: int = self._items

        # Select the contestants in one call.
        contestants: NDArray = np.array([
            # Set 'replace=False' to avoid duplicates.
            choose_randomly(n_size, size=n_contestants,
                            replace=False, shuffle=False)
            for _ in range(n_size)
        ], dtype=int)

        # Select the new indices via Tournament selection.
        chosen: list[int] = [
            # We need to 'dereference' the returned index of
            # pareto back to the population index (via row).
            row[np_pareto_front_index(fitness_array[row])[0]]
            for row in contestants
        ]

        return [
            # Ensure 'k' is passed as integer.
            population[int(k)] for k in chosen
        ]
    # _end_def_

# _end_class_
