import numpy as np
from numpy.typing import NDArray

from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.utilities import np_pareto_front_index
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class ParetoFrontSelector(SelectionOperator):
    """
    Description:

        TBD ...

    """

    def __init__(self, select_probability: float = 1.0,
                 n_contestants: int = 2) -> None:
        """
        Construct a 'ParetoFrontSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].

        :param n_contestants: the number of participants in the tournament (int).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(selection_probability=select_probability)

        # Set the value of the contestants in the placeholder _items.
        self._items: int = max(2, int(n_contestants))
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Select the individuals, from the input population that will be passed on
        to the next genetic operations of crossover and mutation to form the new
        population of solutions.
        """
        # Properly build a 2D array from
        # the fitness tuples (objectives).
        fitness_array: NDArray = np.array([
            p.fitness for p in population
        ], dtype=float)

        # Total size of the population.
        n_size: int = len(population)

        # Local copy of random choice.
        choose_randomly = self.rng.choice

        # Number of contestants.
        n_contestants = self._items

        # Select the contestants for the tournaments.
        contestants: NDArray = np.array([
            choose_randomly(n_size, size=n_contestants,
                            replace=False, shuffle=False)
            for _ in range(n_size)
        ], dtype=int)

        # Select the new indices via Tournament selection.
        chosen: list[int] = [
            row[np_pareto_front_index(fitness_array[row])[0]]
            for row in contestants
        ]

        return [population[k] for k in chosen]
    # _end_def_

    @increase_counter
    def select_22(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Select the individuals, from the input population that will be passed on
        to the next genetic operations of crossover and mutation to form the new
        population of solutions.
        """
        # Properly build a 2D array from
        # the fitness tuples (objectives).
        fitness_array: NDArray = np.array([
            p.fitness for p in population
        ], dtype=float)

        # Total size of the population.
        n_size: int = len(population)

        # Extract original index positions directly.
        # Note: The fitness values have already been
        # set for maximization : so here the default
        # mode = "max" is assumed.
        pareto_idx: NDArray = np_pareto_front_index(fitness_array)

        # Remaining size.
        r_size: int = n_size - pareto_idx.size

        # Check if the remaining size is positive.
        if r_size > 0:
            # Fast extraction of the remaining indices.
            remaining_idx: NDArray = np.setdiff1d(np.arange(n_size),
                                                  pareto_idx,
                                                  assume_unique=True)
            # Local copy of random choice.
            choose_randomly = self.rng.choice

            # Number of contestants.
            n_contestants = self._items

            # Check if we have enough remaining indices.
            if remaining_idx.size < n_contestants:

                # Select all the remaining indices.
                contestants: NDArray = np.tile(remaining_idx,
                                               (r_size, 1))
            else:

                # Select the contestants for the tournaments.
                contestants: NDArray = np.array([
                    choose_randomly(remaining_idx, size=n_contestants,
                                    replace=False, shuffle=False)
                    for _ in range(r_size)
                ], dtype=int)

            # Select the extras via Tournament selection.
            extras = [
                row[np_pareto_front_index(fitness_array[row])[0]]
                for row in contestants
            ]

            # Combined fast concatenation.
            chosen: NDArray = np.concatenate((pareto_idx, extras))
        else:
            # Getting here is highly unlikely.
            chosen: NDArray = pareto_idx
        # _end_if_

        # Ensure the items are shuffled.
        self.rng.shuffle(chosen)

        return [population[k] for k in chosen]
    # _end_def_

# _end_class_
