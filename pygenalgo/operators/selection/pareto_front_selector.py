import numpy as np
from numpy.typing import NDArray

from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.utilities import np_pareto_front_index
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class ParetoFrontSelector(SelectionOperator):
    """
    Description:
        This selector is used exclusively by the MultiObjectiveGA class.
        It selects first individuals that lie on the pareto front of the
        objectives space (fitness objectives) and if there are remaining
        slots it uses a Tournament selection logic to fill up the spaces.
    """

    def __init__(self, select_probability: float = 1.0,
                 n_contestants: int = 5) -> None:
        """
        Construct a 'ParetoFrontSelector' object with a given probability value.

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

        # Extract original index positions directly.
        # Note: The fitness values have already been
        # set for maximization : so here the default
        # mode = "max" is assumed.
        pareto_idx: NDArray = np_pareto_front_index(fitness_array)

        # Remaining size.
        rem_size: int = n_size - pareto_idx.size

        # Quick exit if all parents are on the Pareto front.
        if rem_size <= 1:
            # This rarely happens.
            return self.rng.shuffle(population)
        # _end_if_

        # Fast extraction of the remaining indices.
        remaining_idx: NDArray = np.setdiff1d(np.arange(n_size),
                                              pareto_idx,
                                              assume_unique=True)
        # Local copy of random choice.
        choose_randomly = self.rng.choice

        # Local number of contestants. Ensure that this
        # number is not higher than the population size.
        n_contestants: int = min(self._items, rem_size)

        # Select the contestants for the tournaments.
        contestants: NDArray = np.array([
            # Set 'replace=False' to avoid duplicates.
            choose_randomly(remaining_idx, size=n_contestants,
                            replace=False, shuffle=False)
            for _ in range(rem_size)
        ], dtype=int)

        # Preallocate the extras list.
        extras: list[int] = rem_size * [None]

        # Select the new indices iteratively.
        for i, row in enumerate(contestants):
            # Get the indexes on the Pareto front.
            pf_idx = np_pareto_front_index(fitness_array[row])

            # If more than one, choose at random.
            idx = choose_randomly(pf_idx) if pf_idx.size > 1 else pf_idx[0]

            # Dereference the position in the
            # population through "row" vector.
            extras[i] = row[idx]
        # _end_for_

        # Combined fast concatenation.
        chosen: NDArray = np.concatenate((pareto_idx, extras))

        return [
            # Ensure 'k' is passed as integer.
            population[int(k)] for k in chosen
        ]
    # _end_def_

# _end_class_
