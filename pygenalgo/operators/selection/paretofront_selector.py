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

    def __init__(self, select_probability: float = 1.0) -> None:
        """
        Construct a 'ParetoFrontSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].
        """
        # Call the super constructor with the provided initial value.
        super().__init__(selection_probability=select_probability)
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

        # Extract original index positions directly.
        # Note: The fitness values have already been set for
        # maximization so here the default mode = "max" is assumed.
        pareto_idx: NDArray = np_pareto_front_index(fitness_array)

        # Remaining size.
        r_size: int = n_size - pareto_idx.size

        # Check if the remaining size is positive.
        if r_size > 0:
            # Fast extraction of the remaining indices.
            remaining_idx: NDArray = np.setdiff1d(np.arange(n_size),
                                                  pareto_idx,
                                                  assume_unique=True)
            # Compute the half (of the remaining).
            half_rem: int = r_size // 2

            # Select randomly some elements from the remaining.
            extra_1: NDArray = self.rng.choice(remaining_idx,
                                               size=half_rem,
                                               replace=True)

            # Select randomly the other half from the pareto elements.
            extra_2: NDArray = self.rng.choice(pareto_idx,
                                               size=r_size - half_rem,
                                               replace=True)
            # Combined fast concatenation.
            chosen: NDArray = np.concatenate((pareto_idx, extra_1, extra_2))
        else:
            # Getting here is highly unlikely.
            chosen: NDArray = pareto_idx
        # _end_if_

        # Ensure the items are shuffled.
        self.rng.shuffle(chosen)

        return [population[k] for k in chosen]
    # _end_def_

# _end_class_
