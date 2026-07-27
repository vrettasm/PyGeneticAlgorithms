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
        pareto_indices: NDArray = np_pareto_front_index(fitness_array)

        # Remaining size.
        r_size: int = n_size - pareto_indices.size

        # Check if the remaining size is positive.
        if r_size > 0:
            # Fast extraction of the remaining indices using delete.
            remaining: NDArray = np.delete(np.arange(n_size), pareto_indices)

            # Chose 'r_size' values directly from the remaining indices.
            extra: NDArray = self.rng.choice(remaining, size=r_size, replace=True)

            # Combined fast concatenation and array permutation.
            chosen: NDArray = self.rng.permutation(np.concatenate((pareto_indices, extra)))
        else:
            # Getting here is highly unlikely.
            chosen: NDArray = pareto_indices
        # _end_if_

        return [population[k] for k in chosen]
    # _end_def_

# _end_class_
