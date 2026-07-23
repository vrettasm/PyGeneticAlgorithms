import numpy as np
from numpy.typing import NDArray

from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.utilities import np_cdist, np_pareto_front
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class ParetoFrontSelector(SelectionOperator):
    """
    Description:

        TBD ...

    """

    def __init__(self, select_probability: float = 1.0, n_nearest: int = 5) -> None:
        """
        Construct a 'ParetoFrontSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].

        :param n_nearest: the number of the nearest neighbors to consider (int).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(selection_probability=select_probability)

        # Number of neighbors should be at least 5.
        self._items: int = max(5, int(n_nearest))
    # _end_def_

    @increase_counter
    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Select the individuals, from the input population that will be passed on
        to the next genetic operations of crossover and mutation to form the new
        population of solutions.

        :param population: a list of chromosomes to select the parents from.

        :return: the selected parents population (as list of chromosomes).
        """

        # Extract the population positions to numpy array.
        x_pos: NDArray = np.array([
            p.values() for p in population
        ], dtype=float)

        # Extract the population fitness to numpy array.
        x_fit: NDArray = np.array([
            p.fitness for p in population
        ], dtype=float)

        # Compute the pairwise Euclidean distances.
        pairwise_dists: NDArray = np_cdist(x_pos, scaled=True)

        # Sort the distances and get their indexes.
        x_sorted: NDArray = np.argsort(pairwise_dists, axis=1)

        # Make a view of the first '_items'. This provides
        # a notion of a 'neighborhood' for each chromosome.
        neighborhood: NDArray = x_sorted[:, :self._items]

        # Create a list to hold the indexes of the
        # parents that are on the pareto front.
        pareto_index: list = [None] * neighborhood.shape[0]

        # Construct the list with the pareto indexes.
        for i, row in enumerate(neighborhood):
            # Gather the fitness vectors.
            points = x_fit[row]

            # Get the first point of the pareto front.
            pareto_index[i] = np_pareto_front(points,
                                              return_index=True)[0]
        # _end_for_

        # Return the new parents.
        return [
            population[k] for k in pareto_index
        ]
    # _end_def_

# _end_class_
