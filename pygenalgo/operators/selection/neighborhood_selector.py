from operator import attrgetter

import numpy as np
from numpy.typing import NDArray

from pygenalgo.utils.utilities import np_cdist
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class NeighborhoodSelector(SelectionOperator):
    """
    Description:

        Neighborhood Selection focuses on leveraging local relationships within
        the population to guide the selection process. Unlike global methods it
        emphasizes individuals that are closely related in the solution space.
        This operator selects candidates based on their proximity, fostering local
        exploration and potentially discovering niche solutions. Neighborhood Selection
        can lead to a more cohesive search through similar solutions, promoting the
        evolution of specialized characteristics. While it can enhance diversity within
        clusters of solutions, the method also risks missing out on globally optimal
        solutions if too localized, necessitating a balance between exploration and
        exploitation in broader landscapes.

    """

    def __init__(self, select_probability: float = 1.0, n_nearest: int = 5) -> None:
        """
        Construct a 'NeighborhoodSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].

        :param n_nearest: the number of the nearest neighbors to consider (int).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(selection_probability=select_probability)

        # Number of neighbors of the tournament should be more than 2.
        self._items: int = max(2, int(n_nearest))
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

        # Extract the population positions as numpy array.
        x_pos: NDArray = np.array([
            p.values() for p in population
        ], dtype=float)

        # Compute the pairwise Euclidean distances.
        pairwise_dists: NDArray = np_cdist(x_pos, scaled=True)

        # Sort the distances and get their indexes.
        x_sorted: NDArray = np.argsort(pairwise_dists, axis=1)

        # Make a view of the first '_items'. This provides
        # a notion of a 'neighborhood' for each chromosome.
        neighborhood: NDArray = x_sorted[:, :self._items]

        # Define the key.
        key_sort = attrgetter("fitness")

        # Return the new parents.
        return [
            max((population[k] for k in row), key=key_sort)
            for row in neighborhood
        ]
    # _end_def_

# _end_class_
