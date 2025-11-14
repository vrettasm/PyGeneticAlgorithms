import numpy as np
from collections import deque
from operator import attrgetter
from pygenalgo.utils.utilities import np_cdist
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.genetic_operator import increase_counter
from pygenalgo.operators.selection.select_operator import SelectionOperator


class NeighborhoodSelector(SelectionOperator):
    """
    Description:

        Neighborhood Selector implements an object that performs selection by choosing
        an individual from a set of neighboring individuals. The neighbors are selected
        based on their proximity from one another using their Euclidean distance.
    """

    def __init__(self, select_probability: float = 1.0, n_nearest: int = 5):
        """
        Construct a 'NeighborhoodSelector' object with a given probability value.

        :param select_probability: (float) in [0, 1].

        :param n_nearest: the number of the nearest neighbors to consider (int).
        """

        # Call the super constructor with the provided probability value.
        super().__init__(select_probability)

        # Number of neighbors of the tournament should be more than 2.
        self._items = max(2, int(n_nearest))
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

        # Get the population size.
        pop_size = len(population)

        # Extract the population positions as an array.
        x_pos = np.array([p.values() for p in population])

        # Compute the pairwise distances.
        pairwise_dists = np_cdist(x_pos, scaled=True)

        # Sort the distances and get their indexes.
        x_sorted = np.argsort(pairwise_dists, axis=1)

        # Make a view of the first '_items'. This provides
        # a notion of a 'neighborhood' for each chromosome.
        neighborhood = x_sorted[:, :self._items]

        # Calculate the selection probabilities of each member in the
        # neighborhood, using their ranking position (low --> high).
        # This should remain constant during the evolution hence the
        # method is cached for improved performance.
        selection_probs = self.linear_rank_probabilities(self._items)

        # Create a new deque with fixed size.
        new_parents = deque(maxlen=pop_size)

        # Select the new parents.
        for p_ids in neighborhood:
            # Sort the neighbors in ascending order using their fitness.
            sorted_neighbors = sorted((population[i] for i in p_ids),
                                      key=attrgetter("fitness"))

            # Select a new parent from the sorted neighbors.
            n = self.rng.choice(self._items, p=selection_probs)

            # Add it to the new deque.
            new_parents.append(sorted_neighbors[n])
        # _end_for_

        # Return the new parents.
        return list(new_parents)
    # _end_def_

# _end_class_
