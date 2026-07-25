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

        :param population: a list of chromosomes to select the parents from.

        :return: the selected parents population (as list of chromosomes).
        """

        # Extract the population fitness to numpy array.
        fitness: NDArray = np.asarray([
            p.fitness for p in population
        ], dtype=float)

        # Size of the population.
        n_size = len(population)

        # Append a new column with the indexes.
        x_fit = np.concatenate((fitness,
                                np.arange(n_size, dtype=float)[:, None]),
                               axis=1)

        # Estimate the pareto front and get the indexes only.
        pareto_front: NDArray = np_pareto_front_index(x_fit)

        print(pareto_front.size)

        # Check if we need more parents.
        r = n_size - pareto_front.size
        if r > 0:
            # Make a boolean mask of size N.
            mask = np.ones(n_size, dtype=bool)

            # Set the values to False.
            mask[pareto_front] = False

            # This will hold the available.
            remaining = np.nonzero(mask)[0]

            # Sample with replacement from the
            # remaining to fill the population.
            extra = np.random.choice(remaining, size=r, replace=True)

            # Append the extras to the population.
            chosen = np.concatenate((pareto_front, extra), axis=0)
        else:
            # This is highly unlikely but it could happen.
            chosen = pareto_front
        # _end_if_

        # Return the new parents.
        return [
            population[int(k)] for k in chosen
        ]
    # _end_def_

# _end_class_
