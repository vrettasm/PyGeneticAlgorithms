""" Pareto front selector module. """
# Third party imports.
import numpy as np
from numpy.typing import NDArray

# Custom code imports.
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
        slots it uses Adaptive Stochastic Sampling to fill up the slots.
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

        # Size of the pareto array.
        n_pareto: int = pareto_idx.size

        # Remaining size (non-pareto).
        rem_size: int = n_size - n_pareto

        # Edge case no.1:
        if rem_size == 0:

            # Return the same.
            return population
        # _end_if_

        # Local copy of random choice.
        choose_randomly = self.rng.choice

        # Edge case no.2:
        if rem_size == 1:

            # Select one pareto index at random.
            extra_idx: int = choose_randomly(pareto_idx)

            # Set up the chosen array.
            chosen: NDArray = np.append(pareto_idx, extra_idx)
        else:

            # Extract the remaining (non-pareto) indices.
            remaining_idx: NDArray = np.setdiff1d(np.arange(n_size),
                                                  pareto_idx,
                                                  assume_unique=True)

            # Compute dynamically the pareto probability.
            pareto_probability: float = n_pareto / n_size

            # Generate uniform random numbers and convert them to bool.
            pareto_flag: NDArray = self.rng.random(size=rem_size) > pareto_probability

            # Fill the extras list.
            extras: list[int] = [
                choose_randomly(remaining_idx) if flag else choose_randomly(pareto_idx)
                for flag in pareto_flag
            ]

            # Combined both results in one array.
            chosen: NDArray = np.concatenate((pareto_idx, extras))

        return [
            # Ensure 'k' is passed as integer.
            population[int(k)] for k in chosen
        ]
    # _end_def_

# _end_class_
