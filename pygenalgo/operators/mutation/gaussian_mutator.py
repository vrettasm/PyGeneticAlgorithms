import numpy as np
from numpy.typing import NDArray, ArrayLike

from pygenalgo.utils.utilities import clamp
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class GaussianMutator(MutationOperator):
    """
    Description:

        Gaussian mutator, mutates the chromosome by selecting randomly a position
        and perturbing it with a Gaussian random value to the current gene value.
    """

    def __init__(self, mutate_probability: float = 0.1,
                 sigma: ArrayLike | float = 1.0,
                 lower_lim: ArrayLike = None,
                 upper_lim: ArrayLike = None) -> None:
        """
        Construct a 'GaussianMutator' object with a given probability value.

        :param mutate_probability: (float).

        :param sigma: (float) standard deviation of the Gaussian N(0, sigma).

        :param lower_val: (float) lower limit value for the gene mutation.

        :param upper_val: (float) upper limit value for the gene mutation.
        """
        # Call the super constructor with the provided initial value.
        super().__init__(mutation_probability=mutate_probability)

        # Ensure sigma parameter is float.
        sigma: NDArray = np.asarray(sigma, dtype=float)

        # Ensure standard deviation is positive.
        if np.any(sigma <= 0.0):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Standard deviation must be positive.")
        # _end_if_

        # Check if the lower and upper bounds are set.
        if (lower_lim is None) or (upper_lim is None):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Lower or Upper limits are missing.")
        # _end_if_

        # Make sure the limits are numpy arrays.
        lower_lim: NDArray = np.asarray(lower_lim, dtype=float)
        upper_lim: NDArray = np.asarray(upper_lim, dtype=float)

        # Check if there is a size mismatch.
        if lower_lim.size != upper_lim.size:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Lower and Upper limits sizes do not match.")

        # Check if the boundaries are set correctly.
        if np.any(upper_lim <= lower_lim):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Lower and Upper limits are set incorrectly.")

        # Assign variables to the _items placeholder.
        self._items: tuple[float, ...] = (
            sigma, lower_lim, upper_lim
        )
    # _end_def_

    def mutate(self, individual: Chromosome) -> None:
        """
        Perform the mutation operation by randomly adding the
        Gaussian value to a randomly selected gene position.

        :param individual: (Chromosome).

        :return: None.
        """
        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.is_operator_applicable():

            # Get the size of the chromosome.
            n_genes: int = len(individual)

            # Extract the variables from the placeholder.
            sigma, xl, xu = self._items

            # If sigma is scalar use the
            # value for all gene positions.
            if np.isscalar(sigma):
                sigma *= np.ones_like(xl)

            # Select a random position in the genome.
            idx = self.rng.integers(n_genes, dtype=int)

            # Get the old value of the Gene.
            old_value: float = individual[idx].value

            # Calculate the new Gene value by sampling from N(mu,sigma).
            new_value: float = self.rng.normal(loc=old_value, scale=sigma[idx])

            # Ensure it stays within limits.
            individual[idx].value = clamp(new_value, xl[idx], xu[idx])

            # Set the fitness to None.
            individual.invalidate_fitness()

            # Increase the mutator counter.
            self.inc_counter()
    # _end_def_

# _end_class_
