from pygenalgo.utils.utilities import clamp
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class GaussianMutator(MutationOperator):
    """
    Description:

        Gaussian mutator, mutates the chromosome by selecting randomly a position
        and perturbing it with a Gaussian random value to the current gene value.
    """

    def __init__(self, mutate_probability: float = 0.1, sigma: float = 1.0,
                 lower_val: float = None, upper_val: float = None) -> None:
        """
        Construct a 'GaussianMutator' object with a given probability value.

        :param mutate_probability: (float).

        :param sigma: (float) standard deviation of the Gaussian N(0, sigma).

        :param lower_val: (float) lower limit value for the gene mutation.

        :param upper_val: (float) upper limit value for the gene mutation.
        """
        # Call the super constructor with the provided
        # probability value.
        super().__init__(mutate_probability)

        # Ensure sigma parameter is float.
        sigma = float(sigma)

        # Ensure standard deviation is positive.
        if sigma <= 0.0:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Standard deviation must be positive.")
        # _end_if_

        # Ensure that both lower and upper limits are provided.
        if lower_val is None or upper_val is None:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Lower or Upper limits are missing.")
        # _end_if_

        # Ensure lower_val parameter is float.
        lower_val = float(lower_val)

        # Ensure upper_val parameter is float.
        upper_val = float(upper_val)

        # Ensure the order is correct.
        if upper_val <= lower_val:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"The limit values are incorrect.")
        # _end_if_

        # Assign variables to the _items placeholder.
        self._items = [sigma, lower_val, upper_val]
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
            n_genes = len(individual)

            # Extract the values from the placeholder variable.
            sigma, xl, xu = self._items

            # Select a random position in the genome.
            i = self.rng.integers(n_genes)

            # Get the old value of the Gene.
            old_value = individual[i].value

            # Calculate the new Gene value by sampling from N(value, sigma),
            # ensuring it stays within limits.
            individual[i].value = clamp(self.rng.normal(loc=old_value,
                                                        scale=sigma), xl, xu)
            # Set the fitness to NaN.
            individual.invalidate_fitness()

            # Increase the mutator counter.
            self.inc_counter()
    # _end_def_

# _end_class_
