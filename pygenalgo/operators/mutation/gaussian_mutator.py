from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class GaussianMutator(MutationOperator):
    """
    Description:

        Gaussian mutator, mutates the chromosome by selecting randomly a position
        and add a Gaussian random value to the current gene value.
    """

    def __init__(self, mutate_probability: float = 0.1, sigma: float = 1.0):
        """
        Construct a 'GaussianMutator' object with a given probability value.

        :param mutate_probability: (float).

        :param sigma: (float) standard deviation of the Gaussian.
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(mutate_probability)

        # Standard deviation (scale) of the Gaussian sample.
        self._items = max(min(float(sigma), 1.0), 0.0)
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

            # Select randomly the mutation point and update its value.
            individual[self.rng.integers(n_genes)].gaussian(sigma=self._items)

            # Set the fitness to NaN.
            individual.invalidate_fitness()

            # Increase the mutator counter.
            self.inc_counter()
        # _end_if_

    # _end_def_

# _end_class_
