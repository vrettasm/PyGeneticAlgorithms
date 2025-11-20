from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class RandomMutator(MutationOperator):
    """
    Description:

        Random mutator, mutates the chromosome by selecting randomly a position and replace
        the Gene with a new one that has been generated randomly (uniform probability).
    """

    def __init__(self, mutate_probability: float = 0.1):
        """
        Construct a 'RandomMutator' object with a given
        probability value.

        :param mutate_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(mutate_probability)
    # _end_def_

    def mutate(self, individual: Chromosome) -> None:
        """
        Perform the mutation operation by randomly replacing
        a gene with a new one that has been generated randomly.

        :param individual: (Chromosome).

        :return: None.
        """

        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.is_operator_applicable():

            # Get the size of the chromosome.
            n_genes = len(individual)

            # Select randomly the mutation point and
            # replace the old gene with a new one.
            individual[self.rng.integers(n_genes)].random()

            # Set the fitness to NaN.
            individual.invalidate_fitness()

            # Increase the mutator counter.
            self.inc_counter()
    # _end_def_

# _end_class_
