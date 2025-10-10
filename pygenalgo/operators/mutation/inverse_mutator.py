from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class InverseMutator(MutationOperator):
    """
    Description:

        Inverse mutator mutates the chromosome by inverting the order of
        the gene values between two randomly selected gene end-positions.
    """

    def __init__(self, mutate_probability: float = 0.1):
        """
        Construct a 'InverseMutator' object with a given probability value.

        :param mutate_probability: (float).
        """

        # Call the super constructor with the provided probability value.
        super().__init__(mutate_probability)
    # _end_def_

    def mutate(self, individual: Chromosome) -> None:
        """
        Perform the mutation operation by inverting the genes
        between at two random positions.

        :param individual: (Chromosome).

        :return: None.
        """

        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.is_operator_applicable():

            # Get the size of the chromosome.
            n_genes = len(individual)

            # Select randomly two mutation end-points.
            i, j = self.rng.choice(n_genes, size=2, replace=False)

            # Swap indices (if necessary).
            if i > j:
                i, j = j, i
            # _end_if_

            # Make a slice list of the genes
            # we want to inverse their order: i -> j.
            sliced_chromosome = individual.genome[i:j]

            # Invert the copied slice in place.
            sliced_chromosome.reverse()

            # Put back the inverse items.
            individual.genome[i:j] = sliced_chromosome

            # Set the fitness to NaN.
            individual.invalidate_fitness()

            # Increase the mutator counter.
            self.inc_counter()
        # _end_if_

    # _end_def_

# _end_class_
