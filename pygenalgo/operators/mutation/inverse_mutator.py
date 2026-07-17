from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.utilities import two_indices_fast
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class InverseMutator(MutationOperator):
    """
    Description:

        Inverse mutator mutates the chromosome by inverting the order of
        the gene values between two randomly selected gene end-positions.
    """

    def __init__(self, mutate_probability: float = 0.1) -> None:
        """
        Construct a 'InverseMutator' object with a given probability value.

        :param mutate_probability: (float).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(mutation_probability=mutate_probability)
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

            # Select two random (distinct) values.
            i, j = two_indices_fast(self.rng, len(individual))

            # Swap indices (if necessary).
            if i > j:
                i, j = j, i
            # _end_if_

            # Make a slice list of the genes
            # we want to inverse their order: i -> j.
            sliced_chromosome: list[Chromosome] = individual.genome[i:j]

            # Invert the copied slice in place.
            sliced_chromosome.reverse()

            # Put back the inverse items.
            individual.genome[i:j] = sliced_chromosome

            # Set the fitness to NaN.
            individual.invalidate_fitness()

            # Increase the mutator counter.
            self.inc_counter()
    # _end_def_

# _end_class_
