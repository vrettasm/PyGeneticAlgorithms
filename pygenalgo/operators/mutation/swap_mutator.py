from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.utilities import two_indices_fast
from pygenalgo.operators.mutation.mutate_operator import MutationOperator


class SwapMutator(MutationOperator):
    """
    Description:

        Swap mutator mutates the chromosome by swapping the gene
        values between two randomly selected gene positions.
    """

    def __init__(self, mutate_probability: float = 0.1) -> None:
        """
        Construct a 'SwapMutator' object with a given probability value.

        :param mutate_probability: (float).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(mutation_probability=mutate_probability)
    # _end_def_

    def mutate(self, individual: Chromosome) -> None:
        """
        Perform the mutation operation by swapping
        the genes at two random positions.

        :param individual: (Chromosome).

        :return: None.
        """
        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.is_operator_applicable():

            # Select two random (distinct) values.
            i, j = two_indices_fast(self.rng, len(individual))

            # Swap in place between the two positions.
            individual[i], individual[j] = individual[j], individual[i]

            # Set the fitness to NaN.
            individual.invalidate_fitness()

            # Increase the mutator counter.
            self.inc_counter()
    # _end_def_

# _end_class_
