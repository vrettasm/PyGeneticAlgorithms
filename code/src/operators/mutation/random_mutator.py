from code.src.genome.gene import Gene
from mutate_operator import MutationOperator
from code.src.genome.chromosome import Chromosome

class RandomMutator(MutationOperator):
    """
    Description:

        Random mutator attempts to mutate the chromosome by setting
        the gene value (of a randomly chosen position) with another
        gene selected randomly from a valid GenePool.
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

    def mutate(self, parent: Chromosome, pool: list[Gene]):
        """
        Perform the mutation operation by randomly replacing
        a gene with a new one from an accepted list of genes.

        :param parent: (Chromosome).

        :param pool: (Genes).

        :return: child (as Chromosomes).
        """

        # Initially make a copy of the parent.
        child = parent.make_deepcopy()

        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.probability >= self.rng.random():

            # Select randomly the mutation point.
            locus = self.rng.integers(0, len(child))

            # Replace the old gene with a new one.
            child[locus] = self.rng.choice(pool)

            # Increase the application counter.
            self.inc_counter()
        # _end_if_

        # Return the offspring.
        return child
    # _end_def_

# _end_class_
