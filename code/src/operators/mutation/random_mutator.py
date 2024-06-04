from code.src.genome.chromosome import Chromosome
from mutate_operator import MutationOperator


class RandomMutator(MutationOperator):
    """
    Description:

        Random mutator, mutates the chromosome by selecting randomly
        a position and replace the Gene with a new one that has been
        generated randomly (uniform probability).
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

    def mutate(self, parent: Chromosome):
        """
        Perform the mutation operation by randomly replacing
        a gene with a new one that has been generated randomly.

        :param parent: (Chromosome).

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
            child[locus] = child[locus].random()

            # Increase the application counter.
            self.inc_counter()
        # _end_if_

        # Return the offspring.
        return child
    # _end_def_

# _end_class_
