from mutate_operator import MutationOperator
from code.src.genome.chromosome import Chromosome

class SwapMutator(MutationOperator):
    """
    Description:

        Swap mutator mutates the chromosome by swapping the
        gene values between two randomly selected gene positions.
    """

    def __init__(self, mutate_probability: float = 0.1):
        """
        Construct a 'SwapMutator' object with a given probability value.

        :param mutate_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(mutate_probability)

    # _end_def_

    def mutate(self, parent: Chromosome):
        """
        Perform the mutation operation by swapping
        the genes at two random positions.

        :param parent: (Chromosome).

        :return: child (as Chromosomes).
        """

        # Initially make a copy of the parent.
        child = parent.make_deepcopy()

        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.probability >= self.rng.random():

            # Select randomly the two mutation points.
            loci = self.rng.choice(range(0, len(parent)), size=2,
                                   replace=False, shuffle=False)
            # Extract the indexes.
            i, j = loci

            # Swap in place between the two positions.
            child[i], child[j] = child[j], child[i]

            # Increase the application counter.
            self.inc_counter()
        # _end_if_

        # Return the offspring.
        return child
    # _end_def_

# _end_class_
