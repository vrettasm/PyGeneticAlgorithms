from mutate_operator import MutationOperator
from code.src.genome.chromosome import Chromosome

class ShuffleMutator(MutationOperator):
    """
    Description:

        Shuffle mutator attempts to mutate the chromosome by shuffling
        the gene values between two randomly selected gene positions.
    """

    def __init__(self, mutate_probability: float = 0.1):
        """
        Construct a 'ShuffleMutator' object with a given
        probability value.

        :param mutate_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(mutate_probability)

    # _end_def_

    def mutate(self, parent: Chromosome):
        """
        Perform the mutation operation by shuffling the genes
        between at two random positions.

        :param parent: (Chromosome).

        :return: child (as Chromosomes).
        """

        # Initially make a copy of the parent.
        child = parent.make_deepcopy()

        # If the mutation probability is higher than
        # a uniformly random value, make the changes.
        if self.probability >= self.rng.random():

            # Select randomly the two mutation end-points.
            loci = sorted(self.rng.choice(range(0, len(child)), size=2,
                                          replace=False, shuffle=False))
            # Extract the indexes.
            i, j = loci

            # Make a slice list of the genes
            # we want to shuffle: i -> j.
            shuffled_chromosome = child[i:j]

            # Shuffle the copied slice in place.
            self.rng.shuffle(shuffled_chromosome)

            # Put back the shuffled items.
            child[i:j] = shuffled_chromosome

            # Increase the application counter.
            self.inc_counter()
        # _end_if_

        # Return the offspring.
        return child
    # _end_def_

# _end_class_
