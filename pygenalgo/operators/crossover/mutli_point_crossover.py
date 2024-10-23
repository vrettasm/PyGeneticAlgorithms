import numpy as np

from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator


class MultiPointCrossover(CrossoverOperator):
    """
    Description:

        Multipoint crossover creates two children chromosomes (offsprings),
        by taking two parent chromosomes and cutting them at randomly chosen,
        sites (loci).

        It produces faster mixing, compared with single-point crossover.
    """

    def __init__(self, crossover_probability: float = 0.9, num_loci: int = 2):
        """
        Construct a 'MultiPointCrossover' object with a given
        probability value.

        :param crossover_probability: (float).

        :param num_loci: (int).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(crossover_probability)

        # Make sure number of loci are at least 2.
        self._items = max(int(num_loci), 2)
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome):
        """
        Perform the crossover operation on the two input parent
        chromosomes, using multiple cutting points (num_loci).

        NOTE: the number of loci is held in the '_items' variable.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """

        # Get the number of genes from the first parent (chromosome).
        # N.B.: It is assumed that both parents have the same size.
        num_genes = len(parent1)

        # Ensure the number of requested cutting points
        # do not exceed the length of the chromosomes.
        if self._items >= num_genes:
            raise ValueError(f"{self.__class__.__name__}:"
                             " Number of requested crossover points"
                             " exceeds the length of the chromosome.")
        # _end_def_

        # Initially each child points to a clone of a single parent.
        child1 = parent1.clone()
        child2 = parent2.clone()

        # If the crossover probability is higher than
        # a uniformly random value, make the changes.
        if self.probability > self.rng.random():

            # Select randomly the crossover points and sort them.
            loci = sorted(self.rng.choice(num_genes, size=self._items,
                                          replace=False, shuffle=False))

            # Initialize an array with True values
            # with the same size as the chromosome.
            flags = np.array(num_genes * [True], dtype=bool)

            # Use the loci positions to invert the
            # bool values between two loci positions.
            for j in loci:
                flags[:j] = np.logical_not(flags[:j])
            # _end_for_

            # Go through all the children's genes and
            # swap them when the flags' are true.
            for i, flags_i in enumerate(flags):

                if flags_i:
                    # Swap in place between the two positions.
                    child1[i], child2[i] = child2[i], child1[i]
                # _end_if_

            # _end_for_

            # After the crossover neither offspring has accurate fitness.
            if np.any(flags):
                child1.fitness = np.nan
                child2.fitness = np.nan
            # _end_if_

            # Increase the crossover counter.
            self.inc_counter()
        # _end_if_

        # Return the two offsprings.
        return child1, child2
    # _end_def_

# _end_class_
