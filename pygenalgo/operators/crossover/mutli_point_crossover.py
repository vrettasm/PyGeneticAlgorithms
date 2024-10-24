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

        # If the crossover probability is higher than
        # a uniformly random value, make the changes.
        if self.probability > self.rng.random():

            # Select randomly the crossover points and sort them.
            loci = sorted(self.rng.choice(num_genes, size=self._items,
                                          replace=False, shuffle=False))

            # Initialize the offspring genomes to None.
            genome_1 = num_genes * [None]
            genome_2 = num_genes * [None]

            # Initialize a set of hyperparameters.
            reset_flag, upper_lim, j = True, loci[0], 0

            # Scan the whole list of genes.
            for i in range(num_genes):

                # Once we surpass the upper limit (in loci)
                # we reset the  flag value to allow changes
                # to take place within that range.
                if i >= upper_lim:

                    # Swap the reset flag.
                    reset_flag = not reset_flag

                    # Increase the index of the loci.
                    j += 1

                    # We make sure the upper limit value does not exceed
                    # the number of genes. Also, this avoids the out of
                    # bound IndexError.
                    upper_lim = loci[j] if j < len(loci) else num_genes
                # _end_if_

                # Check the flag value.
                if reset_flag:
                    genome_1[i] = parent1[i]
                    genome_2[i] = parent2[i]
                else:
                    genome_1[i] = parent2[i]
                    genome_2[i] = parent1[i]
                # _end_if_

            # _end_for_

            # Create the two NEW offsprings.
            child1 = Chromosome(genome_1, _fitness=np.nan)
            child2 = Chromosome(genome_2, _fitness=np.nan)

            # Increase the crossover counter.
            self.inc_counter()
        else:
            # Each child points to a clone of a single parent.
            child1 = parent1.clone()
            child2 = parent2.clone()
        # _end_if_

        # Return the two offsprings.
        return child1, child2
    # _end_def_

# _end_class_
