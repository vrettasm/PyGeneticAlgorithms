from numpy import nan as np_nan
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator


class PositionBasedCrossover(CrossoverOperator):
    """
    Description:

        Position based crossover (POS) creates two children chromosomes, by ensuring that the
        original genome (from both parents) isn't repeated, thus creating invalid offsprings.

        It is used predominantly in combinatorial problems.
    """

    def __init__(self, crossover_probability: float = 0.9):
        """
        Construct a 'PositionBasedCrossover' object with a given probability value.

        :param crossover_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(crossover_probability)
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome):
        """
        Perform the crossover operation on the two input parent chromosomes.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """

        # If the crossover probability is higher than
        # a uniformly random value, make the changes.
        if self.probability > self.rng.random():

            # Get the size of the chromosomes.
            M = len(parent1)

            # Select randomly 'K' number of crossover points.
            K = self.rng.integers(low=1, high=M-1, size=1).item()

            # Select randomly the 'K' crossover points.
            C = sorted(self.rng.choice(range(M), size=K, replace=False, shuffle=False))

            # Initialize the genome of the two new chromosomes to None.
            genome_1 = M * [None]
            genome_2 = M * [None]

            # Copy the genes of the parents at
            # the preselected gene C positions.
            for i in C:
                genome_1[i] = parent2[i]
                genome_2[i] = parent1[i]
            # _end_for_

            # Fill the rest of the positions in both offsprings.
            for gene1, gene2 in zip(parent1.genome, parent2.genome):

                # Check if 'gene1' exists in 1st offspring.
                if gene1 not in genome_1:
                    # Find the first 'None' entry.
                    j = genome_1.index(None)

                    # Assign the current gene value.
                    genome_1[j] = gene1
                # _end_if_

                # Check if 'gene2' exists in 2nd offspring.
                if gene2 not in genome_2:
                    # Find the first 'None' entry.
                    k = genome_2.index(None)

                    # Assign the current gene value.
                    genome_2[k] = gene2
                # _end_if_

            # _end_for_

            # After the crossover neither offspring has accurate fitness.
            child1 = Chromosome(genome_1, _fitness=np_nan)
            child2 = Chromosome(genome_2, _fitness=np_nan)

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
