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

        # If the crossover probability is higher than a uniformly
        # random value and the parents aren't identical apply the
        # changes.
        if (parent1 is not parent2) and (parent1 != parent2) and \
                self.is_operator_applicable():

            # Get the size of the chromosomes.
            number_of_genes = len(parent1)

            # Select randomly a number of crossover points.
            number_of_points = self.rng.integers(1, high=number_of_genes-1)

            # Select randomly the crossover points.
            cross_points = sorted(self.rng.choice(number_of_genes,
                                                  size=number_of_points,
                                                  replace=False, shuffle=False))

            # Initialize the genome of the new chromosomes to 'None'.
            genome_1: list = number_of_genes * [None]
            genome_2: list = number_of_genes * [None]

            # Copy the genes of the parents at
            # the preselected gene cross points.
            for i in cross_points:
                genome_1[i] = parent2.genome[i].clone()
                genome_2[i] = parent1.genome[i].clone()
            # _end_for_

            # Fill the rest of the positions in both offsprings.
            for gene1, gene2 in zip(parent1.genome, parent2.genome):

                # Check if 'gene1' exists in 1st offspring.
                if gene1 not in genome_1:
                    # Find the first 'None' entry.
                    j = genome_1.index(None)

                    # Assign the current gene value.
                    genome_1[j] = gene1.clone()
                # _end_if_

                # Check if 'gene2' exists in 2nd offspring.
                if gene2 not in genome_2:
                    # Find the first 'None' entry.
                    k = genome_2.index(None)

                    # Assign the current gene value.
                    genome_2[k] = gene2.clone()
                # _end_if_

            # _end_for_

            # Create the two NEW offsprings.
            child1 = Chromosome(genome_1)
            child2 = Chromosome(genome_2)

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
