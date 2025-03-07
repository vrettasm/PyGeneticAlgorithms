from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator


class PartiallyMappedCrossover(CrossoverOperator):
    """
    Description:

        Partially Mapped Crossover (PMX) creates two children chromosomes, by ensuring that the
        original genome (from both parents) isn't repeated, thus creating invalid offsprings.

        It is used predominantly in combinatorial problems.
    """

    def __init__(self, crossover_probability: float = 0.9):
        """
        Construct a 'PartiallyMappedCrossover' object with a given probability value.

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
        if self.is_operator_applicable():

            # Get the size of the chromosomes.
            number_of_genes = len(parent1)

            # Initialize the genome of the two new chromosomes to None.
            genome_1 = number_of_genes * [None]
            genome_2 = number_of_genes * [None]

            # Select randomly the two crossover points.
            i, j = self.rng.choice(number_of_genes, size=2,
                                   replace=False, shuffle=False)

            # Swap indices (if necessary).
            if i > j:
                i, j = j, i
            # _end_if_

            # Make a set of indices for the middle segment.
            id_segment = set(range(i, j))

            # Copy the relevant part of the segment.
            for k in id_segment:
                genome_1[k] = parent1.genome[k].clone()
                genome_2[k] = parent2.genome[k].clone()
            # _end_for_

            # Create auxiliary Sets for faster membership check.
            segment_of_genome_1 = set(genome_1[i:j])
            segment_of_genome_2 = set(genome_2[i:j])

            # Start building the 1st offspring.
            for x, gene_x in enumerate(parent2.genome[i:j], start=i):

                # Check if the 'gene_x' exists in child1 genome.
                if gene_x in segment_of_genome_1:
                    continue
                # _end_if_

                # Initialize the local search variables.
                idx, found = x, False

                # Repeat until you find the right position.
                while not found:

                    # Look for the position of gene[idx] in parent2.
                    x_pos = parent2.genome.index(genome_1[idx])

                    # If the position is inside the segment update
                    # the index and repeat the process.
                    if x_pos in id_segment:
                        idx = x_pos
                    else:

                        # Copy the gene.
                        genome_1[x_pos] = gene_x.clone()

                        # Break the loop.
                        found = True
                    # _end_if_

                # _end_while_

            # _end_for_

            # Start building the 2nd offspring.
            for y, gene_y in enumerate(parent1.genome[i:j], start=i):

                # Check if the 'gene_y' exists in child2 genome.
                if gene_y in segment_of_genome_2:
                    continue
                # _end_if_

                # Initialize the local search variables.
                idy, found = y, False

                # Repeat until you find the right position.
                while not found:

                    # Look for the position of gene[idx] in parent1.
                    y_pos = parent1.genome.index(genome_2[idy])

                    # If the position is inside the segment update
                    # the index and repeat the process.
                    if y_pos in id_segment:
                        idy = y_pos
                    else:

                        # Copy the gene.
                        genome_2[y_pos] = gene_y.clone()

                        # Break the loop.
                        found = True
                    # _end_if_

                # _end_while_

            # _end_for_

            # Final step to fill child1/2 genomes.
            for k, (gene_a, gene_b) in enumerate(zip(parent1.genome,
                                                     parent2.genome)):
                # Check if the gene exists.
                if gene_a not in genome_2:
                    genome_2[k] = gene_a.clone()
                # _end_if_

                # Check if the gene exists.
                if gene_b not in genome_1:
                    genome_1[k] = gene_b.clone()
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
