""" Partially mapped crossover (PMX) operator module. """
# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.utilities import two_indices_fast
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offsprings)


class PartiallyMappedCrossover(CrossoverOperator):
    """
    Description:

        Partially Mapped Crossover (PMX) creates two children chromosomes,
        by ensuring that the original genome, from both parents, isn't repeated,
        thus creating invalid offsprings.

        It is used predominantly in combinatorial problems.
    """

    def __init__(self, crossover_probability: float = 0.9) -> None:
        """
        Construct a 'PartiallyMappedCrossover' object with
        a given probability value.

        :param crossover_probability: (float).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(crossover_probability=crossover_probability)
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Offsprings:
        """
        Perform the crossover operation on the two input parent chromosomes.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """
        # If the crossover probability is higher than a uniformly
        # random value and the parents aren't identical apply the
        # changes.
        if self.is_operator_applicable() and (parent1 != parent2):

            # Get the size of the chromosomes.
            number_of_genes: int = len(parent1)

            # Initialize the genome lists for the new
            # chromosomes to 'None'.
            child_1: list = number_of_genes * [None]
            child_2: list = number_of_genes * [None]

            # Select two random (distinct) crossover points.
            i, j = two_indices_fast(self.rng, number_of_genes, in_order=True)

            # Make a set of indices for the middle segment.
            id_segment: set[int] = set(range(i, j))

            # Copy the relevant part of the segment.
            for k in id_segment:
                child_1[k] = parent1.genome[k].clone()
                child_2[k] = parent2.genome[k].clone()
            # _end_for_

            # Create auxiliary Sets for faster membership check.
            segment_of_genome_1 = set(child_1[i:j])
            segment_of_genome_2 = set(child_2[i:j])

            # Start building the offsprings.
            for n, (gene_x, gene_y) in enumerate(zip(parent2.genome[i:j],
                                                     parent1.genome[i:j]), start=i):

                # Check if the 'gene_x' exists in child1.
                if gene_x not in segment_of_genome_1:
                    # Initialize the local search variables.
                    idx, found = n, False

                    # Repeat until you find the right position.
                    while not found:
                        # Look for the position of gene[idx] in parent2.
                        x_pos = parent2.genome.index(child_1[idx])

                        # If the position is inside the segment update
                        # the index and repeat the process.
                        if x_pos in id_segment:
                            idx = x_pos
                        else:
                            # Copy the gene.
                            child_1[x_pos] = gene_x.clone()

                            # Break the loop.
                            found = True
                # _end_if_

                # Check if the 'gene_y' exists in child2.
                if gene_y not in segment_of_genome_2:
                    # Initialize the local search variables.
                    idy, found = n, False

                    # Repeat until you find the right position.
                    while not found:
                        # Look for the position of gene[idx] in parent1.
                        y_pos = parent1.genome.index(child_2[idy])

                        # If the position is inside the segment update
                        # the index and repeat the process.
                        if y_pos in id_segment:
                            idy = y_pos
                        else:
                            # Copy the gene.
                            child_2[y_pos] = gene_y.clone()

                            # Break the loop.
                            found = True
                # _end_if_
            # _end_for_

            # Final step to fill child1/2 genomes.
            for k, (gene_a, gene_b) in enumerate(zip(parent1.genome,
                                                     parent2.genome)):
                # Check if the gene exists.
                if gene_a not in child_2:
                    child_2[k] = gene_a.clone()

                # Check if the gene exists.
                if gene_b not in child_1:
                    child_1[k] = gene_b.clone()
            # _end_for_

            # Increase the crossover counter.
            self.inc_counter()

            # Return two new offsprings.
            return Chromosome(child_1), Chromosome(child_2)
        # _end_if_

        # Return two cloned offsprings.
        return parent1.clone(), parent2.clone()
    # _end_def_

# _end_class_
