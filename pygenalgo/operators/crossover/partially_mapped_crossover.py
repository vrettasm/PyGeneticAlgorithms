import numpy as np
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

        # Initially each child points to a clone of a single parent.
        child1 = parent1.clone()
        child2 = parent2.clone()

        # If the crossover probability is higher than
        # a uniformly random value, make the changes.
        if self.probability > self.rng.random():

            # Get the size of the chromosomes.
            M = len(parent1)

            # Select randomly the two crossover points.
            i, j = sorted(self.rng.choice(range(M), size=2, replace=False, shuffle=False))

            # Make a set of indices for the middle segment.
            segment = set(range(i, j))

            # Clean up the entries (on both children) outside the segment.
            for c in range(M):

                # Skip the segment positions.
                if c in segment:
                    continue
                # _end_if_

                child1[c] = None
                child2[c] = None
            # _end_for_

            # Start building the 1st offspring.
            for k, gene_k in enumerate(parent2.genome[i:j], start=i):

                # Check if the k-th gene exist in child1.
                if gene_k in child1.genome[i:j]:
                    continue
                # _end_if_

                # Initialize the local search variables.
                idx, found = k, False

                # Repeat until you find the right position.
                while not found:

                    # Look for the position of gene[idx] in parent2.
                    x_pos = parent2.genome.index(child1.genome[idx])

                    # If the position is inside the segment update
                    # the index and repeat the process.
                    if x_pos in segment:
                        idx = x_pos
                    else:

                        # Copy the gene.
                        child1.genome[x_pos] = gene_k

                        # Break the loop.
                        found = True
                    # _end_if_

                # _end_while_

            # _end_for_

            # Start building the 2nd offspring.
            for f, gene_f in enumerate(parent1.genome[i:j], start=i):

                # Check if the f-th gene exist in child2.
                if gene_f in child2.genome[i:j]:
                    continue
                # _end_if_

                # Initialize the local search variables.
                idx, found = f, False

                # Repeat until you find the right position.
                while not found:

                    # Look for the position of gene[idx] in parent1.
                    y_pos = parent1.genome.index(child2.genome[idx])

                    # If the position is inside the segment update
                    # the index and repeat the process.
                    if y_pos in segment:
                        idx = y_pos
                    else:

                        # Copy the gene.
                        child2.genome[y_pos] = gene_f

                        # Break the loop.
                        found = True
                    # _end_if_

                # _end_while_

            # _end_for_

            # Final step to fill child1/2 genomes.
            for x, (gene_a, gene_b) in enumerate(zip(parent1.genome,
                                                     parent2.genome)):
                # Check if the gene exists.
                if gene_a not in child2.genome:
                    child2.genome[x] = gene_a
                # _end_if_

                # Check if the gene exists.
                if gene_b not in child1.genome:
                    child1.genome[x] = gene_b
                # _end_if_

            # _end_for_

            # After the crossover neither offspring has accurate fitness.
            child1.fitness = np.nan
            child2.fitness = np.nan

            # Increase the crossover counter.
            self.inc_counter()
        # _end_if_

        # Return the two offsprings.
        return child1, child2
    # _end_def_

# _end_class_