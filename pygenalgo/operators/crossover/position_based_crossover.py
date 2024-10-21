import numpy as np
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

        # Initially each child points to a clone of a single parent.
        child1 = parent2.clone()
        child2 = parent1.clone()

        # If the crossover probability is higher than
        # a uniformly random value, make the changes.
        if self.probability > self.rng.random():

            # Get the size of the chromosomes.
            M = len(parent1)

            # Select randomly 'K' number of crossover points.
            K = self.rng.integers(low=1, high=M-1, size=1).item()

            # Select randomly the 'K' crossover points.
            C = sorted(self.rng.choice(range(M), size=K, replace=False, shuffle=False))

            # Clean up the entries (on both children)
            # outside the selected positions.
            for i in range(M):

                # Skip the preselected positions.
                if i in C:
                    continue
                # _end_if_

                child1[i] = None
                child2[i] = None
            # _end_for_

            # Fill the rest of the positions in both offsprings.
            for gene1, gene2 in zip(parent1.genome, parent2.genome):

                # Check if 'gene1' exists in 1st offspring.
                if gene1 not in child1:

                    for n in range(M):
                        if child1[n] is None:
                            child1[n] = gene1
                            break
                        # _end_if_
                    # _end_for_

                # _end_if_

                # Check if 'gene2' exists in 2nd offspring.
                if gene2 not in child2:

                    for n in range(M):
                        if child2[n] is None:
                            child2[n] = gene2
                            break
                        # _end_if_
                    # _end_for_

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
