from numpy import nan as np_nan
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator


class UniformCrossover(CrossoverOperator):
    """
    Description:

        Uniform crossover creates two children chromosomes (offsprings),
        by taking two parent chromosomes and swap their genes in every
        other location.

        It produces fast mixing, compared with single-point crossover.
    """

    def __init__(self, crossover_probability: float = 0.9):
        """
        Construct a 'UniformCrossover' object with a given
        probability value.

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

            # Get the length of the chromosome.
            M = len(parent1)

            # Generate 'len(parent1)' random numbers in one call.
            # It is assumed that both parents have the same size.
            swap_probs = self.rng.random(size=M)

            # Initialize the offspring genomes to None.
            genome_1 = M * [None]
            genome_2 = M * [None]

            # Go through all the children's genome.
            for i, (prob_i, gene_a, gene_b) in enumerate(zip(swap_probs,
                                                             parent1.genome,
                                                             parent2.genome)):
                # Swap genes with 50% probability.
                if prob_i > 0.5:
                    genome_1[i] = gene_b
                    genome_2[i] = gene_a
                else:
                    genome_1[i] = gene_a
                    genome_2[i] = gene_b
                # _end_if_

            # _end_for_

            # Create the two NEW offsprings.
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
