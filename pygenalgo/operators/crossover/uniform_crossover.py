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
        # a uniformly random value, apply the changes.
        if self.is_operator_applicable():

            # Get the length of the chromosome.
            number_of_genes = len(parent1)

            # Preallocate 1st genome.
            genome_1 = [None] * number_of_genes

            # Preallocate 2nd genome.
            genome_2 = [None] * number_of_genes

            # Generate all random numbers in one call.
            swap_probability = self.rng.random(size=number_of_genes) > 0.5

            # Allocate the genes according to the swap probability.
            for i, (flag, p1, p2) in enumerate(zip(swap_probability.tolist(),
                                                   parent1.genome, parent2.genome)):
                if flag:
                    genome_1[i] = p1
                    genome_2[i] = p2
                else:
                    genome_1[i] = p2
                    genome_2[i] = p1
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
