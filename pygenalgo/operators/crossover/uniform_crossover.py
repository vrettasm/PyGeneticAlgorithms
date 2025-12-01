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

    def __init__(self, crossover_probability: float = 0.9) -> None:
        """
        Construct a 'UniformCrossover' object with a given
        probability value.

        :param crossover_probability: (float).
        """
        # Call the super constructor with the provided
        # probability value.
        super().__init__(crossover_probability)
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
        """
        Perform the crossover operation on the two input parent chromosomes.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """
        # If the crossover probability is higher than a uniformly
        # random value and the parents aren't identical apply the
        # changes.
        if (parent1 != parent2) and self.is_operator_applicable():

            # Get the length of the chromosome.
            number_of_genes = len(parent1)

            # Preallocate 1st offspring genome list.
            genome_1: list = [None] * number_of_genes

            # Preallocate 2nd offspring genome list.
            genome_2: list = [None] * number_of_genes

            # Generate uniform random numbers and convert them to bool.
            swap_bool_flag = self.rng.random(size=number_of_genes) > 0.5

            # Set the genes according to the swap probability.
            for i, (swap_flag, gene_1, gene_2) in enumerate(zip(swap_bool_flag,
                                                                parent1.genome,
                                                                parent2.genome)):
                if swap_flag:
                    genome_1[i] = gene_2.clone()
                    genome_2[i] = gene_1.clone()
                else:
                    genome_1[i] = gene_1.clone()
                    genome_2[i] = gene_2.clone()
            # _end_for_

            # Create two NEW offsprings.
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
