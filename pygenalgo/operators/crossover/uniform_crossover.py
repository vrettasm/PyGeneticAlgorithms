from pygenalgo.genome.gene import Gene
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
        # Call the super constructor with the provided initial value.
        super().__init__(crossover_probability=crossover_probability)
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

            # Get the lengths of the chromosomes.
            length_1: int = len(parent1)
            length_2: int = len(parent2)

            # Create the 1st offspring genome list.
            genome_1: list[Gene] = [
                gene.clone() for gene in parent1.genome
            ]

            # Create the 2nd offspring genome list.
            genome_2: list[Gene] = [
                gene.clone() for gene in parent2.genome
            ]

            # Find the minimum length.
            min_length: int = min(length_1, length_2)

            # Generate uniform random numbers and convert them to bool.
            swap_bool_flag = self.rng.random(size=min_length) > 0.5

            # Swap the genes according to the probability.
            for i, swap_flag in enumerate(swap_bool_flag):
                if swap_flag:
                    genome_1[i], genome_2[i] = genome_2[i], genome_1[i]
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
