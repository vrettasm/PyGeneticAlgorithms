from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator


class SinglePointCrossover(CrossoverOperator):
    """
    Description:

        Single-point crossover creates two children chromosomes (offsprings),
        by taking two parent chromosomes and cutting them at some, randomly
        chosen, site (locus).

        It produces very slow mixing, compared with multipoint or uniform crossover.
    """

    def __init__(self, crossover_probability: float = 0.9) -> None:
        """
        Construct a 'SinglePointCrossover' object with
        a given probability value.

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

            # Get the lengths of the chromosomes.
            length_1: int = len(parent1)
            length_2: int = len(parent2)

            # Find the minimum length.
            min_length: int = min(length_1, length_2)

            # Select randomly a crossover point from [0, min_length-1].
            idx: int = self.rng.integers(0, high=min_length, dtype=int)

            # Construct 1st offspring genome list at 'idx'.
            genome_1: list[Gene] = [
                x.clone() for x in parent2.genome[:idx] +
                                   parent1.genome[idx:]
            ]

            # Construct 2nd offspring genome list at 'idx'.
            genome_2: list[Gene] = [
                y.clone() for y in parent1.genome[:idx] +
                                   parent2.genome[idx:]
            ]

            # Create two NEW offsprings.
            child1 = Chromosome(genome_1)
            child2 = Chromosome(genome_2)

            # Increase the crossover counter.
            self.inc_counter()
        else:
            # Otherwise each child will point
            # to a deepcopy of a single parent.
            child1 = parent1.clone()
            child2 = parent2.clone()
        # _end_if_

        # Return the two offsprings.
        return child1, child2
    # _end_def_

# _end_class_
