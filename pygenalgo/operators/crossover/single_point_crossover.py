""" Single point crossover operator module. """
# Custom code imports.
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offspring)


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
        # Call the super constructor with the provided initial value.
        super().__init__(crossover_probability=crossover_probability)
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Offspring:
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

            # Create the 1st offspring genome list.
            child_1: list[Gene] = parent1.clone_genome()

            # Create the 2nd offspring genome list.
            child_2: list[Gene] = parent2.clone_genome()

            # Find the minimum length of the two chromosomes.
            min_length: int = min(len(child_1), len(child_2))

            # Select randomly a crossover point from [1, min_length-1].
            idx: int = self.rng.integers(1, high=min_length, dtype=int)

            # Swap their genes in the range idx to min_length.
            child_1[idx:min_length], child_2[idx:min_length] = (child_2[idx:min_length],
                                                                child_1[idx:min_length])
            # Increase the crossover counter.
            self.inc_counter()

            # Return two new offsprings.
            return Chromosome(child_1), Chromosome(child_2)
        # _end_if_

        # Return two cloned offsprings.
        return parent1.clone(), parent2.clone()
    # _end_def_

# _end_class_
