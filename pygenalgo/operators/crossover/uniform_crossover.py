""" Uniform crossover (UX) operator module. """
# Custom code imports.
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offsprings)


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
        if (parent1 != parent2) and self.is_operator_applicable():

            # Create the 1st offspring genome list.
            child_1: list[Gene] = [
                gene.clone() for gene in parent1.genome
            ]

            # Create the 2nd offspring genome list.
            child_2: list[Gene] = [
                gene.clone() for gene in parent2.genome
            ]

            # Find the minimum length of the two chromosomes.
            min_length: int = min(len(child_1), len(child_2))

            # Generate uniform random numbers and convert them to bool.
            swap_bool_flag = self.rng.random(size=min_length) > 0.5

            # Swap the genes according to the probability.
            for i, swap_flag in enumerate(swap_bool_flag):
                if swap_flag:
                    child_1[i], child_2[i] = child_2[i], child_1[i]
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
