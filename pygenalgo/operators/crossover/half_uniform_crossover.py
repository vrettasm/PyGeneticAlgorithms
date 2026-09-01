""" Half uniform crossover (HUX) operator module. """
# Custom code imports.
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offsprings)


class HalfUniformCrossover(CrossoverOperator):
    """
    Description:

        Half uniform crossover (HUX) creates two children chromosomes
        (offsprings). It identifies the positions where the parents differ,
        then swaps exactly half of those differing genes. Positions where
        the parents are identical remain unchanged.

        The primary benefit of Half Uniform Crossover (HUX) over Uniform
        Crossover (UX) is the preservation of genetic diversity and the
        prevention of premature convergence.
    """

    def __init__(self, crossover_probability: float = 0.9) -> None:
        """
        Construct a 'HalfUniformCrossover' object with a
        given probability value.

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

            # Create the 1st offspring genome list.
            child_1: list[Gene] = [
                gene.clone() for gene in parent1.genome
            ]

            # Create the 2nd offspring genome list.
            child_2: list[Gene] = [
                gene.clone() for gene in parent2.genome
            ]

            # Find differences in the genomes.
            diff_indices: list[int] = [
                i for i, (g1, g2) in enumerate(zip(child_1, child_2)) if g1 != g2
            ]

            # Determine exactly half to swap.
            num_swap: int = len(diff_indices) // 2

            # Randomly sample num_swap indices.
            swap_indices = self.rng.choice(diff_indices,num_swap,
                                           replace=False)

            # Swap the genes at random locations.
            for j in swap_indices:
                child_1[j], child_2[j] = child_2[j], child_1[j]
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
