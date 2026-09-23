""" Order crossover (OX1) operator module. """
from typing import Callable
from functools import partial

# Custom code imports.
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.utilities import two_indices_fast
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offsprings)


class OrderCrossover(CrossoverOperator):
    """
    Description:

    Order crossover (OX1) creates offspring that preserve a contiguous segment
    from one parent and preserve the relative order of the remaining genes from
    the other parent. It is commonly used for permutation-based problems.
    """

    def __init__(self, crossover_probability: float = 0.9) -> None:
        """
        Construct a 'OrderCrossover' object with a given probability value.

        :param crossover_probability: (float).
        """
        # Call the super constructor with the provided initial value.
        super().__init__(crossover_probability=crossover_probability)
    # _end_def_

    @staticmethod
    def _ox1_mix_genomes(g1: list[Gene], g2: list[Gene], loc1: int, loc2: int,
                         n_size: int) -> list[Gene]:
        """
        Helper method to execute standard OX1 logic on raw lists.
        """
        # Slice and clone the middle segment in a single pass.
        mid_segment: list[Gene] = [
            gene.clone() for gene in g1[loc1:loc2]
        ]

        # Convert to set for O(1) tracking.
        copied_set: set[Gene] = set(mid_segment)

        # Extract the remaining items from g2 starting from loc2.
        remaining: list[Gene] = [
            gene.clone()
            for gene in (g2[loc2:] + g2[:loc2])
            if gene not in copied_set
        ]

        # Reconstruct the child's genome by splitting
        # the remaining list around the middle segment.
        split: int = n_size - loc2

        return remaining[split:] + mid_segment + remaining[:split]
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

            # Get the number of genes.
            n_genes: int = len(parent1)

            # Select two random (distinct) crossover points.
            # The values are returned in order: loc1 < loc2.
            loc1, loc2 = two_indices_fast(self.rng, n_genes,
                                          in_order=True)
            # Local (partial) function.
            make_child_genome: Callable = partial(self._ox1_mix_genomes,
                                                  loc1=loc1, loc2=loc2, n_size=n_genes)
            # Generate child1 genome.
            child_1: list[Gene] = make_child_genome(parent1.genome, parent2.genome)

            # Generate child2 genome.
            child_2: list[Gene] = make_child_genome(parent2.genome, parent1.genome)

            # Increase the crossover counter.
            self.inc_counter()

            # Return two new offsprings.
            return Chromosome(child_1), Chromosome(child_2)
        # _end_if_

        # Return two cloned offsprings.
        return parent1.clone(), parent2.clone()
    # _end_def_

# _end_class_
