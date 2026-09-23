""" Position based crossover (POS) operator module. """
# Custom code imports.
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offsprings)


class PositionBasedCrossover(CrossoverOperator):
    """
    Description:

        Position based crossover (POS) creates two children chromosomes, by ensuring that the
        original genome (from both parents) isn't repeated, thus creating invalid offsprings.

        It is used predominantly in combinatorial problems.
    """

    def __init__(self, crossover_probability: float = 0.9) -> None:
        """
        Construct a 'PositionBasedCrossover' object with a given probability value.

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

            # Get the size of the chromosomes.
            number_of_genes: int = len(parent1)

            # Select randomly a number of crossover points.
            number_of_points: int = self.rng.integers(1,
                                                      high=number_of_genes-1,
                                                      dtype=int)
            # Select randomly the crossover points.
            cross_points = self.rng.choice(number_of_genes,
                                           size=number_of_points,
                                           replace=False, shuffle=False)

            # Initialize the genome lists for the new
            # chromosomes to 'None'.
            child_1: list = number_of_genes * [None]
            child_2: list = number_of_genes * [None]

            # Create empty sets for fast lookups.
            added_to_c1: set[Gene] = set()
            added_to_c2: set[Gene] = set()

            # Copy the genes of the parents at
            # the preselected gene cross points.
            for i in cross_points:
                child_1[i] = parent1.genome[i].clone()
                child_2[i] = parent2.genome[i].clone()

                # Update the sets.
                added_to_c1.add(parent1.genome[i])
                added_to_c2.add(parent2.genome[i])
            # _end_for_

            # Fill remaining slots keeping the relative order of the OTHER parent.
            idx_c1: int = 0
            idx_c2: int = 0

            # Fill Child 1 using Parent 2's remaining sequence.
            for gene2 in parent2.genome:
                if gene2 not in added_to_c1:
                    # Find the next available empty slot
                    # in child_1.
                    while child_1[idx_c1] is not None:
                        idx_c1 += 1

                    # Clone the gene.
                    child_1[idx_c1] = gene2.clone()

                    # Update the set.
                    added_to_c1.add(gene2)
            # _end_for_

            # Fill Child 2 using Parent 1's remaining sequence.
            for gene1 in parent1.genome:
                if gene1 not in added_to_c2:
                    # Find the next available empty slot
                    # in child_2.
                    while child_2[idx_c2] is not None:
                        idx_c2 += 1

                    # Clone the gene.
                    child_2[idx_c2] = gene1.clone()

                    # Update the set.
                    added_to_c2.add(gene1)
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
