""" Order crossover (OX1) operator module. """
# Custom code imports.
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.utils.utilities import two_indices_fast
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offsprings)


class OrderCrossover(CrossoverOperator):
    """
    Description:

        Order crossover (OX1) creates two children chromosomes, by ensuring that the original
        genome (from both parents) isn't repeated, thus creating invalid offsprings.

        It is used predominantly in combinatorial problems.
    """

    def __init__(self, crossover_probability: float = 0.9) -> None:
        """
        Construct a 'OrderCrossover' object with a given probability value.

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

            # Select two random (distinct) crossover points.
            loc1, loc2 = two_indices_fast(self.rng, len(parent1))

            # Create auxiliary Sets for faster membership check.
            used_in_parent1 = set(parent1.genome[:loc1])
            used_in_parent2 = set(parent2.genome[:loc2])

            # Construct 1st offspring genome list at locus.
            child_1: list[Gene] = [
                gene.clone() for gene in parent1.genome[:loc1] +
                                         [x for x in parent2 if x not in used_in_parent1]
            ]

            # Construct 2nd offspring genome list at locus.
            child_2: list[Gene] = [
                gene.clone() for gene in parent2.genome[:loc2] +
                                         [y for y in parent1 if y not in used_in_parent2]
            ]

            # Increase the crossover counter.
            self.inc_counter()

            # Return two new offsprings.
            return Chromosome(child_1), Chromosome(child_2)
        # _end_if_

        # Return two cloned offsprings.
        return parent1.clone(), parent2.clone()
    # _end_def_

# _end_class_
