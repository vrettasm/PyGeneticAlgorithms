""" Uniform crossover (UX) operator module. """
# Third party import.
import numpy as np

# Custom code imports.
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offspring)


class UniformCrossover(CrossoverOperator):
    """
    Description:

        Uniform crossover creates two children chromosomes (offspring),
        by taking two parent chromosomes and swap their genes independently
        at each location based on a uniform random coin flip.

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

            # Generate uniform random probabilities in [0, 1].
            swap_probabilities = self.rng.random(size=min_length)

            # Keep only those indices that are higher than 50%.
            swap_indices = np.where(swap_probabilities > 0.5)[0]

            # Swap the genes according to the probability.
            for i in swap_indices:
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
