from numpy import nan as np_nan
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

    def __init__(self, crossover_probability: float = 0.9):
        """
        Construct a 'UniformCrossover' object with a given
        probability value.

        :param crossover_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(crossover_probability)
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome):
        """
        Perform the crossover operation on the two input parent chromosomes.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """

        # Initially each child points to a clone of a single parent.
        child1 = parent1.clone()
        child2 = parent2.clone()

        # If the crossover probability is higher than
        # a uniformly random value, make the changes.
        if self.probability > self.rng.random():

            # Get the number of genes from the first parent (chromosome).
            # N.B.: It is assumed that both parents have the same size.
            N = len(parent1)

            # Swap flag.
            swap_flag = False

            # Generate 'N' random numbers in one call.
            swap_prob = self.rng.random(N)

            # Go through all the children's genome.
            for i in range(0, N):

                # The two genes will swap with 50% probability.
                if swap_prob[i] > 0.5:

                    # Swap in place between the two positions.
                    child1[i], child2[i] = child2[i], child1[i]

                    # Set the swap to true.
                    swap_flag = True
                # _end_if_

            # _end_for_

            # Even if one gene has swapped the fitness will not be accurate.
            if swap_flag:
                child1.fitness = np_nan
                child2.fitness = np_nan
            # _end_if_

            # Increase the crossover counter.
            self.inc_counter()

        # _end_if_

        # Return the two offsprings.
        return child1, child2
    # _end_def_

# _end_class_
