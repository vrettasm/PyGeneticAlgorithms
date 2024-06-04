from src.genome.chromosome import Chromosome
from crossover_operator import CrossoverOperator


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

        # Get the number of genes from the first parent (chromosome).
        # N.B.: It is assumed that both parents have the same size.
        num_genes = len(parent1)

        # Initially each child points to a copy of a single parent.
        child1 = parent1.make_deepcopy()
        child2 = parent2.make_deepcopy()

        # Make a local copy of the crossover probability.
        swap_probability = self.probability

        # Go through all the children's genome.
        for i in range(0, num_genes):

            # Check the swap probability only for
            # the 'even' positions in the genome.
            if i % 2 == 0 and swap_probability >= self.rng.random():

                # Swap in place between the two positions.
                child1[i], child2[i] = child2[i], child1[i]
            # _end_if_

        # _end_for_

        # Increase the application counter.
        self.inc_counter()

        # Return the two offsprings.
        return child1, child2
    # _end_def_

# _end_class_
