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

    def __init__(self, crossover_probability: float = 0.9):
        """
        Construct a 'SinglePointCrossover' object with
        a given probability value.

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
        # If the crossover probability is higher than a uniformly
        # random value and the parents aren't identical apply the
        # changes.
        if (parent1 != parent2) and self.is_operator_applicable():

            # Select randomly the crossover point.
            locus = self.rng.integers(1, high=len(parent1))

            # Construct 1st offspring genome list at locus.
            genome_1 = [x.clone() for x in parent1.genome[:locus] +
                        parent2.genome[locus:]]

            # Construct 2nd offspring genome list at locus.
            genome_2 = [y.clone() for y in parent2.genome[:locus] +
                        parent1.genome[locus:]]

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
