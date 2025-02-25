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

        # If the crossover probability is higher than
        # a uniformly random value, make the changes.
        if self.is_operator_applicable():

            # Select randomly the crossover point.
            locus = self.rng.integers(1, high=len(parent1))

            # Create the two NEW offsprings at locus.
            child1 = Chromosome(parent1[:locus] + parent2[locus:])
            child2 = Chromosome(parent2[:locus] + parent1[locus:])

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
