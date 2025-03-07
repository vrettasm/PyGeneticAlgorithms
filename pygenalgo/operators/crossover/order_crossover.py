from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator


class OrderCrossover(CrossoverOperator):
    """
    Description:

        Order crossover (OX1) creates two children chromosomes, by ensuring that the original
        genome (from both parents) isn't repeated, thus creating invalid offsprings.

        It is used predominantly in combinatorial problems.
    """

    def __init__(self, crossover_probability: float = 0.9):
        """
        Construct a 'OrderCrossover' object with a given probability value.

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

            # Select randomly the crossover point from [1, M-1].
            locus = self.rng.integers(1, high=len(parent1)-1)

            # Create auxiliary Sets for faster membership check.
            used_parent1 = set(parent1.genome[:locus])
            used_parent2 = set(parent2.genome[:locus])

            # Preallocate 1st genome.
            genome_1 = [gene.clone() for gene in parent1.genome[:locus] +
                        [x for x in parent2 if x not in used_parent1]]

            # Preallocate 2nd genome.
            genome_2 = [gene.clone() for gene in parent2.genome[:locus] +
                        [y for y in parent1 if y not in used_parent2]]

            # Create the 1st NEW offspring at locus.
            child1 = Chromosome(genome_1)

            # Create the 2nd NEW offspring at locus.
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
