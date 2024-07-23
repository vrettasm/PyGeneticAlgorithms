from src.genome.chromosome import Chromosome
from src.operators.crossover.crossover_operator import CrossoverOperator
from src.operators.crossover.uniform_crossover import UniformCrossover
from src.operators.crossover.mutli_point_crossover import MultiPointCrossover
from src.operators.crossover.single_point_crossover import SinglePointCrossover


class SuperCrossover(CrossoverOperator):
    """
    Description:

        Super crossover, crosses the chromosomes by applying randomly
        all other crossovers (one at a time), with equal probability.
    """

    # Tuple with all the other crossover operators. Note that in here
    # the crossover probabilities for each operator are set to 1.0.
    cross = (UniformCrossover(1.0), MultiPointCrossover(1.0), SinglePointCrossover(1.0))

    def __init__(self, crossover_probability: float = 0.9):
        """
        Construct a 'SuperCrossover' object with a given
        probability value.

        :param crossover_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(crossover_probability)

    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome):
        """
        Perform the crossover operation on the two input parent
        chromosomes, by selecting randomly a predefined method.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """

        # If the crossover probability is higher than
        # a uniformly random value, make the changes.
        if self.probability >= self.rng.random():

            # Select randomly the crossover method.
            index = self.rng.integers(len(self.cross))

            # Call the selected crossover.
            self.cross[index].crossover(parent1, parent2)

            # Increase the crossover counter.
            self.inc_counter()
        # _end_if_

    # _end_def_

# _end_class_
