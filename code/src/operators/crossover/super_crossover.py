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

            # Select randomly, with equal probability (but this can be changed),
            # a crossover operator and call its crossover method.
            child1, child2 = self.rng.choice(self.cross).crossover(parent1, parent2)

            # Increase the crossover counter.
            self.inc_counter()
        else:
            # Otherwise each child points to a clone of a single parent.
            child1 = parent1.clone()
            child2 = parent2.clone()
        # _end_if_

        # Return the two offsprings.
        return child1, child2

    # _end_def_

    @property
    def all_counters(self):
        """
        Accessor (getter) of the application counter from all the internal crossovers.
        This is mostly to verify that everything is working as expected.

        :return: a dictionary with the counter calls for all crossover methods.
        """
        return {cross_op.__class__.__name__: cross_op.counter for cross_op in self.cross}
    # _end_def_

# _end_class_
