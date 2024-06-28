from src.genome.chromosome import Chromosome
from src.operators.genetic_operator import GeneticOperator

class CrossoverOperator(GeneticOperator):

    def __init__(self, crossover_probability):
        """
        Construct a 'CrossoverOperator' object with a given
        probability value.

        :param crossover_probability: (float).
        """

        # Call the super constructor with the provided probability value.
        super().__init__(crossover_probability)
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome):
        """
        Abstract method that "reminds" the user that if they want to
        create a Crossover Class that inherits from here they should
        implement a crossover method.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: Nothing but raising an error.
        """
        raise NotImplementedError(f"{self.__class__.__name__}: "
                                  f"You should implement this method!")

# _end_class_
