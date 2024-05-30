from code.src.operators.genetic_operator import GeneticOperator

class CrossoverOperator(GeneticOperator):

    def __init__(self, crossover_probability: float = 1.0):
        """
        Construct a 'CrossoverOperator' object with a given
        probability value.

        :param crossover_probability: (float)
        """

        # Call the super constructor with the provided probability value.
        super().__init__(crossover_probability)
    # _end_def_

# _end_class_
