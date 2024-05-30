from code.src.operators.genetic_operator import GeneticOperator

class MutationOperator(GeneticOperator):

    def __init__(self, mutation_probability: float = 0.001):
        """
        Construct a 'MutationOperator' object with a given
        probability value.

        :param mutation_probability: (float)
        """

        # Call the super constructor with the provided probability value.
        super().__init__(mutation_probability)
    # _end_def_

# _end_class_


mut_op = MutationOperator(-0.9)

print(mut_op)