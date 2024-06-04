from src.operators.genetic_operator import GeneticOperator

class SelectionOperator(GeneticOperator):

    def __init__(self, selection_probability: float):
        """
        Construct a 'SelectionOperator' object with a
        given probability value.

        :param selection_probability: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(selection_probability)
    # _end_def_

# _end_class_
