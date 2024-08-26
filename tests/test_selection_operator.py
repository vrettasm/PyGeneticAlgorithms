import unittest
from pygenalgo.operators import SelectionOperator


class TestSelectionOperator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(">> TestSelectionOperator - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(">> TestSelectionOperator - FINISH -", end='\n\n')
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a valid probability.
        self.select_op = SelectionOperator(selection_probability=0.5)
    # _end_def_

    def test_select(self):
        """
        The select method should be implemented.

        :return: None.
        """

        # Create an empty population of parents.
        population = []

        # Check if the select method is implemented.
        with self.assertRaises(NotImplementedError):
            self.select_op.select(population)
        # _end_with_

    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
