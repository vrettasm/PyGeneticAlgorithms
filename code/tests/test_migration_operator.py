import unittest
from src.engines.auxiliary import SubPopulation
from src.operators.migration.migration_operator import MigrationOperator


class TestMigrationOperator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestMigrationOperator - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestMigrationOperator - FINISH -")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default settings.

        :return: None.
        """
        # Create an object with a valid probability.
        self.mig_op = MigrationOperator(migration_probability=0.5)
    # _end_def_

    def test_migrate(self):
        """
        The migrate method should be implemented.

        :return: None.
        """

        # Create a dummy list.
        pop = []

        # Check if the migrate method is implemented.
        with self.assertRaises(NotImplementedError):
            self.mig_op.migrate(pop)
        # _end_with_

    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()