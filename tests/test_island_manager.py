import unittest
from unittest.mock import MagicMock

from pygenalgo.engines.island_manager import IslandManager
from pygenalgo.operators.genetic_operator import GeneticOperator


class TestIslandManager(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up mock operators and a new IslandManager
        instance for each test.

        :return: None.
        """
        # Create mock GeneticOperator instances to isolate
        # the class behavior.
        self.mock_op_0 = MagicMock(spec=GeneticOperator)
        self.mock_op_1 = MagicMock(spec=GeneticOperator)

        # Configure mock counters for property/method testing.
        self.mock_op_0.counter = 5
        self.mock_op_1.counter = 12

        # Create an IslandManager.
        self.operators = [self.mock_op_0, self.mock_op_1]
        self.manager = IslandManager(self.operators)
    # _end_def_

    def test_slots_defined(self) -> None:
        """
        Verify __slots__ configuration is correct and
        that it prevents arbitrary attribute binding.

        :return: None.
        """
        # Correct configuration.
        self.assertEqual(set(IslandManager.__slots__),
                         {"_operators", "_idx"})

        # Adding new fields should not be allowed.
        with self.assertRaises(AttributeError):
            setattr(self.manager, "new_attribute", None)
    # _end_def_

    def test_initialization_success(self) -> None:
        """
        Verify successful instantiation yields correct initial state.

        :return: None.
        """
        self.assertEqual(self.manager.idx, 0)
        self.assertEqual(self.manager.operator, self.mock_op_0)
        self.assertEqual(self.manager.operators_list, self.operators)
    # _end_def_

    def test_initialization_with_none_raises_value_error(self) -> None:
        """
        Verify passing None as the operator list raises a ValueError.

        :return: None.
        """
        with self.assertRaises(ValueError):
            IslandManager(None)
    # _end_def_

    def test_initialization_with_empty_list_raises_value_error(self) -> None:
        """
        Verify passing an empty operator list raises a ValueError.

        :return: None.
        """
        with self.assertRaises(ValueError):
            IslandManager([])
    # _end_def_

    def test_set_idx_valid(self) -> None:
        """
        Verify index can be updated to a valid boundary
        and changes the current operator.

        :return: None.
        """
        self.manager.idx = 1
        self.assertEqual(self.manager.idx, 1)
        self.assertEqual(self.manager.operator, self.mock_op_1)
    # _end_def_

    def test_set_idx_type_coercion(self) -> None:
        """
        Verify index accepts numeric types or strings
        that cleanly parse to integer.

        :return: None.
        """
        self.manager.idx = "1"
        self.assertEqual(self.manager.idx, 1)

        # Converting from float should not be allowed
        with self.assertRaises(ValueError):
            self.manager.idx = "1.0"
    # _end_def_

    def test_set_idx_out_of_bounds_negative(self) -> None:
        """
        Verify negative indices raise an IndexError.

        :return: None.
        """
        with self.assertRaises(IndexError):
            self.manager.idx = -1
    # _end_def_

    def test_set_idx_out_of_bounds_positive(self) -> None:
        """
        Verify indices equal to or exceeding list
        length raise an IndexError.

        :return: None.
        """
        with self.assertRaises(IndexError):
            self.manager.idx = 2
    # _end_def_

    def test_get_all_counters(self) -> None:
        """
        Verify formatting structure and accuracy
        of combined application counters.

        :return: None.
        """
        expected_dict = {
            "0-GeneticOperator": 5,
            "1-GeneticOperator": 12
        }

        self.assertEqual(self.manager.get_all_counters(),
                         expected_dict)
    # _end_def_

    def test_reset_island_counters(self) -> None:
        """
        Verify that reset method delegates reset
        tracking call down to all operators.

        :return: None.
        """
        self.manager.reset_island_counters()
        self.mock_op_0.reset_counter.assert_called_once()
        self.mock_op_1.reset_counter.assert_called_once()
    # _end_def_

if __name__ == '__main__':
    unittest.main()
