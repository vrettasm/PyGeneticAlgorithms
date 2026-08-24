
class IslandOperator:

    @staticmethod
    def set_pointer(operators: list, idx: int) -> None:
        """
        Set the pointer to the genetic operator that
        we want to execute.

        :param idx: the index of the mutation operator.
        :param operators: a list with genetic operators.

        :return: None.
        """
        # Sanity check.
        if idx < 0 or idx >= len(operators):
            raise IndexError("selected index out of range.")
        # _end_if_

        # Update the index in the dict.
        operators["idx"] = idx
    # _end_def_

    @staticmethod
    def all_counters(operators: list) -> dict:
        """
        Accessor (getter) of the application counter from all
        the internal mutators. This is mostly to verify that
        everything is working as expected.

        :return: a dictionary with the counter calls for all
                 mutator methods.
        """
        return {
            f"{n}-{mut_op.__class__.__name__}": mut_op.counter
            for n, mut_op in enumerate(operators)
        }
    # _end_def_

    @staticmethod
    def reset_counter(operators: list) -> None:
        """
        Sets ALL the counters to zero.

        :return: None.
        """
        # Call the reset on each of the internal operators.
        for op in operators:
            op.reset_counter()
    # _end_def_

# _end_class_
