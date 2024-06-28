from dataclasses import dataclass
from typing import Any


@dataclass(init=True, repr=True)
class Gene(object):

    # Datum holds a reference of the gene-data structure.
    datum: Any = None

    # This 'private' function is used in the 'random()'
    # method to be used by the mutation operators.
    _func: Any = None

    # This flag is used to set the Gene as valid (True)
    # or invalid (False).
    valid: bool = True

    @property
    def is_valid(self) -> bool:
        """
        Accessor (getter) of the validity parameter.

        :return: the valid value.
        """
        return self.valid
    # _end_def_

    @is_valid.setter
    def is_valid(self, new_value: bool):
        """
        Accessor (setter) of the validity flag.

        :param new_value: (bool).
        """
        # Check for correct type.
        if isinstance(new_value, bool):

            # Update the flag value.
            self.valid = new_value
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Validity flag should be bool: {type(new_value)}.")
        # _end_if_
    # _end_def_

    def __add__(self, other):
        """
        When two genes are added we return a list with the two genes.

        :param other: the second object we want to 'add' to.

        :return: a list with [self, other].
        """

        # Make sure both items are of type 'Gene'.
        if isinstance(other, Gene):
            return [self, other]
        else:
            raise NotImplemented
        # _end_if_

    # _end_def_

    def random(self):
        """
        This method should be different for each type of Gene. It describes
        how a specific type of Gene creates a random version of itself. The
        main  idea is that inside the Chromosome, each Gene can represent a
        very different concept of the  problem solution, so its Gene should
        have its own way to perform random mutation.

        This way by calling on the random() method, each Gene will know how
        to mutate itself without breaking ay rules/constraints.

        :return: _func().
        """
        if callable(self._func):
            return self._func()
        else:
            raise NotImplementedError(f"{self.__class__.__name__}: "
                                      f"Random function is not set.")
        # _end_if_

    # _end_def_

# _end_class_
