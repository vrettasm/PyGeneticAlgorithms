from dataclasses import dataclass
from typing import Any


@dataclass(init=True, repr=True)
class Gene(object):

    # Define the class members.
    datum: Any = None
    _func: Any = None
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
        This is a "placeholder"  method, and it should be different
        for each type of Gene. It describes how  a specific type of
        Gene creates a random version  of itself. The main idea  is
        that inside  the Chromosome, each Gene can represent a very
        different concept of the  problem, so its  Gene should have
        its own way to perform random mutation. This way by calling
        on the "random()" method, each Gene will know how to mutate
        itself without breaking ay rules/constraints.

        Example:
        -----------------------------------------------------------
            # G1 represents a gene (in the chromosome list) that
            # is an integer.
            g1 = Gene(0)

            # We set the random function to call the randint(100).
            g1.random = lambda: Gene(np.random.randint(100))

            # Now, when we call this g1 will have a 'new' version
            # of itself with a different random value, generated
            # by the randint method.
            g1 = g1.random()
        -----------------------------------------------------------

        :return: None.
        """
        if callable(self._func):
            return self._func()
        else:
            raise NotImplementedError(f"{self.__class__.__name__}: "
                                      f"Random function is not set.")
        # _end_if_

    # _end_def_

# _end_class_
