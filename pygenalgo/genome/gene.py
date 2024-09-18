from typing import Any, Callable
from copy import deepcopy


class Gene(object):
    """
    Description:

        This is the main class that encodes the data of a single Gene in the chromosome.
        The class encapsulates not only the data, but also the way that this gene can be
        mutated using a random function. This Gene can be from a single 'bit' to a whole
        image. This way provides us with flexibility to parameterize the chromosome with
        different "kinds of genes" each one responsible for a specific function.
    """

    # Object variables.
    __slots__ = ("_datum", "_func", "valid")

    def __init__(self, _datum: Any, _func: Callable, valid: bool = True):
        """
        Initialize a Gene object.
        
        :param _datum: Datum holds a reference of the gene-data structure.
        
        :param _func: This 'private' function is used in the 'random()' method to be used by the mutation operators.
        
        :param valid: This flag is used to set the Gene as valid (True) or invalid (False).
        """
        
        # Copy the data reference.
        self._datum = _datum
        
        # Make sure the random function is callable.
        if not callable(_func):
            raise TypeError(f"{self.__class__.__name__}: Random function is not callable.")
        else:
            # Get the random function.
            self._func = _func
        # _end_if_
        
        # Copy the valid flag. Note that if the '_datum' field is set to None,
        # the Gene is automatically invalid.
        self.valid = False if self._datum is None else valid
    # _end_def_

    @property
    def datum(self) -> Any:
        """
        Accessor (getter) of the data reference.

        :return: the datum value.
        """
        return self._datum
    # _end_def_

    @property
    def is_valid(self) -> bool:
        """
        Accessor (getter) of the validity parameter.

        :return: the valid value.
        """
        return self.valid
    # _end_def_

    @is_valid.setter
    def is_valid(self, new_value: bool) -> None:
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

    def __add__(self, other) -> list:
        """
        When two genes are added we return a list with the two genes.

        :param other: the second object we want to 'add' to.

        :return: a list with [self, other].
        """

        # Make sure both items are of type 'Gene'.
        if isinstance(other, Gene) and self != other:
            return [self, other]
        else:
            raise NotImplemented
        # _end_if_
    # _end_def_

    def __eq__(self, other) -> bool:
        """
        When we compare two Genes we care only about the data they hold.

        :param other: the second object we want to compare to.

        :return: true or false.
        """

        # Make sure both items are of type 'Gene'.
        if isinstance(other, Gene):
            return self._datum == other._datum
        # _end_if_

        return False
    # _end_def_

    def __hash__(self) -> int:
        """
        Auxiliary method to hash the Gene object.

        :return: the hash value of the datum.
        """
        return hash(self._datum)
    # _end_def_

    def random(self) -> None:
        """
        This method should be different for each type of Gene. It describes
        how a specific type of Gene creates a random version of itself. The
        main  idea is that inside the Chromosome, each Gene can represent a
        very different concept of the  problem solution, so its Gene should
        have its own way to perform random mutation.

        This way by calling on the random() method, each Gene will know how
        to mutate itself without breaking any rules/constraints.

        :return: None.
        """

        # Use the random function to set a new value at the data.
        self._datum = self._func()
    # _end_def_

    def flip(self) -> None:
        """
        This method flips the value of the gene data. It is used only dy the
        FlipMutator operator for problems where the chromosome is represented
        by a list of bits.

        i)   1 -> 0
        ii)  0 -> 1

        :return: None.
        """

        self._datum = int(not self._datum)
    # _end_def_

    def __str__(self) -> str:
        """
        Override to print a readable string presentation of the object.

        :return: a string representation of a Gene object.
        """
        return f"{self.__class__.__name__}: datum={self._datum}"
    # _end_def_

    def __repr__(self) -> str:
        """
        Repr operator is called when a string representation is needed that can be evaluated.

        :return: Gene().
        """
        return f"{self.__class__.__name__}(datum={self._datum}, _func={self._func}, valid={self.valid})"
    # _end_def_

    def clone(self):
        """
        Makes a duplicate of the self object.

        :return: a 'deep-copy' of the object.
        """
        return deepcopy(self)
    # _end_def_

# _end_class_
