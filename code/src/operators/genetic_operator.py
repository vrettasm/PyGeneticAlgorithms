from numpy.random import default_rng

# Public interface.
__all__ = ["GeneticOperator"]

# Current version.
__version__ = "0.1.0"

# Author.
__author__ = "Michalis Vrettas, PhD"

# Email.
__email__ = "michail.vrettas@gmail.com"

class GeneticOperator(object):

    # Object variables.
    __slots__ = ("_probability", "_rng")

    def __init__(self, _probability: float):
        """
        Construct a 'GeneticOperator' object with a given
        probability value.

        :param _probability: (float).
        """

        # Ensure the default entry value is within range [0, 1].
        self._probability = max(min(float(_probability), 1.0), 0.0)

        # Create a random number generator.
        self._rng = default_rng()
    # _end_def_

    @property
    def probability(self):
        """
        Accessor (getter) of the probability.

        :return: the float value of the probability.
        """
        return self._probability
    # _end_def_

    @probability.setter
    def probability(self, new_value: float):
        """
        Accessor (setter) of the probability.

        :param new_value: (float) in [0, 1].
        """

        # Check for the correct type.
        if isinstance(new_value, float):

            # Ensure the correct range of values.
            if 0.0 <= new_value <= 1.0:

                # Update the probability value.
                self._probability = new_value
            else:
                ValueError(f"{self.__class__.__name__}: "
                           f"Probability should be in [0, 1].")
            # _end_if_
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Probability should be float: {type(new_value)}.")
        # _end_if_

    # _end_def_

    @property
    def rng(self):
        """
        Accessor method.

        :return: the random number generator.
        """
        return self._rng
    # _end_def_

    def __str__(self):
        """
        Description:
            Override to print a readable string presentation of the
            genetic operator object, using the selected parameters.
        """

        # Initialize the string with the class name and its ID.
        str_self = f" {self.__class__.__name__}: ({id(self)})\n"

        # Add all the fields with their values.
        for s in self.__slots__:
            str_self += (" " + s + ": " +
                         str(self.__getattribute__(s)) + "\n")
        # _end_for_

        # Return the string.
        return str_self
    # _end_def_

    def __repr__(self):
        """
        Description:
            Override to provide a simple string that’s a valid Python expression
            which could be used to recreate the object: ClassName(_probability).
        """
        return f"{self.__class__.__name__}({self._probability})"
    # _end_def_

# _end_class_
