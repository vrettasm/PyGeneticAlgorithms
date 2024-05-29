from typing import Any
from dataclasses import dataclass

@dataclass(init=True, repr=True)
class Gene(object):

    # Define the class members.
    datum: Any = None
    valid: bool = True

    @property
    def is_valid(self):
        """
        Accessor (getter) of the validity parameter.

        :return: the valid value.
        """
        return self.valid
    # _end_def_

    @is_valid.setter
    def is_valid(self, new_value):
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

# _end_class_
