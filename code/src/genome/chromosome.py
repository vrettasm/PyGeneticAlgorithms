from copy import deepcopy
from dataclasses import dataclass, field

from src.genome.gene import Gene


@dataclass(init=True, repr=True)
class Chromosome(object):

    # Define the genome as a list of genes. This list
    # will encode a "single solution to the problem".
    _genome: list = field(default_factory=list)

    # The fitness value will correspond to how well the
    # chromosome fits in its environment, as defined by
    # the fitness function.
    _fitness: float = 0.0

    # Define a boolean flag. This flag here can be used
    # to include hard/soft constraints to the chromosome.
    _valid: bool = True

    @property
    def valid(self) -> bool:
        """
        Accessor (getter) of the validity parameter.

        :return: the valid value.
        """
        return self._valid
    # _end_def_

    @valid.setter
    def valid(self, new_value: bool):
        """
        Accessor (setter) of the validity flag.

        :param new_value: (bool).
        """
        # Check for correct type.
        if isinstance(new_value, bool):

            # Update the flag value.
            self._valid = new_value
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Validity flag should be bool: {type(new_value)}.")
        # _end_if_

    # _end_def_

    @property
    def fitness(self) -> float:
        """
        Accessor of the fitness value of the chromosome.

        :return: the fitness (float) of the genome.
        """
        return self._fitness
    # _end_def_

    @fitness.setter
    def fitness(self, new_value: float):
        """
        Accessor (setter) of the fitness value.

        :param new_value: (float).
        """

        # Check for correct type.
        if isinstance(new_value, float):

            # Update the fitness value.
            self._fitness = new_value

        elif isinstance(new_value, int):

            # First convert the input to 'float'
            # and then update the fitness value.
            self._fitness = float(new_value)
        else:

            raise TypeError(f"{self.__class__.__name__}: "
                            f"Fitness should be float: {type(new_value)}.")
        # _end_if_

    # _end_def_

    def is_genome_valid(self) -> bool:
        """
        Checks the validity of the whole chromosome, by
        calling individually all genes is_valid method.

        In addition, it "double-checks" that all entries
        in the genome are of type 'Gene'.

        :return: True if ALL genes are valid, else False.
        """
        return all(isinstance(x, Gene) and x.is_valid
                   for x in self._genome)
    # _end_def_

    def __len__(self) -> int:
        """
        Accessor of the total length of the genome.

        :return: the length (int) of the genome.
        """
        return len(self._genome)
    # _end_def_

    def __getitem__(self, index: int):
        return self._genome[index]
    # _end_def_

    def __setitem__(self, index: int, item: Gene):
        self._genome[index] = item
    # _end_def_

    def __contains__(self, item: Gene) -> bool:
        """
        Check for membership.

        :param item: an input Gene that we want to check.

        :return: true if the 'item' belongs in the genome.
        """
        return item in self._genome
    # _end_if_

    def clone(self):
        """
        Makes a duplicate of the self object.

        :return: a 'deep-copy' of the object.
        """
        return deepcopy(self)
    # _end_def_

# _end_class_
