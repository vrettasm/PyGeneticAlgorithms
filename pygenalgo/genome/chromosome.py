from copy import deepcopy
from dataclasses import dataclass, field

from pygenalgo.genome.gene import Gene


@dataclass(init=True, repr=True)
class Chromosome(object):
    """
    Description:

        Implements a dataclass for the Chromosome entity. This class is responsible
        for holding the individual solution(s), of the optimization problem, during
        the evolution process.
    """

    # Define the genome as a list of genes. This list
    # will encode a "single solution to the problem".
    _genome: list = field(default_factory=list[Gene])

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
    def valid(self, new_value: bool) -> None:
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
    def fitness(self, new_value: float) -> None:
        """
        Accessor (setter) of the fitness value.

        :param new_value: (float).
        """

        # Check for correct type.
        if isinstance(new_value, (int, float)):

            # Update the fitness value.
            self._fitness = float(new_value)
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Fitness should be float: {type(new_value)}.")
        # _end_if_
    # _end_def_

    @property
    def genome(self) -> list[Gene]:
        """
        Accessor of the genome list of the chromosome.

        :return: the list (of Genes) of the chromosome.
        """
        return self._genome
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

    def __getitem__(self, index: int) -> Gene:
        """
        Get the item at position 'index'.

        :param index: (int) the position that we want to return.

        :return: the reference to a Gene.
        """
        return self._genome[index]
    # _end_def_

    def __setitem__(self, index: int, item: Gene) -> None:
        """
        Set the 'item' at position 'index'.

        :param index: (int) the position that we want to access.

        :param item: (Gene) the object we want to assign in the genome.

        :return: None.
        """
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

    def hamming_distance(self, other) -> int:
        """
        Compute the "Hamming distance" of the "self" object with the
        "other" chromosome. In practise it's the number of positions
        at which the corresponding genes are different.

        :param other: (Chromosome) to compare the Hamming distance.

        :return: (int) the distance between the two chromosomes.
        """

        # Check for the correct type.
        if isinstance(other, Chromosome):

            # Quick exit if the object is itself.
            if self == other:
                return 0
            # _end_if_

            # Compute the dissimilarities in their genomes.
            return [k != l for k, l in zip(self._genome, other._genome)].count(True)
        else:
            raise NotImplementedError(f"{self.__class__.__name__}: "
                                      f"Can't compute Hamming distance in different class objects.")
        # _end_if_

    # _end_def_

    def __copy__(self):
        """
        This custom method overrides the default copy method
        and is used when we call the copy() method on a class
        object.

        :return: a (shallow) copy of the self object.
        """

        # Return the new copy.
        return Chromosome(self._genome, self._fitness, self._valid)
    # _end_copy_

    def __deepcopy__(self, memo):
        """
        This custom method overrides the default deepcopy method
        and is used when we call the "clone" method of the class.

        :param memo: dictionary of objects already copied during
        the current copying pass.

        :return: a new identical "clone" of the self object.
        """

        # Create a new instance.
        new_object = Chromosome.__new__(Chromosome)

        # Don't copy self reference.
        memo[id(self)] = new_object

        # Don't copy the cache.
        if hasattr(self, "_cache"):
            memo[id(self._cache)] = self._cache.__new__(dict)
        # _end_if_

        # Deep copy all other attributes.
        for key, value in self.__dict__.items():
            setattr(new_object, key, deepcopy(value, memo))
        # _end_for_

        # Return identical instance.
        return new_object
    # _end_def_

    def clone(self):
        """
        Makes a duplicate of the self object.

        :return: a "deep-copy" of the object.
        """
        return deepcopy(self)
    # _end_def_

# _end_class_
