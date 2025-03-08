from copy import deepcopy
from numpy import nan as np_nan
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
    _fitness: float = np_nan

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

    def values(self) -> list:
        """
        Returns the gene values of the chromosome
        as list.

        :return: the list values of the genome.
        """
        return [gene.value for gene in self._genome]
    # _end_def_

    def __eq__(self, other) -> bool:
        """
        Compares the genome of self, with the other chromosome
        and returns True if they are identical otherwise False.

        :param other: chromosome to compare.

        :return: True if the genomes are identical else False.
        """

        # Make sure both objects are of
        # the same type 'Chromosome'.
        if isinstance(other, Chromosome):

            # Compare directly the genomes of both objects.
            return tuple(self._genome) == tuple(other.genome)
        # _end_if_
        return False
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

        # Deepcopy ONLY the genome because
        # it is a (mutable) list of Genes.
        setattr(new_object, "_genome",
                deepcopy(self._genome, memo))

        # Simply copy the fitness value.
        setattr(new_object, "_fitness", self._fitness)

        # Simply copy the boolean flag.
        setattr(new_object, "_valid", self._valid)

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
