from copy import deepcopy
from pygenalgo.genome.gene import Gene

# Public interface.
__all__ = ["Chromosome"]


class Chromosome(object):
    """
    Description:

        Implements a dataclass for the Chromosome entity. This class is responsible
        for holding the individual solution(s), of the optimization problem, during
        the evolution process.
    """

    # Object variables.
    __slots__ = ("_genome", "_fitness", "_valid")

    def __init__(self, genome: list[Gene], fitness: float = float("NaN"),
                 valid: bool = True) -> None:
        """
        Initialize a Chromosome object.

        :param genome: a list of genes. This list will encode a single
                       solution to the problem

        :param fitness: the fitness of the chromosome.

        :param valid: whether the chromosome is valid.
        """

        # Copy the genome.
        self._genome = genome

        # Get the initial fitness value.
        self._fitness = fitness

        # Set the bool flag.
        self._valid = valid
    # _end_def_

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
        # Check for the correct type.
        if not isinstance(new_value, bool):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Validity flag should be bool: {type(new_value)}.")
        # _end_if_

        # Update the flag value.
        self._valid = new_value
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
        # Check for the correct type.
        if not isinstance(new_value, (int, float)):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Fitness should be float: {type(new_value)}.")
        # _end_if_

        # Update the fitness value.
        self._fitness = float(new_value)
    # _end_def_

    @property
    def genome(self) -> list[Gene]:
        """
        Accessor of the genome list of the chromosome.

        :return: the list (of Genes) of the chromosome.
        """
        return self._genome
    # _end_def_

    def invalidate_fitness(self) -> None:
        """
        Invalidates the fitness of the chromosome
        by setting the value to NaN. This is used
        during the evolution process (mutation).

        :return: None.
        """
        self._fitness = float("NaN")
    # _end_def_

    def has_valid_genome(self) -> bool:
        """
        Checks the validity of the whole chromosome, by
        calling individually all genes is_valid method.

        :return: True if ALL genes are valid, else False.
        """
        return all(x.is_valid for x in self._genome)
    # _end_def_

    def values(self) -> list:
        """
        Returns the gene values of the chromosome
        as list.

        :return: the list values of the genome.
        """
        return [gene.value for gene in self._genome]
    # _end_def_

    def hamming_distance(self, other) -> int:
        """
        Compute the Hamming distance of the "self" object, with the
        "other" chromosome. In practice, it's the number of positions
        at which the corresponding genes are different.

        :param other: (Chromosome) to compare the Hamming distance.

        :return: (int) the number of dissimilarities between the two
                 input chromosomes.
        """
        # Make sure both objects are of the same type Chromosome.
        if not isinstance(other, Chromosome):
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Can't compute Hamming distance in different type objects.")
        # _end_if_

        # Quick exit if both objects are the same or equal.
        if self is other or self == other:
            return 0
        # _end_if_

        # Compute the dissimilarities in their genomes.
        return [k != l for k, l in zip(self._genome, other.genome,
                                       strict=True)].count(True)
    # _end_def_

    def clone(self) -> "Chromosome":
        """
        Makes a duplicate of the self object
        by deep-coping only the genome field.

        :return: a "deep-copy" of the object.
        """
        return Chromosome(deepcopy(self._genome), self._fitness, self._valid)
    # _end_def_

    def __eq__(self, other) -> bool:
        """
        Compares the genome of self, with the other chromosome
        and returns True if they are identical otherwise False.

        :param other: chromosome to compare.

        :return: True if the genomes are identical else False.
        """
        # Check if they are the same instance.
        if self is other:
            return True
        # _end_if_

        # Make sure both items are Chromosomes.
        if not isinstance(other, Chromosome):
            return NotImplemented
        # _end_if_

        # Compare directly the two genomes.
        return self._genome == other.genome
    # _end_def_

    def __hash__(self) -> int:
        """
        Auxiliary method to hash the Chromosome object.

        :return: the hash value of the genome.
        """
        return hash(tuple(self._genome))
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

    def __deepcopy__(self, memo: dict) -> "Chromosome":
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

# _end_class_
