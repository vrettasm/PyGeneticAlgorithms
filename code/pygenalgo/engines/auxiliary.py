from dataclasses import dataclass, field
from pygenalgo.genome.chromosome import Chromosome


# Public interface.
__all__ = ["apply_corrections", "SubPopulation"]


def apply_corrections(input_population: list[Chromosome]) -> int:
    """
    Check the population for invalid genes and correct them by applying
    directly the random method. It is assumed that the random method of
    the Gene is always returning a 'valid' value for the Gene.

    :param input_population: List(Chromosome) the population we want to
    apply corrections (if applicable).

    :return: the total number of corrected genes in the population.
    """

    # Holds the number of the corrected chromosomes.
    corrections_counter = 0

    # Go through all the chromosome members
    # of the input population.
    for chromosome in input_population:

        # Holds the corrected genes.
        genes_corrected = 0

        # Go through every Gene in the chromosome.
        for gene in chromosome:

            # Check for validity.
            if not gene.is_valid or gene.datum is None:

                # Call the gene's random function.
                gene.random()

                # Update the counter.
                genes_corrected += 1
        # _end_for_

        # Check if there were any gene corrections.
        if genes_corrected:
            corrections_counter += genes_corrected
        # _end_if_

    # _end_for_

    # Return the total number of corrected genes.
    return corrections_counter
# _end_def_


@dataclass(init=True, repr=True)
class SubPopulation(object):
    """
    Auxiliary class container used in the IslandModelGA
    to hold all the subpopulations (one on each island).
    """

    # SubPopulation ID.
    pop_id: int

    # List of chromosomes.
    population: list = field(default_factory=list[Chromosome])

    @property
    def id(self) -> int:
        """
        Accessor (getter) of the id parameter.

        :return: the id value.
        """
        return int(self.pop_id)
    # _end_def_

    def __len__(self) -> int:
        """
        Accessor of the total length of the population.

        :return: the length (int) of the population.
        """
        return len(self.population)
    # _end_def_

    def __getitem__(self, index: int):
        """
        Returns the reference of the Chromosome at position index.

        :param index: (int) position of chromosome to return.
        """
        return self.population[index]
    # _end_def_

    def __setitem__(self, index: int, item: Chromosome):
        """
        Sets the input Chromosome in the index position
        inside the (sub) population.

        :param index: (int) position in the population.

        :param item: Chromosome to attach to the new position.
        """
        self.population[index] = item
    # _end_def_

    def __contains__(self, item: Chromosome) -> bool:
        """
        Check for membership.

        :param item: an input Chromosome that we want to check
        if it exists in general population.

        :return: true if the 'item' belongs in the population.
        """
        return item in self.population
    # _end_if_

# _end_class_
