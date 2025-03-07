from typing import Callable
from dataclasses import dataclass, field
from pygenalgo.genome.chromosome import Chromosome

# Public interface.
__all__ = ["hamming_distance", "average_hamming_distance",
           "apply_corrections", "SubPopulation"]


def hamming_distance(item1: Chromosome, item2: Chromosome) -> int:
    """
    Compute the Hamming distance of the "item1" object, with the
    "item2" chromosome. In practise it's the number of positions
    at which the corresponding genes are different.

    :param item1: (Chromosome) to compare the Hamming distance.

    :param item2: (Chromosome) to compare the Hamming distance.

    :return: (int) the number of dissimilarities between the two
    input chromosomes.
    """

    # Check for identical types.
    if type(item1) is type(item2):

        # Quick exit if both objects are the
        # same or equal.
        if item1 is item2 or item1 == item2:
            return 0
        # _end_if_

        # Exit with an error if both objects
        # are not the same length.
        if len(item1) != len(item2):
            raise ValueError(f"Input chromosomes must be of the same length.")
        # _end_if_

        # Compute the dissimilarities in their genomes.
        return [k != l for k, l in zip(item1.genome, item2.genome)].count(True)
    else:
        raise RuntimeError(f"Can't compute Hamming distance in different type objects.")
    # _end_if_

# _end_def_

def average_hamming_distance(population: list[Chromosome]) -> float:
    """
    Computes the average Hamming distance of a population. We use this
    to measure the similarity in the whole population of chromosomes.

    :param population: List(Chromosome) the population we want to compute
    the average Hamming distance.

    :return: (float) the total number of differences, in the genes,
    divided by the total number of genes compared.
    """

    # Initialize the counters.
    total_diffs, total_genes = 0, 0

    # Get the size of the chromosome. It is
    # assumed that all chromosomes have the
    # same size.
    number_of_genes = len(population[0])

    # Iterate through all the population.
    for i, item1 in enumerate(population):

        # Local copy of the 1st genome to
        # avoid recalling in the 2nd loop.
        item1_genome = item1.genome

        # Compare the i-th chromosome with the rest of the population.
        # NOTE: Since the distances are symmetrical we don't check the
        # same pair of chromosomes twice.
        for item2 in population[i+1:]:
            # Get the total number of different genes.
            total_diffs += [k != l for k, l in zip(item1_genome,
                                                   item2.genome)].count(True)
            # We add the number of genes we tested.
            total_genes += number_of_genes
        # _end_for_

    # _end_for_

    # Sanity check.
    if total_genes == 0:
        raise RuntimeError("Average Humming Distance: Total number of Genes is zero.")
    # _end_if_

    return float(total_diffs / total_genes)
# _end_def_

def apply_corrections(input_population: list[Chromosome],
                      fit_func: Callable = None) -> int:
    """
    Check the population  for invalid genes and correct them by applying directly
    the random method. It is assumed that the random method of the Gene is always
    returning a 'valid' value for the Gene. After that, we need to reevaluate the
    chromosome to update its fitness.

    :param input_population: List(Chromosome) the population
    we want to apply corrections (if applicable).

    :param fit_func: callable fitness function.

    :return: the total number of corrected genes in the population.
    """

    # Holds the number of the corrected chromosomes.
    corrections_counter = 0

    # Go through all the chromosomes of the input population.
    for chromosome in input_population:

        # Holds the corrected genes.
        corrected_genes = 0

        # Go through every Gene in the chromosome.
        for gene in chromosome:

            # Check for validity.
            if not gene.is_valid or gene.value is None:

                # Call the gene's random function.
                gene.random()

                # Update the status of the gene.
                gene.is_valid = True

                # Update the counter.
                corrected_genes += 1
            # _end_if_

        # _end_for_

        # Check if there were any gene corrections.
        if corrected_genes:

            # Update the total corrections counter.
            corrections_counter += corrected_genes

            # Re-evaluate the fitness of the chromosome.
            chromosome.fitness, _ = fit_func(chromosome)
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
        return self.pop_id
    # _end_def_

    def __len__(self) -> int:
        """
        Accessor of the total length of the population.

        :return: the length (int) of the population.
        """
        return len(self.population)
    # _end_def_

    def __getitem__(self, index: int) -> Chromosome:
        """
        Returns the reference of the Chromosome at position index.

        :param index: (int) position of chromosome to return.
        """
        return self.population[index]
    # _end_def_

    def __setitem__(self, index: int, item: Chromosome) -> None:
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
    # _end_def_

# _end_class_
