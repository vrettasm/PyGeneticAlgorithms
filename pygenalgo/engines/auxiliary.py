from typing import Callable
from dataclasses import dataclass, field
from pygenalgo.genome.chromosome import Chromosome


# Public interface.
__all__ = ["avg_hamming_dist", "apply_corrections", "SubPopulation"]


def avg_hamming_dist(input_population: list[Chromosome]) -> float:
    """
    Computes the average Hamming distance of a population. We use
    this to measure the similarity in the population of chromosomes.

    :param input_population: List(Chromosome) the population we want
    to compute the average Hamming distance.

    :return: (float) the total number of differences, in the genes,
    divided by the total number of genes compared.
    """

    # Initialize the counters.
    total_diffs, total_genes = 0, 0

    # Iterate through all the population.
    for i, p1 in enumerate(input_population[:-1]):

        # Get the size of the chromosome. It is
        # assumed that all chromosomes have the
        # same size.
        N = len(p1)

        # Compare the i-th chromosome with the rest of the population.
        # NOTE: Since the distances are symmetrical we don't check the
        # same pair of chromosomes twice.
        for p2 in input_population[i+1:]:

            # Get the total number of different genes.
            total_diffs += p1.hamming_distance(p2)

            # We add the number of genes we test.
            total_genes += N
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
            chromosome.fitness = fit_func(chromosome)
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
    # _end_if_

# _end_class_
