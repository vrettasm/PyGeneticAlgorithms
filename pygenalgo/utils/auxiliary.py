from typing import Callable
from functools import lru_cache
from dataclasses import dataclass, field
from pygenalgo.genome.chromosome import Chromosome

# Public interface.
__all__ = ["average_hamming_distance", "unique_pairs",
           "apply_corrections", "SubPopulation"]

@lru_cache(maxsize=64)
def unique_pairs(n_size: int) -> int:
    """
    Computes the number of unique pairs among 'n_size'
    distinct numbers by using the combination formula
    for choosing 2 items from a set.

    :param n_size: the number of 'distinct' numbers.

    :return: the 'n choose 2', or C(n, 2).
    """

    # Sanity check #1.
    if not isinstance(n_size, int):
        raise TypeError("'n_size' must be an integer number.")
    # _end_if_

    # Sanity check #2.
    if n_size <= 1:
        raise ValueError("'n_size' must be more than one.")
    # _end_if_

    return int(0.5 * n_size * (n_size - 1))
# _end_def_

def average_hamming_distance(population: list[Chromosome],
                             normal: bool = True) -> float:
    """
    Computes the average Hamming distance of a population. We use this
    to measure the similarity in the whole population of chromosomes.

    :param population: List(Chromosome) the population we want to compute
    the average Hamming distance.

    :param normal: (bool) flag that requires the return of the normalized
    average distance.

    :return: (float) the total number of differences, in the genes,
    divided by the total number of genes compared.
    """

    # Sanity check 1: This should never happen!
    if not population:
        raise RuntimeError("Population list is empty!")
    # _end_if_

    # Get the number of the chromosomes.
    n_chromosomes = len(population)

    # Get the size of the chromosome. It is
    # assumed that all chromosomes have the
    # same size.
    n_genes = len(population[0])

    # Sanity check 2: This should never happen!
    if n_genes == 0:
        raise RuntimeError("The number of genes in the Chromosomes is zero!")
    # _end_if_

    # Initialize the counter.
    total_diffs = 0

    # Create a HashMap with the chromosomes' genome.
    genome_map = {i: item.genome for i, item in enumerate(population)}

    # Here we sum the Hamming distances for all unique
    # pairs of chromosomes (to avoid double counting).
    for i in range(n_chromosomes):
        for j in range(i + 1, n_chromosomes):
            total_diffs += [k != l for k, l in zip(genome_map[i],
                                                   genome_map[j])].count(True)
    # _end_for_

    # Compute the averaged distance, using the total
    # number of 'unique pairs'.
    dist_avg = total_diffs / unique_pairs(n_chromosomes)

    # Return the averaged (or the normalized) value.
    return dist_avg / n_genes if normal else dist_avg
# _end_def_

def apply_corrections(input_population: list[Chromosome],
                      fit_func: Callable = None) -> tuple[int, int]:
    """
    Check the population  for invalid genes and correct them by applying directly
    the random method. It is assumed that the random method of the Gene is always
    returning a 'valid' value for the Gene. After that we need to re-evaluate the
    chromosome to update its fitness.

    :param input_population: List(Chromosome) the population we want to apply
    corrections (if applicable).

    :param fit_func: callable fitness function.

    :return: the number of corrected genes and the additional function evaluations.
    """

    # Holds the number of the corrected chromosomes.
    corrections_counter = 0

    # Counts the additional function evaluations.
    f_eval_counter = 0

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
        if corrected_genes > 0:

            # Update the total corrections counter.
            corrections_counter += corrected_genes

            # Re-evaluate the fitness of the chromosome.
            results = fit_func(chromosome)

            # Assign the new fitness value.
            chromosome.fitness = results["f_value"]

            # Increase counter by one.
            f_eval_counter += 1
        # _end_if_

    # _end_for_

    # Return the total number of corrected genes along
    # with the count of additional function evaluations.
    return corrections_counter, f_eval_counter
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
