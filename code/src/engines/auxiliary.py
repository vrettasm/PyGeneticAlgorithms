import numpy as np
from src.genome.chromosome import Chromosome


# Public interface.
__all__ = ["apply_corrections"]

def apply_corrections(input_population: list[Chromosome]) -> int:
    """
    Check the population for invalid genes and correct them by applying
    directly the random method. It is assumed that the random method of
    the Gene is always returning a 'valid' value for the Gene.

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
