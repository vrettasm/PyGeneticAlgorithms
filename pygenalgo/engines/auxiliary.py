from typing import Callable
from dataclasses import dataclass, field
from pygenalgo.genome.chromosome import Chromosome

# Public interface.
__all__ = ["average_hamming_distance", "pareto_front",
           "apply_corrections", "SubPopulation"]

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

    # Initialize the counter.
    total_diffs = 0

    # Get the number of the chromosomes.
    n_chromosomes = len(population)

    # Get the size of the chromosome. It is
    # assumed that all chromosomes have the
    # same size.
    n_genes = len(population[0])

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
    # _end_for_

    # Compute the total number of counted genes.
    total_genes = (n_chromosomes * (n_chromosomes - 1) / 2.0)

    # Check for normalization.
    if normal and n_genes != 0:
        total_genes *= n_genes
    # _end_if_

    # Return the averaged value.
    return float(total_diffs / total_genes)
# _end_def_

def apply_corrections(input_population: list[Chromosome],
                      fit_func: Callable = None) -> (int, int):
    """
    Check the population  for invalid genes and correct them by applying directly
    the random method. It is assumed that the random method of the Gene is always
    returning a 'valid' value for the Gene. After that we need to re-evaluate the
    chromosome to update its fitness.

    :param input_population: List(Chromosome) the population we want to apply
    corrections (if applicable).

    :param fit_func: callable fitness function.

    :return: total number of corrected genes and the additional function evaluations.
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
        if corrected_genes:

            # Update the total corrections counter.
            corrections_counter += corrected_genes

            # Re-evaluate the fitness of the chromosome.
            chromosome.fitness, _ = fit_func(chromosome)

            # Increase counter by one.
            f_eval_counter += 1
        # _end_if_

    # _end_for_

    # Return the total number of corrected genes along
    # with the count of additional function evaluations.
    return corrections_counter, f_eval_counter
# _end_def_

def pareto_front(points: list) -> list:
    """
    Simple function that calculates the pareto (optimal)
    front points from a given input points list.

    NOTE: The function is working directly with lists,
    even though it can be optimized using numpy arrays.

    :param points: list of points [(fx1, fx2, ..., fxn),
                                   (fy1, fy2, ..., fyn),
                                   ....................,
                                   (fk1, fk2, ..., fkn)]

    :return: List of points that lie on the pareto front.
    """
    # Create a list that will hold
    # ONLY the Pareto front points.
    pareto_points = []

    # Iterate through every point in the list.
    for i, point_i in enumerate(points):

        # Set the pareto optimal flag value to True.
        is_pareto_optimal = True

        # Compare it against every other point.
        for j, point_j in enumerate(points):

            # Check if "dominance" condition is satisfied.
            if i != j and all(p >= q for p, q in zip(point_i, point_j,
                                                     strict=True)):
                # We swap the flag value.
                is_pareto_optimal = False

                # Break the internal loop and
                # continue to the next point.
                break
            # _end_if_

        # _end_for_

        # If we get here and the flag hasn't changed
        # it means that 'point_i' is on the frontier.
        if is_pareto_optimal:
            pareto_points.append(point_i)
        # _end_if_
    # _end_for_

    # Return the points.
    return pareto_points
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
