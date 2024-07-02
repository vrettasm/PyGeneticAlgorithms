import time
import numpy as np
from typing import Callable
from src.genome.chromosome import Chromosome
from src.operators.mutation.mutate_operator import MutationOperator
from src.operators.selection.select_operator import SelectionOperator
from src.operators.crossover.crossover_operator import CrossoverOperator

# Public interface.
__all__ = ["StandardGA", "apply_corrections"]


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


class StandardGA(object):
    """

    """

    # Make a random number generator.
    rng_GA = np.random.default_rng()

    # Object variables.
    __slots__ = ("population", "fitness_func", "_select_op", "_cross_op", "_mutate_op")

    def __init__(self, initial_pop: list[Chromosome] = None, fit_func: Callable = None,
                 select_op: SelectionOperator = None, mutate_op: MutationOperator = None,
                 cross_op: CrossoverOperator = None):
        """

        :param initial_pop:

        :param fit_func:

        :param select_op:

        :param mutate_op:

        :param cross_op:

        :return:
        """

        # Copy the reference of the population.
        self.population = initial_pop.copy()

        # Make sure the fitness function is indeed callable.
        if not callable(fit_func):
            raise TypeError(f"{self.__class__.__name__}: Fitness function is not callable.")
        else:
            # Get the fitness function.
            self.fitness_func = fit_func
        # _end_if_

        # Get Selection Operator.
        self._select_op = select_op

        # Get Mutation Operator.
        self._mutate_op = mutate_op

        # Get Crossover Operator.
        self._cross_op = cross_op

    # _end_def_

    def individual_fitness(self, index: int) -> float:
        """
        Get the fitness value of an individual member of the population.

        :param index: Position of the individual in the population.

        :return: The fitness value (float).
        """
        return self.population[index].fitness

    # _end_def_

    def population_fitness(self) -> list[float]:
        """
        Get the fitness of all the population.

        :return: A list with all the fitness values.
        """
        return [p.fitness for p in self.population]

    # _end_def_

    def evaluate_fitness(self, input_population: list[Chromosome]):
        """
        Evaluate all the chromosomes of the input list with the custom
        fitness function.

        :param input_population: (list)

        :return: None.
        """
        # Get a local copy of the fitness function.
        fit_func = self.fitness_func

        # Evaluate all the individuals.
        for p in input_population:
            p.fitness = fit_func(p)
        # _end_for_

    # _end_def_

    def initialize_population(self):
        """
        Initialize the whole population by calling the random function on each gene separately.

        :return: None.
        """

        # Go through all the chromosome members of the population.
        for i, chromosome in enumerate(self.population):

            # Go through every Gene in the chromosome.
            for gene in chromosome:

                # Call the gene's random function.
                gene.random()
            # _end_for_

            # Sanity Check:
            # The genome of the current chromosome should still be valid.
            if not chromosome.is_genome_valid():
                raise RuntimeError(f"{self.__class__.__name__}: Chromosome {i} is invalid.")
            # _end_if_

        # _end_for_

    # _end_def_

    def best_chromosome(self):
        """
        Auxiliary method.

        :return: Return the chromosome with the highest fitness.
        """
        # Return the chromosome with the highest fitness.
        return max(self.population, key=lambda c: c.fitness)

    # _end_def_

    def run(self, epochs: int = 100, elitism: bool = True, correction: bool = False, f_tol: float = 1.0e-8):
        """

        :param epochs:

        :param elitism:

        :param correction:

        :param f_tol:

        :return: None.
        """

        # Get the size of the population.
        N = len(self.population)

        # Step 1:
        self.initialize_population()

        # Step 2:
        self.evaluate_fitness(self.population)

        # Get the average fitness before optimisation.
        avg_fitness_0 = np.mean([p.fitness for p in self.population])

        # Display an information message.
        print(f"Initial Avg. Fitness = {avg_fitness_0:.4f}")

        # Initial time instant.
        time_t0 = time.perf_counter()

        # Repeat 'epoch' times.
        for i in range(epochs):

            # Step 3: SELECT: the parents.
            # This will create a NEW copy of the population.
            population_i = self._select_op(self.population)

            # Step 4: CROSSOVER: to produce offsprings.
            for j in range(0, N - 1, 2):
                # Replace directly the OLD parents with the NEW offsprings.
                population_i[j], population_i[j + 1] = self._cross_op(population_i[j],
                                                                      population_i[j + 1])
            # _end_for_

            # Step 5: MUTATE: in place the offsprings.
            for p in population_i:
                self._mutate_op(p)
            # _end_for_

            # Check if 'corrections' are enabled.
            if correction:
                # Apply the function.
                total_corrections = apply_corrections(population_i)

                # Print only if there were corrections,
                # to avoid cluttering the screen.
                if total_corrections:
                    print(f"> {total_corrections} correction(s) took place at epoch: {i}.")
                # _end_if_

            # _end_if_

            # Check if 'elitism' is enabled.
            if elitism:

                # Find the individual chromosome with the highest fitness value
                # (from the old population).
                best_chromosome = max(self.population, key=lambda c: c.fitness)

                # Select randomly a position.
                locus = self.rng_GA.integers(0, N)

                # Replace the chromosome with the previous best.
                population_i[locus] = best_chromosome

            # _end_if_

            # Step 6: Evaluate population.
            self.evaluate_fitness(population_i)

            # Calculate the (new) average fitness.
            avg_fitness_i = np.mean([p.fitness for p in population_i])

            # Display an information message.
            print(f"Epoch: {i + 1} -> Avg. Fitness = {avg_fitness_i:.4f}")

            # Step 7: Check for convergence.
            if np.fabs(avg_fitness_i - avg_fitness_0) <= f_tol:

                # Display a warning message.
                print(f"{self.__class__.__name__} finished in {i + 1} iterations.")

                # Exit from the loop.
                break
            # _end_if_

            # Update the average value for the next iteration.
            avg_fitness_0 = avg_fitness_i

            # Update the old population with the new chromosomes.
            self.population = population_i.copy()
        # _end_for_

        # Final time instant.
        time_tf = time.perf_counter()

        # Print final duration in seconds.
        print(f"Elapsed time: {(time_tf - time_t0):.3f} seconds.", end='\n')

    # _end_def_

    # Auxiliary.
    def __call__(self, *args, **kwargs):
        """
        This is only a wrapper of the "run" method.
        """
        return self.run(*args, **kwargs)
    # _end_def_

# _end_class_
