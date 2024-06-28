from src.genome.chromosome import Chromosome
from src.operators.mutation.mutate_operator import MutationOperator
from src.operators.selection.select_operator import SelectionOperator
from src.operators.crossover.crossover_operator import CrossoverOperator

class StandardGA(object):
    """

    """

    # Object variables.
    __slots__ = ("_population", "fitness_func", "_select_op", "_cross_op", "_mutate_op")

    def __int__(self, population: list[Chromosome] = None, func=None, select_op: SelectionOperator = None,
                mutate_op: MutationOperator = None, cross_op: CrossoverOperator = None):

        # Copy the reference of the population.
        self._population = population

        # Make sure the fitness function is indeed callable.
        if not callable(func):
            raise TypeError(f"{self.__class__.__name__}: Fitness function is not callable.")
        else:
            # Get the fitness function.
            self.fitness_func = func
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
        return self._population[index].fitness
    # _end_def_

    def population_fitness(self) -> list:
        """
        Get the fitness of all the population.

        :return: A list with all the fitness values.
        """
        return [p.fitness for p in self._population]
    # _end_def_

    def initialize_population(self):
        """
        Initialize the whole population by calling the random function on each gene separately.

        :return: None.
        """

        # Go through all the chromosome members of the population.
        for i, chromosome in enumerate(self._population):

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

    def run(self, epochs: int = 100):
        """

        :return:
        """

        # Step 1:
        self.initialize_population()

        # Step 2:
        pop_fitness = self.population_fitness()

        # Repeat:
        for i in range(epochs):
            # Step 3:

            # Step 4:

            # Step 5:
            pass
        # _end_for_

    # _end_def_

    # Auxiliary.
    def __call__(self, *args, **kwargs):
        """
        This is only a wrapper of the "run" method.
        """
        return self.run(*args, **kwargs)
    # _end_def_

# _end_class_
