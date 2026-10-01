""" Multipoint crossover module. """
# Custom code imports.
from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offspring)


class MultiPointCrossover(CrossoverOperator):
    """
    Description:

        Multipoint crossover creates two children chromosomes (offsprings),
        by taking two parent chromosomes and cutting them at randomly chosen,
        sites (loci).

        It produces faster mixing, compared with single-point crossover.
    """

    def __init__(self, crossover_probability: float = 0.9, n_points: int = 2) -> None:
        """
        Construct a 'MultiPointCrossover' object with a given probability value.

        :param crossover_probability: (float).

        :param n_points: (int) the number of points to cut the genome.
        """
        # Call the super constructor with the provided initial value.
        super().__init__(crossover_probability=crossover_probability)

        # Make sure number of points are at least 2.
        self._items: int = max(int(n_points), 2)
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Offspring:
        """
        Perform the crossover operation on the two input parent
        chromosomes, using multiple cutting points (num_loci).

        NOTE: the number of loci is held in the '_items' variable.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """
        # If the crossover probability is higher than a uniformly
        # random value and the parents aren't identical apply the
        # changes.
        if self.is_operator_applicable() and (parent1 != parent2):

            # Find the minimum length of the two chromosomes.
            min_length: int = min(len(parent1), len(parent2))

            # Extract the number of cut points.
            num_points: int = self._items

            # Ensure the number of requested cutting points
            # does not exceed the length of the chromosomes.
            if num_points >= min_length:
                raise ValueError(f"{self.__class__.__name__}:"
                                 " Number of requested crossover points"
                                 " exceeds the length of the chromosome.")
            # _end_def_

            # Select randomly the crossover points and sort them.
            loci = sorted(self.rng.choice(range(1, min_length), size=num_points,
                                          replace=False, shuffle=False))

            # Create a list with 'boundary' indices.
            boundaries: list[int] = [0, *loci, min_length]

            # Create the 1st offspring genome list.
            child_1: list[Gene] = parent1.clone_genome()

            # Create the 2nd offspring genome list.
            child_2: list[Gene] = parent2.clone_genome()

            for i in range(len(boundaries) - 1):
                # Swap every second segment.
                if i % 2 == 1:
                    # 'from' index.
                    f: int = boundaries[i]

                    # 'to' index.
                    t: int = boundaries[i + 1]

                    # Swap the genomes.
                    child_1[f:t], child_2[f:t] = child_2[f:t], child_1[f:t]
            # _end_for_

            # Increase the crossover counter.
            self.inc_counter()

            # Return two new offsprings.
            return Chromosome(child_1), Chromosome(child_2)
        # _end_if_

        # Return two cloned offsprings.
        return parent1.clone(), parent2.clone()
    # _end_def_

# _end_class_
