from pygenalgo.genome.gene import Gene
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator


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

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
        """
        Perform the crossover operation on the two input parent
        chromosomes, using multiple cutting points (num_loci).

        NOTE: the number of loci is held in the '_items' variable.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """

        # Find the minimum length of the two chromosomes.
        min_length: int = min(len(parent1), len(parent2))

        # Extract the number of cut points.
        num_points: int = self._items

        # Ensure the number of requested cutting points
        # do not exceed the length of the chromosomes.
        if num_points >= min_length:
            raise ValueError(f"{self.__class__.__name__}:"
                             " Number of requested crossover points"
                             " exceeds the length of the chromosome.")
        # _end_def_

        # If the crossover probability is higher than a uniformly
        # random value and the parents aren't identical apply the
        # changes.
        if (parent1 != parent2) and self.is_operator_applicable():

            # Find the minimum length of the two chromosomes.
            min_length: int = min(len(parent1), len(parent2))

            # Extract the number of cut points.
            num_points: int = self._items

            # Ensure the number of requested cutting points
            # do not exceed the length of the chromosomes.
            if num_points >= min_length:
                raise ValueError(f"{self.__class__.__name__}:"
                                 " Number of requested crossover points"
                                 " exceeds the length of the chromosome.")
            # _end_def_

            # Select randomly the crossover points and sort them.
            loci = sorted(self.rng.choice(min_length, size=num_points,
                                          replace=False, shuffle=False))

            # Create the 1st offspring genome list.
            genome_1: list[Gene] = [
                gene.clone() for gene in parent1.genome
            ]

            # Create the 2nd offspring genome list.
            genome_2: list[Gene] = [
                gene.clone() for gene in parent2.genome
            ]

            # Initialize a set of hyperparameters.
            reset_flag, upper_lim, j = True, loci[0], 0

            # Scan the genomes up to min_length.
            for i in range(min_length):

                # Once we surpass the upper limit (in loci)
                # we reset the  flag value to allow changes
                # to take place within that range.
                if i >= upper_lim:

                    # Swap the reset flag.
                    reset_flag = not reset_flag

                    # Increase the index of the loci.
                    j += 1

                    # We make sure the upper limit value does not exceed
                    # the number of genes. Also, this avoids the out of
                    # bound IndexError.
                    upper_lim = loci[j] if j < num_points else min_length
                # _end_if_

                # Check the flag value.
                if not reset_flag:
                    genome_1[i], genome_2[i] = genome_2[i], genome_1[i]
                # _end_if_

            # _end_for_

            # Create two NEW offsprings.
            child1 = Chromosome(genome_1)
            child2 = Chromosome(genome_2)

            # Increase the crossover counter.
            self.inc_counter()
        else:
            # Each child points to a clone of a single parent.
            child1 = parent1.clone()
            child2 = parent2.clone()
        # _end_if_

        # Return the two offsprings.
        return child1, child2
    # _end_def_

# _end_class_
