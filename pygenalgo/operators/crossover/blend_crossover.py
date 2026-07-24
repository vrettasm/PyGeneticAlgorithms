from numpy import asarray
from numpy import any as np_any
from numpy.typing import ArrayLike, NDArray

from pygenalgo.genome.gene import Gene
from pygenalgo.utils.utilities import clamp
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import CrossoverOperator


class BlendCrossover(CrossoverOperator):
    """
    Description:

        Blend-a crossover (BLX-a) creates two children chromosomes (offsprings) by
        uniformly picking values that lie  between two points that contain the two
        parents but may extend equally on either side determined by a user specified
        parameter 'a'.

        NB: Used only for real coded genomes.
    """

    def __init__(self, crossover_probability: float = 0.9, p_alpha: float = 0.5,
                 lower_lim: ArrayLike = None, upper_lim: ArrayLike = None) -> None:
        """
        Construct a 'BlendCrossover' object with a given probability value.

        :param crossover_probability: (float).

        :param p_alpha: (float).

        :param lower_lim: (ArrayLike) lower limit values for the genes.

        :param upper_lim: (ArrayLike) upper limit values for the genes.
        """

        # Call the super constructor with the provided initial value.
        super().__init__(crossover_probability=crossover_probability)

        # Check if the lower and upper bounds are set.
        if (lower_lim is None) or (upper_lim is None):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Lower or Upper limits are missing.")

        # Ensure p_alpha parameter is float.
        p_alpha = clamp(float(p_alpha), 0.0, 1.0)

        # Make sure the limits are numpy arrays.
        lower_lim = asarray(lower_lim, dtype=float)
        upper_lim = asarray(upper_lim, dtype=float)

        # Check if there is a size mismatch.
        if lower_lim.size != upper_lim.size:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Lower and Upper limits sizes do not match.")

        # Check if the boundaries are set correctly.
        if np_any(upper_lim <= lower_lim):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Lower and Upper limits are set incorrectly.")

        # Assign variables to the _items placeholder.
        self._items: tuple[float, NDArray, NDArray] = (
            p_alpha, lower_lim, upper_lim
        )
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
        """
        Perform the crossover operation on the two input parent chromosomes.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """
        # If the crossover probability is higher than a uniformly
        # random value and the parents aren't identical apply the
        # changes.
        if (parent1 != parent2) and self.is_operator_applicable():

            # Extract the values from the placeholder variable.
            p_alpha, xl, xu = self._items

            # Get the length of the chromosome.
            # Here we assume that both parents
            # have the same lengths.
            number_of_genes: int = len(parent1)

            # Preallocate 1st child's genome.
            genome_1: list = [None] * number_of_genes

            # Preallocate 2nd child's genome.
            genome_2: list = [None] * number_of_genes

            # Generate uniform random numbers in the [0.0, 1.0).
            random_uniform: NDArray = self.rng.random(size=(number_of_genes, 2))

            # Extract locally the genomes of both parents.
            parent_1: list[Gene] = parent1.genome
            parent_2: list[Gene] = parent2.genome

            # Set the new gene values iteratively.
            for i in range(number_of_genes):

                # Extract the gene values once.
                g1 = parent_1[i].value
                g2 = parent_2[i].value

                # Get the min / max values.
                if g1 < g2:
                    min_value, max_value = g1, g2
                else:
                    min_value, max_value = g2, g1
                # _end_if_

                # Get the offset by scaling the distance
                # between the two gene values with alpha.
                offset_distance = p_alpha * (max_value - min_value)

                # Compute the lower and upper limits by
                # removing / adding the offset distance.
                min_value -= offset_distance
                max_value += offset_distance

                # Extract the two random values.
                rv_1, rv_2 = random_uniform[i]

                # Compute the difference.
                diff = max_value - min_value

                # Create two new gene values.
                new_value_1 = min_value + (diff * rv_1)
                new_value_2 = min_value + (diff * rv_2)

                # Local bounds lookups.
                x_lower = xl[i]
                x_upper = xu[i]

                # Ensure the new values are within limits.
                new_value_1 = min(max(new_value_1, x_lower), x_upper)
                new_value_2 = min(max(new_value_2, x_lower), x_upper)

                # Extract the gene functions.
                gene_1_func = parent_1[i].func
                gene_2_func = parent_2[i].func

                # Update the genome of the new offsprings with new Genes.
                genome_1[i] = Gene(datum=new_value_1, func=gene_1_func)
                genome_2[i] = Gene(datum=new_value_2, func=gene_2_func)
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
