from math import fabs
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
                 lower_val: float = None, upper_val: float = None) -> None:
        """
        Construct a 'BlendCrossover' object with a given probability value.

        :param crossover_probability: (float).

        :param p_alpha: (float).

        :param lower_val: (float) lower limit value for the gene.

        :param upper_val: (float) upper limit value for the gene.
        """
        # Call the super constructor with the provided
        # probability value.
        super().__init__(crossover_probability)

        # Ensure p_alpha parameter is float.
        p_alpha = clamp(float(p_alpha), 0.0, 1.0)

        # Ensure lower_val parameter is float.
        lower_val = float(lower_val)

        # Ensure upper_val parameter is float.
        upper_val = float(upper_val)

        # Ensure the order is correct.
        if upper_val < lower_val:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"The limit values are incorrect.")
        # _end_if_

        # Assign variables to the _items placeholder.
        self._items = [p_alpha, lower_val, upper_val]
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
            number_of_genes = len(parent1)

            # Preallocate 1st genome.
            genome_1: list = [None] * number_of_genes

            # Preallocate 2nd genome.
            genome_2: list = [None] * number_of_genes

            # Generate uniform random numbers (floats)
            # in the half-open interval [0.0, 1.0).
            rv_uniform = self.rng.random(size=(number_of_genes, 2))

            # Set the genes accordingly.
            for i, (r_val, gene_1, gene_2) in enumerate(zip(rv_uniform,
                                                            parent1.genome,
                                                            parent2.genome)):
                # Extract the gene values once.
                g1, g2 = gene_1.value, gene_2.value

                # Get the offset by scaling the distance
                # between the two gene values with alpha.
                offset_distance = p_alpha * fabs(g1 - g2)

                # Get the min / max values.
                if g1 < g2:
                    min_value, max_value = g1, g2
                else:
                    min_value, max_value = g2, g1
                # _end_if_

                # Compute the lower and upper
                # limits by removing / adding
                # the offset distance.
                min_value -= offset_distance
                max_value += offset_distance

                # Create two new gene values.
                new_value_1, new_value_2 = lower_lim + (upper_lim - lower_lim) * r_val

                # Ensure the new values are within limits.
                new_value_1 = clamp(new_value_1, xl, xu)
                new_value_2 = clamp(new_value_2, xl, xu)

                # Update the genome of the new offsprings with new Genes.
                genome_1[i] = Gene(datum=new_value_1, func=gene_1.func)
                genome_2[i] = Gene(datum=new_value_2, func=gene_2.func)
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
