from math import fabs
from pygenalgo.genome.gene import Gene
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

    def __init__(self, crossover_probability: float = 0.9, p_alpha: float = 0.5):
        """
        Construct a 'BlendCrossover' object with a given probability value.

        :param crossover_probability: (float).

        :param p_alpha: (float).
        """

        # Call the super constructor with the provided
        # probability value.
        super().__init__(crossover_probability)

        # Make sure 'alpha' parameter is in [0, 1].
        self._items = max(0.0, min(float(p_alpha), 1.0))
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome):
        """
        Perform the crossover operation on the two input parent chromosomes.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """

        # If the crossover probability is higher than a uniformly
        # random value and the parents aren't identical apply the
        # changes.
        if (parent1 is not parent2) and (parent1 != parent2) and \
                self.is_operator_applicable():

            # Get the length of the chromosome.
            number_of_genes = len(parent1)

            # Preallocate 1st genome.
            genome_1: list[Gene] = [None] * number_of_genes

            # Preallocate 2nd genome.
            genome_2: list[Gene] = [None] * number_of_genes

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
                offset_distance = self._items * fabs(g1 - g2)

                # Get the min / max values.
                if g1 < g2:
                    min_value, max_value = g1, g2
                else:
                    min_value, max_value = g2, g1
                # _end_if_

                # Compute the 'lower' / 'upper' limits.
                lower_lim = min_value - offset_distance
                upper_lim = max_value + offset_distance

                # Create two new gene values.
                new_value_1, new_value_2 = lower_lim + (upper_lim - lower_lim) * r_val

                # Update the genome of the new offsprings with two new Genes.
                genome_1[i] = Gene(datum=new_value_1, func=gene_1.func)
                genome_2[i] = Gene(datum=new_value_2, func=gene_2.func)
            # _end_for_

            # Create the two NEW offsprings.
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
