""" Cycle Crossover (CX) operator module. """
# Custom code imports.
from pygenalgo.genome.chromosome import Chromosome
from pygenalgo.operators.crossover.crossover_operator import (CrossoverOperator, Offsprings)


class CycleCrossover(CrossoverOperator):
    """
    Description:

        Cycle Crossover (CX) creates two children chromosomes by identifying
        cycles of identical positions between parents. This ensures that every
        gene in the offspring maintains an absolute position inherited from
        one of the two parents.

        It is highly effective for combinatorial and permutation problems.
    """

    def __init__(self, crossover_probability: float = 0.9) -> None:
        """
        Construct a 'CycleCrossover' object with a given probability value.

        :param crossover_probability: (float).
        """
        super().__init__(crossover_probability=crossover_probability)
    # _end_def_

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Offsprings:
        """
        Perform the cycle crossover operation on the two input parent chromosomes.

        :param parent1: (Chromosome).

        :param parent2: (Chromosome).

        :return: child1 and child2 (as Chromosomes).
        """
        # If the crossover probability is higher than a uniformly
        # random value and the parents aren't identical apply the
        # changes.
        if self.is_operator_applicable() and (parent1 != parent2):

            # Get the size of the chromosomes.
            number_of_genes: int = len(parent1)

            # Initialize empty offspring arrays.
            child_1: list = number_of_genes * [None]
            child_2: list = number_of_genes * [None]

            # Pre-compute positions for O(1) lookups instead of using .index().
            p1_pos_map = {gene: idx for idx, gene in enumerate(parent1.genome)}

            # Track which indices have been visited/assigned.
            visited: list[bool] = [False] * number_of_genes
            cycle_count: int = 0

            # Find all cycles across the chromosomes.
            for idx in range(number_of_genes):

                # Skip to speed up.
                if visited[idx]:
                    continue

                # Alternating cycles get assigned to different parents.
                # Cycle 0, 2, 4... copies P1 -> C1 and P2 -> C2
                # Cycle 1, 3, 5... copies P2 -> C1 and P1 -> C2
                use_parent1_first: int = (cycle_count % 2 == 0)

                # Auxiliary index.
                current_idx: int = idx

                while not visited[current_idx]:
                    # Update the flag.
                    visited[current_idx] = True

                    if use_parent1_first:
                        child_1[current_idx] = parent1.genome[current_idx].clone()
                        child_2[current_idx] = parent2.genome[current_idx].clone()
                    else:
                        child_1[current_idx] = parent2.genome[current_idx].clone()
                        child_2[current_idx] = parent1.genome[current_idx].clone()

                    # Look up where the gene from parent2
                    # at current_idx lives in parent1.
                    next_gene = parent2.genome[current_idx]
                    current_idx = p1_pos_map[next_gene]
                # _end_while_

                # Increase cycle counter.
                cycle_count += 1
            # _end_for_

            # Increase the crossover counter.
            self.inc_counter()

            # Return two new offsprings.
            return Chromosome(child_1), Chromosome(child_2)

        # Return two cloned offsprings.
        return parent1.clone(), parent2.clone()
    # _end_def_

# _end_class_
