from copy import deepcopy
from code.src.genome.gene import Gene
from dataclasses import dataclass, field

@dataclass(init=True, repr=True)
class Chromosome(object):

    # Define the genome as a list of genes.
    genome: list[Gene] = field(default_factory=list)

    def is_valid(self) -> bool:
        """
        Checks the validity of the whole chromosome, by
        calling individually all genes is_valid method.

        In addition, it "double-checks" that all entries
        in the genome are of type 'Gene'.

        :return: True if ALL genes are valid, else False.
        """
        return all(isinstance(x, Gene) and x.is_valid
                   for x in self.genome)
    # _end_def_

    def __len__(self) -> int:
        """
        Accessor of the total length of the genome.

        :return: the length (int) of the genome.
        """
        return len(self.genome)
    # _end_def_

    def __getitem__(self, index: int):
        return self.genome[index]
    # _end_def_

    def __setitem__(self, index: int, item: Gene):
        self.genome[index] = item
    # _end_def_

    def __contains__(self, item: Gene) -> bool:
        """
        Check for membership.

        :param item: an input Gene that we want to check.

        :return: true if the 'item' belongs in the genome.
        """
        return item in self.genome
    # _end_if_

    def make_deepcopy(self):
        """
        Makes a duplicate of the self object.

        :return: a 'deep-copy' of the object.
        """
        return deepcopy(self)
    # _end_def_

# _end_class_
