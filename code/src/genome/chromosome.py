from gene import Gene
from dataclasses import dataclass, field

@dataclass(init=True, repr=True)
class Chromosome(object):

    # Define the genome as a list of genes.
    genome: list[Gene] = field(default_factory=list)

    def is_valid(self):
        """
        Checks the validity of the whole chromosome, by
        calling individually all genes is_valid method.

        Additionally, it "double-checks" that all entries
        in the genome are of type 'Gene'.

        :return: True if ALL genes are valid, else False.
        """
        return all(isinstance(x, Gene) and x.is_valid
                   for x in self.genome)
    # _end_def_

# _end_class_
