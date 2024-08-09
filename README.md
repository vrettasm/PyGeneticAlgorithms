# PyGenAlgo:  A simple toolkit for genetic algorithms.

"Genetic Algorithms [(GA)](https://en.wikipedia.org/wiki/Genetic_algorithm), are meta heuristic algorithms
inspired by the process of natural selection and belong to a larger class of evolutionary algorithms (EA)."

-- (From Wikipedia, the free encyclopedia)

This repository implements a genetic algorithm (GA) in Python3 programming language, using only **Numpy** and **Joblib**
as additional libraries. The basic approach offers a "StandardGA" class, where the whole population of chromosomes is
replaced by a new one at the end of each iteration (or epoch). More recently, the new "IslandModelGA" class was added
that offers a new genetic operator (MigrationOperator), that allows for periodic migration of the best individuals,
among the different island populations.
  
**NOTE**:
For computationally expensive fitness functions the StandardGA class provides the option of parallel evaluation
(of the individual chromosomes), by setting in the method run(..., parallel=True). However, for fast fitness
functions this will actually cause the algorithm to execute slower (due to the time required to open and close the
parallel pool). So the default setting here is "parallel=False". Regarding the IslandModelGA, this is running in
parallel mode by definition.

  > **NEWS**:
  > Two new genetic operators have been added (**SuperCrossover** and **SuperMutator**). At each iteration they call
  > randomly one of the other crossover / mutation operators. This way we can add a different mixing of the operators
  > that may affect the evolution. By default, the selection probabilities of all operators are set equally, but this
  > constrained can be relaxed in the future allowing each operator to have a different probability of being selected.
  >

The current implementation offers a variety of genetic operators including:

- **Selection operators**:
  - [Linear Rank Selector](code/pygenalgo/operators/selection/linear_rank_selector.py)
  - [Random Selector](code/pygenalgo/operators/selection/random_selector.py)
  - [Roulette Wheel Selector](code/pygenalgo/operators/selection/roulette_wheel_selector.py)
  - [Stochastic Universal Selector](code/pygenalgo/operators/selection/stochastic_universal_selector.py)
  - [Tournament Selector](code/pygenalgo/operators/selection/tournament_selector.py)
  - [Truncation Selector](code/pygenalgo/operators/selection/truncation_selector.py)

- **Crossover operators**:
  - [Single-Point Crossover](code/pygenalgo/operators/crossover/single_point_crossover.py)
  - [Multi-Point Crossover](code/pygenalgo/operators/crossover/mutli_point_crossover.py)
  - [Uniform Crossover](code/pygenalgo/operators/crossover/uniform_crossover.py)
  - [Super Crossover](code/pygenalgo/operators/crossover/super_crossover.py)

- **Mutation operators**:
  - [Random Mutator](code/pygenalgo/operators/mutation/random_mutator.py)
  - [Shuffle Mutator](code/pygenalgo/operators/mutation/shuffle_mutator.py)
  - [Swap Mutator](code/pygenalgo/operators/mutation/swap_mutator.py)
  - [Super Mutator](code/pygenalgo/operators/mutation/super_mutator.py)

- **Migration operators**
  - [Clockwise Migrator](code/pygenalgo/operators/migration/clockwise_migration.py)

Note that incorporating additional genetic operators is easily facilitated by inheriting from the base classes:
- [SelectionOperator](code/pygenalgo/operators/selection/select_operator.py)
- [CrossoverOperator](code/pygenalgo/operators/crossover/crossover_operator.py)
- [MutationOperator](code/pygenalgo/operators/mutation/mutate_operator.py)
- [MigrationOperator](code/pygenalgo/operators/migration/migration_operator.py)

and implementing the basic interface as described therein. In the examples that follow I show how one can use this code
to run a GA for optimization problems (maximization/minimization) with and without constraints. The project is ongoing
so new things might come along the way.

### Installation

There are two options to install the software.

The easiest way is to visit the GitHub web-page of the project and simply download the source code in
[zip](https://github.com/vrettasm/PyGeneticAlgorithms/archive/refs/heads/master.zip) format. This option does not
require a prior installation of git on the computer.

Alternatively one can clone the project directly using git as follows:

    git clone https://github.com/vrettasm/PyGeneticAlgorithms.git

### Required packages

The recommended version is Python 3.10 (and above). To simplify the required packages just use:

    pip install -r requirements.txt

### Fitness function

The most important thing the user has to do is to define the "fitness function". A template is provided here,
in addition to the examples below.
```python
from pygenalgo.genome.chromosome import Chromosome

# Fitness function <template>.
def fitness_func(individual: Chromosome, f_min: bool = False):
    """
    This is how a fitness function should look like. The whole
    evaluation should be implemented (or wrapped around) this
    function.
    
    :param individual: Individual chromosome to be evaluated.
    
    :param f_min: Boolean flag indicating whether we are dealing
    with a minimization or maximization problem.
    """
    
    # CODE TO IMPLEMENT.
    
    # Assign the estimated value.
    f_val = ...
    
    # If we want minimization we return the negative.
    return -f_val if f_min else f_val
# _end_def_
```
Once the fitness function is defined correctly the next steps are straightforward as described in the examples.

### Examples

Some optimization examples on how to use this algorithm:

1. StandardGA
   1. [Minimization](examples/sphere.ipynb)
   2. [Maximization](examples/rastrigin.ipynb)
   3. [Minimization with constraints](examples/rosenbrock_on_a_disk.ipynb)
   4. [Multi-objective with constraints](examples/binh_and_korn_multiobjective.ipynb)
2. IslandModelGA
   1. [Sphere Function](examples/sphere_in_parallel.ipynb)
   2. [Easom Function](examples/easom_in_parallel.ipynb)
3. Fun Puzzles
   1. [Traveling Salesman Problem](examples/tsp.ipynb)
   2. [N-Queens puzzle](examples/queens_puzzle.ipynb)

Constraint optimization problems can be easily addressed using the
[Penalty Method](https://en.wikipedia.org/wiki/Penalty_method) as shown in the example (1.III) above.
Moreover, multi-objective optimizations (with or without constraints) can also be solved as shown in
the example (1.IV), using the _weighted sum_ method.

### Contact

For any questions/comments (**regarding this code**) please contact me at: vrettasm@gmail.com
