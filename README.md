# Genetic Algorithms in Python

"Genetic Algorithms [(GA)](https://en.wikipedia.org/wiki/Genetic_algorithm), are meta heuristic algorithms
inspired by the process of natural selection and belong to a larger class of evolutionary algorithms (EA)."

-- (From Wikipedia, the free encyclopedia)

This repository implements a GA in Python3 programming language (using only Numpy as additional library).
The initial approach offers a "_StandardGA_" class, where the whole population is replaced by another one
of offsprings at the end of each iteration (or epoch).

NOTE:
For computationally expensive fitness functions the StandardGA class provides the option of parallel
evaluation, by setting in the method run(..., parallel=True). However, for fast fitness functions this
might actually cause the evolution to converge slower. So the default setting here is "parallel=False".

  > NEWS:
  > The IslandModelGA is now **available**! For the moment there is only one migration policy
  > as implemented by the Clockwise Migrator. Any new migration policies can be easily added
  > by inheriting directly from the Migration Operator (base) class.
  > 

The current implementation offers a variety of genetic operators including:

- **Selection operators**:
  - [Linear Rank Selector](code/src/operators/selection/linear_rank_selector.py)
  - [Random Selector](code/src/operators/selection/random_selector.py)
  - [Roulette Wheel Selector](code/src/operators/selection/roulette_wheel_selector.py)
  - [Stochastic Universal Selector](code/src/operators/selection/stochastic_universal_selector.py)
  - [Tournament Selector](code/src/operators/selection/tournament_selector.py)
  - [Truncation Selector](code/src/operators/selection/truncation_selector.py)

- **Crossover operators**:
  - [Single-Point Crossover](code/src/operators/crossover/single_point_crossover.py)
  - [Multi-Point Crossover](code/src/operators/crossover/mutli_point_crossover.py)
  - [Uniform Crossover](code/src/operators/crossover/uniform_crossover.py)

- **Mutation operators**:
  - [Random Mutator](code/src/operators/mutation/random_mutator.py)
  - [Shuffle Mutator](code/src/operators/mutation/shuffle_mutator.py)
  - [Swap Mutator](code/src/operators/mutation/swap_mutator.py)

- **Migration operators**
  - [Clockwise Migrator](code/src/operators/migration/clockwise_migration.py)

Note that incorporating additional genetic operators is easily facilitated by inheriting from the base classes:
- [SelectionOperator](code/src/operators/selection/select_operator.py)
- [CrossoverOperator](code/src/operators/crossover/crossover_operator.py)
- [MutationOperator](code/src/operators/mutation/mutate_operator.py)
- [MigrationOperator](code/src/operators/migration/migration_operator.py)

and implement the basic interface as described in these classes. In the examples below I show how one can use
this code to run a GA for optimization problems (maximization/minimization) with and without constraints. The
project is ongoing so new things might come along the way.

### Examples

Some optimization examples on how to use this algorithm:

1. StandardGA
   1. [Minimization](examples/sphere.ipynb)
   2. [Maximization](examples/rastrigin.ipynb)
   3. [Minimization with constrains](examples/rosenbrock_on_a_disk.ipynb)
2. IslandModelGA
   1. [Sphere Function](examples/sphere_in_parallel.ipynb)
   2. [Easom Function](examples/easom_in_parallel.ipynb)

Constraint optimization problems can be easily addressed using the
[Penalty Method](https://en.wikipedia.org/wiki/Penalty_method) as described in the link above.

### Contact

For any questions/comments (**regarding this code**) please contact me at: vrettasm@gmail.com
