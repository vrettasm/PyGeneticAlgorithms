# Genetic Algorithms in Python

"Genetic Algorithms [(GA)](https://en.wikipedia.org/wiki/Genetic_algorithm), are metaheuristic algorithms
inspired by the process of natural selection and belong to a larger class of evolutionary algorithms (EA)."

-- (From Wikipedia, the free encyclopedia)

This repository implements a GA in Python3 programming language (using only Numpy as additional library).
The initial approach offers a "_StandardGA_" class, where the whole population is replaced by another one
of offsprings at the end of each iteration (or epoch).

The current implementation offers a variety of genetic operators including:

- **Selection operators**:
  - [Linear Rank Selector](code/src/operators/selection/linear_rank_selector.py)
  - Random Selector
  - Roulette Wheel Selector
  - Stochastic Universal Selector
  - Tournament Selector
  - Truncation Selector

- **Crossover operators**:
  - Single-Point Crossover
  - Multi-Point Crossover
  - Uniform Crossover

- **Mutation operators**:
  - Random Mutator
  - Shuffle Mutator
  - Swap Mutator

Note that incorporating additional genetic operators is easily facilitated by inheriting from the base classes:
- SelectionOperator
- CrossoverOperator 
- MutationOperator

and implement the basic interface as described in these classes. In the examples below I show how one can use
this code to run a GA for optimization problems (maximization/minimization) with and without constraints. The
project is ongoing so new things might come along the way.

### Examples

Some optimization examples on how to use this algorithm:

1. [Minimization](examples/sphere.ipynb)
2. [Maximization](examples/rastrigin.ipynb)
3. [Minimization with constrains](examples/rosenbrock_on_a_disk.ipynb)

Constraint optimization problems can be easily addressed using the
[Penalty Method](https://en.wikipedia.org/wiki/Penalty_method) as described in the link above.

### To Do:

- Implement the 'IslandModel'
- Add tests
- Add more examples

### Contact

For any questions/comments (**regarding this code**) please contact me at: vrettasm@gmail.com
