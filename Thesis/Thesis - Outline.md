Title - Fermionic Excitation Simulator
# Introduction
Motivation, problem statement, goal of the thesis
# Theoretical Fundamentals of Quantumalgortihm
- Short Introduction into states, operations and other fundamentals
- unitary Operation and Generators
# Excitations
General overview
## Qubit Excitations
Definition and Explanation
## Fermionic Excitations
Definition and Explanation
## Fermionic systems
- Fermions, their challenges and their differences to quibts 
- Fermionic annihilation and creation operator: $a_i^\dagger$ und $a_i$.
# Different design approaches

## Approach A: Operator-Algebra 
- Concept: Implementation of the exact Algebra
- Structure: Addition, subtraction und multiplication of chained operators.
## Approach B: Generator-based evolution (symbolically)
- Concept: Apply Generator directly on a state 
- Unitary operation: How to calculate $U = \exp(\sum \theta_{pq} \hat{E}_{pq})$ efficient?
# Implementation in C++
- Explanation
- Decision on implementation
- Software architecture
- Effizienz strategy:
    - Why C++ (Memory management, Speed, etc.)
    - Data structures
    - Optimization of the loop
# Evaluation & results
- Check for Correctness with Tequila
- Performance-analyse: how does the computation scale with difference states
- Comparison of different approaches
# Conclusion and Outlook
- Future possible Additions