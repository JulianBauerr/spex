Title - Fermionic Excitation Simulator
# Introduction
Motivation, problem statement, goal of the thesis





# Theoretical Fundamentals of Quantum Computing
## XXX
- Short Introduction into states, operations and other fundamentals
- unitary Operation and Generators
### Quantum states
### Operators and Observables
### Unitary Operations and Generators
### Tensor Product (maybe)


## Fermions
Definition of the particle
- Pauli Exclusion Principle
- Spin and the Spin-Statistics Theorem
- Wavefunction types (a)symmetric
### The Pauli Exclusion Principle
### Wavefunction Symmetry and Antisymmetry
### Indistinguishability & Pauli Exclusion
### Differences: Fermions vs. Qubits
### Second Quantization and Fermionic Operators
### Jordan-Wigner transformation


## Quantum Excitations
### General overview
### Qubit Excitations
Definition and Explanation
### Fermionic Excitations
Definition and Explanation





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





# Summary and Outlook
- Future possible Additions