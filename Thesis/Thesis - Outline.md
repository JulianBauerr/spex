Title - Fermionic Excitation Simulator
# 1. Introduction
Motivation, problem statement, goal of the thesis

---

# 2. Theoretical Fundamentals of Quantum Computing
## Fundamentals of Quantum Information
### Quantum states
### Tensor Product
### Operators and Observables
### Unitary Operations and Generators

## Fermionic Many-Body Systems
### Indistinguishability
### Wavefunction Symmetry and Antisymmetry
### The Pauli Exclusion Principle

## The Second Quantization Framework
### Fock space
### Fermionic Operators

## Mapping Fermions to Quantum Hardware
### The Hardware Constraint: Fermions vs. Qubits
### The Jordan-Wigner Transformation

## Simulating Excitations
### Qubit Excitations
### Fermionic Excitations

---

## 3. Algorithmic Design of the Matrix-Free Simulator
no C++ syntax)

### 3.2 Direct State Evolution
- Bypass full matrix exponentiation $U = \exp(\sum \theta \hat{E})$
- Restrict operations strictly to the affected 2D subspace (affected bits)
- Apply amplitude updates using analytical evaluations: $\cos(\theta/2)$ and $\sin(\theta/2)$
### 3.3 Enforcing Fermionic Statistics (Phase Algorithm)
- Algorithm to calculate Jordan-Wigner Z-string (parity)
- Count occupied states between target indices
- Trigger phase flips based on count (logic for `phase = -phase`)

---

## 4. My C++ Implementation
(code snippets and computer science concepts)

### 4.1 Core Data Structures
- Detail container choice for the State (e.g., `std::map<uint64_t, double>` vs. `std::unordered_map`)
- Justify this structural choice
### 4.2 Bitwise Operations & Hardware Acceleration
- Apply `(1ULL << idx)` for bit masking
- Use `^` for flipping bits (creation/annihilation logic)
- Leverage `__builtin_popcountll()` for hardware-accelerated set-bit counting ( speed-up for fermionic phase calculations)
### 4.3 Subspace Filtering (Optimization)
- Breakdown of "skip" logic (checking P0)
- Explain how bypassing unaffected basis states skips unnecessary math and saves compute time

---

## 5. Evaluation & Results
(comparing custom simulator against Tequilla)

### 5.1 Validation
- Prove algorithmic correctness
- Mathematical comparison: My output vs. Tequila's output
### 5.2 Performance Benchmarking (maybe)
- Measure runtime scaling vs. increasing qubit/fermion count
- Key Graph: Compare my custom C++ runtime vs. Tequila's runtime on the exact same system (highlighting the speed of my matrix-free approach for specific excitations)

---