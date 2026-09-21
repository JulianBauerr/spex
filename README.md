# SPEX

A fermionic excitation, fSWAP, and expectation-value simulator built on fermionic
particles for [Tequila](https://github.com/tequilahub/tequila), implemented in C++
with [pybind11](https://github.com/pybind/pybind11).

The core is a C++ extension (`spex_tequila`) that operates on sparse quantum states
represented as a `dict` mapping integer Fock bit-strings to complex amplitudes. On
top of it, `spex_expval.py` provides a high-level `SpexExpval` objective for
computing expectation values on Hartree-Fock states.

## Features

- **Fermionic expectation values** — `⟨φ| O |ψ⟩` for single terms or sums of terms.
- **Fermionic excitations** — `exp(-i θ/2 · (w·A + w̄·A†))` for a general
  `FermionTerm`, and `apply_abstract_generator` for sums of terms.
- **Qubit excitations** — `exp(-i θ/2 · (|k⟩⟨l| + |l⟩⟨k|))` on paired orbitals.
- **fSWAP** — the fermionic SWAP operator on two modes.
- **High-level objective** — `SpexExpval` builds a Hamiltonian from molecular
  integrals (or a Tequila molecule) and evaluates expectation values.

## Requirements

- Python ≥ 3.9
- A C++17 compiler (for building the extension)

Runtime dependencies (installed automatically):

- `numpy`
- `tequila-basic ≥ 1.9.0`
- `openfermion ≥ 1.7.0`

## Installation

Build and install from source:

```bash
git clone https://git.rz.uni-augsburg.de/qalg-a/spex
cd spex
pip install .
```

For development (rebuilds the extension in place):

```bash
pip install -e .
```

## Usage

A quantum state is a Python `dict` of integer Fock bit-strings to complex
amplitudes, e.g. `{0b001: 1.0}` is the Fock state with spin-orbital `0` occupied.

### Low-level API (`spex_tequila`)

```python
import math
import spex_tequila as spex

# Expectation value of a single term: ⟨01| a†_0 a_1 |10⟩ = 1
phi = {0b001: 1.0}
psi = {0b010: 1.0}
term = spex.FermionTerm([0], [1], 1.0)  # creation [0], annihilation [1], weight 1
spex.expectation_value_fermionic_term(phi, psi, term)  # -> (1+0j)

# Sum of terms
terms = [spex.FermionTerm([0], [1], 1.0), spex.FermionTerm([1], [0], 2.0)]
spex.expectation_value_fermionic(phi, psi, terms)

# Fermionic pair creation: exp(-i π/2 · (a†_0 a†_1 + a_1 a_0)) |00⟩
spex.apply_fermion_excitation(
    {0b000: 1.0}, spex.FermionTerm([0, 1], [], 1.0j), math.pi
)  # -> {0b011: (-1+0j)}

# Qubit excitation on orbitals 0 and 1
spex.apply_qubit_excitation({0b001: 1.0}, [0], [1], math.pi / 2)

# fSWAP on modes 0 and 1
spex.apply_fswap({0b01: 1.0}, 0, 1)  # -> {0b10: (1+0j)}

# Parse a fermion operator string
t = spex.parse_fermion_string("0^ 1", 2.0)
t.to_fermion_string()  # -> "0^ 1"
```

### High-level API (`SpexExpval`)

```python
import numpy as np
from spex_expval import SpexExpval

# Build a Hamiltonian from raw integrals and evaluate it on the Hartree-Fock state.
expval = SpexExpval(
    int1e=[[0.0, 1.0], [1.0, 0.0]],   # one-body integrals
    int2e=np.zeros((2, 2, 2, 2)),     # two-body integrals
    n_alpha=1, n_beta=0,              # occupation
)
expval.simulate()  # -> Hartree-Fock expectation value

# Alternatively, drive it with a Tequila molecule:
import tequila as tq
mol = tq.Molecule("H 0 0 0\nH 0 0 1.4", basis_set="sto-3g")
SpexExpval(molecule=mol).simulate()
```

## Testing

Run the test suite with `pytest`:

```bash
pytest tests/
```

## License

[MIT](LICENSE) — Copyright (c) 2026 Julian Bauer.
