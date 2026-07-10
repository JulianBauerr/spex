import numpy as np
import pytest
import tequila as tq
from openfermion import FermionOperator, jordan_wigner

import spex_tequila as spex


def _tq_expectation(fermion_terms, state_circuit):
    """Compute ⟨ψ| O |ψ⟩ via Tequila for a list of FermionTerms."""
    of_terms = [t.weight * FermionOperator(t.to_fermion_string()) for t in fermion_terms]
    op = sum(of_terms, FermionOperator())
    qubit_op = jordan_wigner(op)
    H_tq = tq.QubitHamiltonian.from_openfermion(qubit_op)
    E = tq.ExpectationValue(H=H_tq, U=state_circuit)
    return complex(tq.simulate(E))


def _spex_expectation(psi, fermion_terms):
    return complex(spex.expectation_value_fermionic(psi, psi, fermion_terms))


S2 = 1.0 / np.sqrt(2)


class TestExpectationValueWithTequila:
    """Cross-validate expectation_value_fermionic against Tequila + OpenFermion."""

    def test_hopping(self):
        """⟨ψ| a†_0 a_1 + a†_1 a_0 |ψ⟩ for ψ = (|01⟩+|10⟩)/√2"""
        terms = [
            spex.FermionTerm([0], [1], 1.0),
            spex.FermionTerm([1], [0], 1.0),
        ]
        c = tq.gates.X(1) + tq.gates.H(0) + tq.gates.CX(0, 1)

        psi = {0b01: complex(S2), 0b10: complex(S2)}
        assert np.isclose(_spex_expectation(psi, terms), _tq_expectation(terms, c))

    @pytest.mark.parametrize("orb,qubit", [(0, 0), (1, 1)])
    def test_number_operator(self, orb, qubit):
        """⟨ψ| a†_i a_i |ψ⟩ = 1 when orbital i is occupied."""
        terms = [spex.FermionTerm([orb], [orb], 1.0)]
        c = tq.gates.X(qubit)

        psi = {1 << orb: 1.0}
        assert np.isclose(_spex_expectation(psi, terms), _tq_expectation(terms, c))

    def test_kinetic_sum(self):
        """⟨ψ| a†_0 a_0 + a†_1 a_1 |ψ⟩ = 2 on |11⟩"""
        terms = [
            spex.FermionTerm([0], [0], 1.0),
            spex.FermionTerm([1], [1], 1.0),
        ]
        c = tq.gates.X(0) + tq.gates.X(1)

        psi = {0b11: 1.0}
        assert np.isclose(_spex_expectation(psi, terms), _tq_expectation(terms, c))

    def test_complex_weight_hopping(self):
        """⟨ψ| (2+3j)·a†_0 a_1 + (2-3j)·a†_1 a_0 |ψ⟩ on Bell state"""
        w = 2.0 + 3.0j
        terms = [
            spex.FermionTerm([0], [1], w),
            spex.FermionTerm([1], [0], np.conj(w)),
        ]
        c = tq.gates.X(1) + tq.gates.H(0) + tq.gates.CX(0, 1)

        psi = {0b01: complex(S2), 0b10: complex(S2)}
        val = _spex_expectation(psi, terms)
        tq_val = _tq_expectation(terms, c)
        assert np.isclose(val.real, tq_val.real)
        assert np.isclose(val.imag, tq_val.imag)

    def test_pair_creation_hamiltonian(self):
        """⟨ψ| a†_0 a†_1 + a_1 a_0 |ψ⟩ on (|00⟩+|11⟩)/√2"""
        terms = [
            spex.FermionTerm([1, 0], [], 1.0),
            spex.FermionTerm([], [0, 1], 1.0),
        ]
        c = tq.gates.H(0) + tq.gates.CX(0, 1)

        psi = {0b00: complex(S2), 0b11: complex(S2)}
        assert np.isclose(_spex_expectation(psi, terms), _tq_expectation(terms, c))

    def test_off_diagonal_term(self):
        """⟨ψ| a†_0 a_2 |ψ⟩ on a state where it's non-zero"""
        terms = [spex.FermionTerm([0], [2], 1.0)]
        c = tq.gates.X(2)  # |100⟩ = bit 2 occupied

        psi = {0b100: 1.0}
        val = _spex_expectation(psi, terms)
        tq_val = _tq_expectation(terms, c)
        assert np.isclose(val.real, tq_val.real)
        assert np.isclose(val.imag, tq_val.imag)
