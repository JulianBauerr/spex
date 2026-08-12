import numpy as np
import pytest
import tequila as tq
from openfermion import FermionOperator, jordan_wigner

import spex_tequila as spex

from tests.helpers import S2, fock


def _tq_expectation(terms, circuit):
    op = sum((t.weight * FermionOperator(t.to_fermion_string()) for t in terms), FermionOperator())
    H = tq.QubitHamiltonian.from_openfermion(jordan_wigner(op))
    return complex(tq.simulate(tq.ExpectationValue(H=H, U=circuit)))


def _spex_expectation(psi, terms):
    return complex(spex.expectation_value_fermionic(psi, psi, terms))


class TestExpectationValueWithTequila:

    def test_hopping(self):
        terms = [spex.FermionTerm([0], [1], 1.0), spex.FermionTerm([1], [0], 1.0)]
        circuit = tq.gates.X(1) + tq.gates.H(0) + tq.gates.CX(0, 1)
        psi = {fock(0): S2, fock(1): S2}
        assert np.isclose(_spex_expectation(psi, terms), _tq_expectation(terms, circuit))

    @pytest.mark.parametrize("orb,qubit", [(0, 0), (1, 1)])
    def test_number_operator(self, orb, qubit):
        terms = [spex.FermionTerm([orb], [orb], 1.0)]
        circuit = tq.gates.X(qubit)
        psi = {fock(orb): 1.0}
        assert np.isclose(_spex_expectation(psi, terms), _tq_expectation(terms, circuit))

    def test_kinetic_sum(self):
        terms = [spex.FermionTerm([0], [0], 1.0), spex.FermionTerm([1], [1], 1.0)]
        circuit = tq.gates.X(0) + tq.gates.X(1)
        psi = {fock(0, 1): 1.0}
        assert np.isclose(_spex_expectation(psi, terms), _tq_expectation(terms, circuit))

    def test_complex_weight_hopping(self):
        w = 2.0 + 3.0j
        terms = [spex.FermionTerm([0], [1], w), spex.FermionTerm([1], [0], np.conj(w))]
        circuit = tq.gates.X(1) + tq.gates.H(0) + tq.gates.CX(0, 1)
        psi = {fock(0): S2, fock(1): S2}
        val = _spex_expectation(psi, terms)
        tq_val = _tq_expectation(terms, circuit)
        assert np.isclose(val.real, tq_val.real)
        assert np.isclose(val.imag, tq_val.imag)

    def test_pair_creation_hamiltonian(self):
        terms = [spex.FermionTerm([1, 0], [], 1.0), spex.FermionTerm([], [0, 1], 1.0)]
        circuit = tq.gates.H(0) + tq.gates.CX(0, 1)
        psi = {fock(): S2, fock(0, 1): S2}
        assert np.isclose(_spex_expectation(psi, terms), _tq_expectation(terms, circuit))

    def test_off_diagonal_term(self):
        terms = [spex.FermionTerm([0], [2], 1.0)]
        circuit = tq.gates.X(2)
        psi = {fock(2): 1.0}
        val = _spex_expectation(psi, terms)
        tq_val = _tq_expectation(terms, circuit)
        assert np.isclose(val.real, tq_val.real)
        assert np.isclose(val.imag, tq_val.imag)
