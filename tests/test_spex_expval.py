# Tests for the SpexExpval objective in spex_expval.py

import unittest

import numpy as np
import tequila as tq
from openfermion import FermionOperator, jordan_wigner

import spex_tequila as spex
from spex_expval import SpexExpval

from tests.helpers import assert_complex_almost_equal, fock

def _make_expval(operator=None, int1e=None, int2e=None, n_alpha=1, n_beta=0, **kwargs):
    if int1e is None:
        int1e = [[0.0, 0.0], [0.0, 0.0]]
    if int2e is None:
        int2e = np.zeros((2, 2, 2, 2))
    return SpexExpval(
        int1e=int1e,
        int2e=int2e,
        n_alpha=n_alpha,
        n_beta=n_beta,
        operator=operator,
        **kwargs,
    )


def _term_list(expr):
    return [
        (list(t.creation_idx), list(t.annihilation_idx), complex(t.weight))
        for t in expr
    ]


def _tequila_hamiltonian(terms):
    op = sum(
        (t.weight * FermionOperator(t.to_fermion_string()) for t in terms),
        FermionOperator(),
    )
    return tq.QubitHamiltonian.from_openfermion(jordan_wigner(op))


def _hf_circuit(n_alpha, n_beta, norb):
    occupied = list(range(n_alpha)) + [norb + i for i in range(n_beta)]
    circuit = None
    for qubit in occupied:
        circuit = tq.gates.X(qubit) if circuit is None else circuit + tq.gates.X(qubit)
    return circuit


def _tequila_expectation(terms, n_alpha, n_beta, norb):
    H = _tequila_hamiltonian(terms)
    U = _hf_circuit(n_alpha, n_beta, norb)
    return float(tq.simulate(tq.ExpectationValue(H=H, U=U)))


def _assert_matches_tequila(test_case, expval, places=8):
    ref = _tequila_expectation(
        expval.hamiltonian, expval.n_alpha, expval.n_beta, expval.norb
    )
    test_case.assertAlmostEqual(expval.simulate(), ref, places=places)


class TestHamiltonianBuilding(unittest.TestCase):
    def test_one_body_number(self):
        e = _make_expval(int1e=[[1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(_term_list(e.hamiltonian), [
            ([0], [0], 1.0 + 0j),
            ([2], [2], 1.0 + 0j),
        ])

    def test_one_body_hopping(self):
        e = _make_expval(int1e=[[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(_term_list(e.hamiltonian), [
            ([0], [1], 1.0 + 0j),
            ([2], [3], 1.0 + 0j),
            ([1], [0], 1.0 + 0j),
            ([3], [2], 1.0 + 0j),
        ])

    def test_two_body(self):
        g = np.zeros((2, 2, 2, 2))
        g[0, 1, 0, 1] = 2.0
        e = _make_expval(int2e=g)
        self.assertEqual(_term_list(e.hamiltonian), [
            ([0, 0], [1, 1], 1.0 + 0j),
            ([2, 2], [3, 3], 1.0 + 0j),
            ([2, 0], [1, 3], 1.0 + 0j),
            ([0, 2], [3, 1], 1.0 + 0j),
        ])

    def test_core_energy(self):
        e = _make_expval(n_alpha=0, n_beta=0, e_core=5.0)
        self.assertEqual(_term_list(e.hamiltonian), [([], [], 5.0 + 0j)])


class TestOperatorBuilding(unittest.TestCase):
    def test_string_H_returns_hamiltonian(self):
        e = _make_expval()
        self.assertIs(e.build_operator("H"), e.hamiltonian)

    def test_string_I_is_callable(self):
        e = _make_expval()
        self.assertTrue(callable(e.build_operator("I")))

    def test_number_is_constant_term(self):
        e = _make_expval()
        self.assertEqual(_term_list(e.build_operator(5.0)), [([], [], 5.0 + 0j)])

    def test_fermion_operator_alpha(self):
        e = _make_expval()
        self.assertEqual(
            _term_list(e.build_operator(FermionOperator("0^ 0", 2.0))),
            [([0], [0], 2.0 + 0j)],
        )

    def test_fermion_operator_beta_remapped(self):
        # openfermion index 1 is the beta spin orbital of spatial orbital 0,
        # remapped to spex spin orbital 0 + norb = 2.
        e = _make_expval()
        self.assertEqual(
            _term_list(e.build_operator(FermionOperator("1^ 1", 1.0))),
            [([2], [2], 1.0 + 0j)],
        )

    def test_fermion_operator_mixed_spin(self):
        e = _make_expval()
        self.assertEqual(
            _term_list(e.build_operator(FermionOperator("0^ 1", 1.0))),
            [([0], [2], 1.0 + 0j)],
        )

    def test_complex_weight_preserved(self):
        e = _make_expval()
        terms = e.build_operator(FermionOperator("0^ 0", 2.0 + 3.0j))
        assert_complex_almost_equal(complex(terms[0].weight), 2.0 + 3.0j)


class TestExpectationValueWithTequila(unittest.TestCase):
    def test_number_operator(self):
        e = _make_expval(operator=FermionOperator("0^ 0", 1.0))
        self.assertAlmostEqual(
            e.simulate(), _tequila_expectation(e.operator, 1, 0, 2), places=8
        )

    def test_number_operator_weight(self):
        e = _make_expval(operator=FermionOperator("0^ 0", 2.0))
        self.assertAlmostEqual(
            e.simulate(), _tequila_expectation(e.operator, 1, 0, 2), places=8
        )

    def test_beta_number_on_empty_orbital(self):
        e = _make_expval(operator=FermionOperator("1^ 1", 2.0))
        self.assertAlmostEqual(
            e.simulate(), _tequila_expectation(e.operator, 1, 0, 2), places=8
        )

    def test_hopping_single_term(self):
        # a_1 annihilates the empty orbital 1, so the expectation value is zero
        e = _make_expval(operator=FermionOperator("0^ 1", 1.0))
        self.assertAlmostEqual(
            e.simulate(), _tequila_expectation(e.operator, 1, 0, 2), places=8
        )

    def test_sum_and_cancellation(self):
        op = (
            FermionOperator("0^ 0", 2.0)
            + FermionOperator("0^ 0", 3.0)
            + FermionOperator("0^ 0", -3.0)
        )
        e = _make_expval(operator=op)
        self.assertAlmostEqual(
            e.simulate(), _tequila_expectation(e.operator, 1, 0, 2), places=8
        )

    def test_constant_term(self):
        e = _make_expval(operator=5.0)
        self.assertAlmostEqual(
            e.simulate(), _tequila_expectation(e.operator, 1, 0, 2), places=8
        )


class TestHamiltonianEnergyWithTequila(unittest.TestCase):
    def test_number_alpha(self):
        _assert_matches_tequila(self, _make_expval(int1e=[[1.0, 0.0], [0.0, 0.0]]))

    def test_number_closed_shell(self):
        _assert_matches_tequila(
            self, _make_expval(int1e=[[1.0, 0.0], [0.0, 0.0]], n_alpha=1, n_beta=1)
        )

    def test_hopping(self):
        _assert_matches_tequila(self, _make_expval(int1e=[[0.0, 1.0], [1.0, 0.0]]))

    def test_full_one_body(self):
        _assert_matches_tequila(
            self, _make_expval(int1e=[[1.0, 0.5], [0.5, 2.0]])
        )

    def test_two_body(self):
        g = np.zeros((2, 2, 2, 2))
        g[0, 0, 1, 1] = 2.0
        _assert_matches_tequila(self, _make_expval(int2e=g, n_alpha=2, n_beta=0))

    def test_two_body_closed_shell(self):
        g = np.zeros((2, 2, 2, 2))
        g[0, 0, 1, 1] = 2.0
        _assert_matches_tequila(self, _make_expval(int2e=g, n_alpha=1, n_beta=1))

    def test_core_energy(self):
        _assert_matches_tequila(
            self, _make_expval(int1e=[[0.0, 0.0], [0.0, 0.0]], e_core=5.0)
        )

    def test_combined(self):
        g = np.zeros((2, 2, 2, 2))
        g[0, 0, 1, 1] = 2.0
        g[0, 0, 0, 0] = 1.0
        e = _make_expval(
            int1e=[[0.5, 0.1], [0.1, 0.3]],
            int2e=g,
            n_alpha=1,
            n_beta=1,
            e_core=0.7,
        )
        _assert_matches_tequila(self, e)


class TestRandomHamiltoniansWithTequila(unittest.TestCase):
    def test_random_hamiltonians(self):
        rng = np.random.default_rng(0)
        for norb in (2, 3):
            for n_alpha, n_beta in [(1, 0), (1, 1), (2, 1), (2, 2)]:
                if n_alpha > norb or n_beta > norb:
                    continue
                h = rng.normal(size=(norb, norb))
                h = (h + h.T) / 2
                g = rng.normal(size=(norb,) * 4)
                g = (g + g.transpose(1, 0, 3, 2) + g.transpose(2, 3, 0, 1) + g.transpose(3, 2, 1, 0)) / 4
                with self.subTest(norb=norb, n_alpha=n_alpha, n_beta=n_beta):
                    e = SpexExpval(int1e=h, int2e=g, n_alpha=n_alpha, n_beta=n_beta)
                    ref = _tequila_expectation(e.hamiltonian, n_alpha, n_beta, norb)
                    self.assertAlmostEqual(e.simulate(), ref, places=8)


class TestHartreeFockState(unittest.TestCase):
    def test_single_alpha(self):
        e = _make_expval()
        self.assertEqual(e._init_state_bra, {fock(0): 1.0})

    def test_closed_shell(self):
        # one alpha + one beta electron occupy spin orbitals 0 and 2
        e = _make_expval(n_alpha=1, n_beta=1)
        self.assertEqual(e._init_state_bra, {fock(0, 2): 1.0})


if __name__ == "__main__":
    unittest.main()
