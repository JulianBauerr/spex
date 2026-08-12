import unittest

import spex_tequila as spex

from tests.helpers import PI, assert_state_dicts_almost_equal, fock


class TestQubitExcitation(unittest.TestCase):

    K = [0, 1]
    L = [2, 3]

    def test_forward_swap(self):
        result = spex.apply_qubit_excitation({fock(0, 1): 1.0}, self.K, self.L, PI)
        assert_state_dicts_almost_equal(result, {fock(2, 3): 1.0})

    def test_reverse_swap(self):
        result = spex.apply_qubit_excitation({fock(2, 3): 1.0}, self.K, self.L, PI)
        assert_state_dicts_almost_equal(result, {fock(0, 1): -1.0})

    def test_nullspace_empty_and_full(self):
        state = {fock(): 0.5, fock(0, 1, 2, 3): 0.5}
        result = spex.apply_qubit_excitation(state, self.K, self.L, PI)
        assert_state_dicts_almost_equal(result, state)

    def test_nullspace_edge_case(self):
        result = spex.apply_qubit_excitation({fock(0, 3): 1.0}, self.K, self.L, PI)
        assert_state_dicts_almost_equal(result, {fock(0, 3): 1.0})

    def test_theta_zero(self):
        result = spex.apply_qubit_excitation({fock(0, 1): 1.0}, self.K, self.L, 0.0)
        assert_state_dicts_almost_equal(result, {fock(0, 1): 1.0})
