import unittest

import spex_tequila as spex

from tests.helpers import PI_2, assert_state_dicts_almost_equal, fock


class TestFSwap(unittest.TestCase):

    def test_identity_same_index(self):
        result = spex.apply_fswap({fock(1, 3): 1.0}, 1, 1)
        assert_state_dicts_almost_equal(result, {fock(1, 3): 1.0})

    def test_00_unchanged(self):
        result = spex.apply_fswap({fock(): 1.0}, 0, 1)
        assert_state_dicts_almost_equal(result, {fock(): 1.0})

    def test_01_swapped(self):
        result = spex.apply_fswap({fock(0): 1.0}, 0, 1)
        assert_state_dicts_almost_equal(result, {fock(1): 1.0})

    def test_10_swapped(self):
        result = spex.apply_fswap({fock(1): 1.0}, 0, 1)
        assert_state_dicts_almost_equal(result, {fock(0): 1.0})

    def test_11_negative(self):
        result = spex.apply_fswap({fock(0, 1): 1.0}, 0, 1)
        assert_state_dicts_almost_equal(result, {fock(0, 1): -1.0})

    def test_self_inverse(self):
        state = {fock(): 1.0, fock(0, 1): 1.0}
        once = spex.apply_fswap(state, 0, 2)
        twice = spex.apply_fswap(once, 0, 2)
        assert_state_dicts_almost_equal(twice, state)

    def test_higher_orbitals_unaffected(self):
        result = spex.apply_fswap({fock(0, 2, 3): 1.0}, 1, 2)
        assert_state_dicts_almost_equal(result, {fock(0, 1, 3): 1.0})

    def test_superposition(self):
        state = {fock(0): 1.0, fock(1): 1.0}
        result = spex.apply_fswap(state, 0, 1)
        assert_state_dicts_almost_equal(result, state)

    def test_superposition_with_phase(self):
        state = {fock(0): 1.0, fock(0, 1): 1.0}
        result = spex.apply_fswap(state, 0, 1)
        assert_state_dicts_almost_equal(result, {fock(1): 1.0, fock(0, 1): -1.0})


class TestFSwapExcitationEquivalence(unittest.TestCase):

    def _excitation(self, state, src, dst):
        return spex.apply_fermion_excitation(state, spex.FermionTerm([dst], [src], 1.0j), PI_2)

    def test_excitation_equivalence(self):
        # fSWAP(1,2) ∘ exc(0→2) ∘ fSWAP(1,2) == exc(0→1)
        initial = {fock(0, 1): 1.0}
        expected = self._excitation(initial, 0, 1)

        tmp = spex.apply_fswap(initial, 1, 2)
        tmp = self._excitation(tmp, 0, 2)
        result = spex.apply_fswap(tmp, 1, 2)

        assert_state_dicts_almost_equal(result, expected)

    def test_three_way_equivalence(self):
        # fSWAP(0,2) ∘ exc(1→2) ∘ fSWAP(0,2) == exc(1→0)
        initial = {fock(0, 1, 2): 1.0}
        expected = self._excitation(initial, 1, 0)

        tmp = spex.apply_fswap(initial, 0, 2)
        tmp = self._excitation(tmp, 1, 2)
        result = spex.apply_fswap(tmp, 0, 2)

        assert_state_dicts_almost_equal(result, expected)
