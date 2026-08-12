import unittest

import spex_tequila as spex

from tests.helpers import S2, PI, PI_2, assert_state_dicts_almost_equal, fock


class TestFermionExcitation(unittest.TestCase):

    def test_excite_from_empty_orbital(self):
        result = spex.apply_fermion_excitation(
            {fock(0): 1.0}, spex.FermionTerm([1], [2], 1.0j), 0.25
        )
        assert_state_dicts_almost_equal(result, {fock(0): 1.0})

    def test_pauli_blocking(self):
        result = spex.apply_fermion_excitation(
            {fock(0, 1): 1.0}, spex.FermionTerm([0], [1], 1.0j), PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0, 1): 1.0})

    def test_vacuum(self):
        result = spex.apply_fermion_excitation(
            {fock(): 1.0}, spex.FermionTerm([0], [1], 1.0j), PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(): 1.0})

    def test_single_excitation_50_50_split(self):
        result = spex.apply_fermion_excitation(
            {fock(0): 1.0}, spex.FermionTerm([0], [1], 1.0j), PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0): S2, fock(1): -S2})

    def test_single_excitation_full_transfer(self):
        result = spex.apply_fermion_excitation(
            {fock(0): 1.0}, spex.FermionTerm([0], [1], 1.0j), PI
        )
        assert_state_dicts_almost_equal(result, {fock(1): -1.0})

    def test_single_deexcitation(self):
        result = spex.apply_fermion_excitation(
            {fock(1): 1.0}, spex.FermionTerm([0], [1], 1.0j), -PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(1): S2, fock(0): -S2})

    def test_parity_jump_over_empty_orbital(self):
        result = spex.apply_fermion_excitation(
            {fock(0): 1.0}, spex.FermionTerm([0], [2], 1.0j), PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0): S2, fock(2): -S2})

    def test_parity_jump_over_occupied_orbital(self):
        result = spex.apply_fermion_excitation(
            {fock(0, 1): 1.0}, spex.FermionTerm([0], [2], 1.0j), PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0, 1): S2, fock(1, 2): S2})

    def test_double_excitation(self):
        result = spex.apply_fermion_excitation(
            {fock(0, 1): 1.0}, spex.FermionTerm([0, 1], [2, 3], 1.0j), PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0, 1): S2, fock(2, 3): S2})

    def test_double_excitation_parity_trap(self):
        result = spex.apply_fermion_excitation(
            {fock(0, 1, 3): 1.0}, spex.FermionTerm([0, 1], [2, 4], 1.0j), PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0, 1, 3): S2, fock(2, 3, 4): -S2})

    def test_superposition_partial_branch_hit(self):
        result = spex.apply_fermion_excitation(
            {fock(0): 0.8, fock(1): 0.6}, spex.FermionTerm([0], [2], 1.0j), PI
        )
        assert_state_dicts_almost_equal(result, {fock(2): -0.8, fock(1): 0.6})

    def test_destructive_quantum_interference(self):
        result = spex.apply_fermion_excitation(
            {fock(0): S2, fock(1): S2}, spex.FermionTerm([0], [1], 1.0j), PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0): 1.0})

    def test_complex_phase_preservation(self):
        result = spex.apply_fermion_excitation(
            {fock(0): 1.0j}, spex.FermionTerm([0], [1], 1.0j), PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0): S2 * 1.0j, fock(1): -S2 * 1.0j})

    def test_complex_interference(self):
        result = spex.apply_fermion_excitation(
            {fock(0): S2, fock(1): S2 * 1.0j}, spex.FermionTerm([0], [1], 1.0j), PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0): 0.5 + 0.5j, fock(1): -0.5 + 0.5j})
