import math
import unittest

import spex_tequila as spex

from tests.helpers import (
    S2,
    PI,
    PI_2,
    assert_complex_almost_equal,
    assert_state_dicts_almost_equal,
    fock,
)


class TestExpectationValueTerm(unittest.TestCase):

    def test_basic_matching(self):
        # ⟨01| a†_0 a_1 |10⟩ = 1
        phi = {fock(0): 1.0}
        psi = {fock(1): 1.0}
        term = spex.FermionTerm([0], [1], 1.0)
        assert_complex_almost_equal(
            spex.expectation_value_fermionic_term(phi, psi, term), 1.0
        )

    def test_complex_weight(self):
        phi = {fock(0): 1.0}
        psi = {fock(1): 1.0}
        term = spex.FermionTerm([0], [1], 2.0 + 3.0j)
        assert_complex_almost_equal(
            spex.expectation_value_fermionic_term(phi, psi, term), 2.0 + 3.0j
        )

    def test_non_matching(self):
        # a_1 cannot annihilate the empty orbital 1
        phi = {fock(0): 1.0}
        psi = {fock(): 1.0}
        term = spex.FermionTerm([0], [1], 1.0)
        assert_complex_almost_equal(
            spex.expectation_value_fermionic_term(phi, psi, term), 0.0j
        )

    def test_asymmetric_pair_create(self):
        # a†_1 a†_0 |00⟩ = -|11⟩
        phi = {fock(0, 1): 1.0}
        psi = {fock(): 1.0}
        term = spex.FermionTerm([0, 1], [], 1.0)
        assert_complex_almost_equal(
            spex.expectation_value_fermionic_term(phi, psi, term), -1.0
        )

    def test_asymmetric_pair_annihilate(self):
        # a_1 a_0 |11⟩ = +|00⟩
        phi = {fock(): 1.0}
        psi = {fock(0, 1): 1.0}
        term = spex.FermionTerm([], [0, 1], 1.0)
        assert_complex_almost_equal(
            spex.expectation_value_fermionic_term(phi, psi, term), 1.0
        )

    def test_complex_asymmetric(self):
        # 1j·a†_1 a†_0 |00⟩ = -1j·|11⟩
        phi = {fock(0, 1): 1.0}
        psi = {fock(): 1.0}
        term = spex.FermionTerm([0, 1], [], 1.0j)
        assert_complex_almost_equal(
            spex.expectation_value_fermionic_term(phi, psi, term), -1.0j
        )

    def test_multi_term_with_phase(self):
        # ⟨011| a†_0 a_2 |110⟩ = -1
        phi = {fock(0, 1): 1.0}
        psi = {fock(1, 2): 1.0}
        term = spex.FermionTerm([0], [2], 1.0)
        assert_complex_almost_equal(
            spex.expectation_value_fermionic_term(phi, psi, term), -1.0
        )


class TestExpectationValueSum(unittest.TestCase):

    def test_sum_two_terms(self):
        phi = {fock(0): 1.0}
        psi = {fock(1): 1.0}
        terms = [spex.FermionTerm([0], [1], 1.0), spex.FermionTerm([1], [0], 2.0)]
        assert_complex_almost_equal(spex.expectation_value_fermionic(phi, psi, terms), 1.0)

    def test_cancellation(self):
        phi = {fock(0): 1.0}
        psi = {fock(1): 1.0}
        terms = [spex.FermionTerm([0], [1], 1.0), spex.FermionTerm([0], [1], -1.0)]
        assert_complex_almost_equal(spex.expectation_value_fermionic(phi, psi, terms), 0.0j)

    def test_empty_list(self):
        phi = {fock(): 1.0}
        psi = {fock(): 1.0}
        assert_complex_almost_equal(spex.expectation_value_fermionic(phi, psi, []), 0.0j)

    def test_constant_term(self):
        # ⟨ψ| 5·I |ψ⟩ = 5
        phi = {fock(): 1.0}
        psi = {fock(): 1.0}
        assert_complex_almost_equal(
            spex.expectation_value_fermionic(phi, psi, [spex.FermionTerm([], [], 5.0)]), 5.0
        )

    def test_complex_constant_term(self):
        phi = {fock(0): 1.0}
        psi = {fock(0): 1.0}
        assert_complex_almost_equal(
            spex.expectation_value_fermionic(phi, psi, [spex.FermionTerm([], [], 3.0 + 4.0j)]),
            3.0 + 4.0j,
        )


class TestAsymmetricExcitation(unittest.TestCase):

    def test_pair_creation_full(self):
        # G = i(a†_0 a†_1 - a_1 a_0) on |00⟩ at θ=π
        result = spex.apply_fermion_excitation({fock(): 1.0}, spex.FermionTerm([0, 1], [], 1.0j), PI)
        assert_state_dicts_almost_equal(result, {fock(0, 1): -1.0})

    def test_pair_creation_half(self):
        result = spex.apply_fermion_excitation({fock(): 1.0}, spex.FermionTerm([0, 1], [], 1.0j), PI_2)
        assert_state_dicts_almost_equal(result, {fock(): S2, fock(0, 1): -S2})

    def test_pair_annihilation_full(self):
        result = spex.apply_fermion_excitation(
            {fock(0, 1): 1.0}, spex.FermionTerm([], [0, 1], 1.0j), PI
        )
        assert_state_dicts_almost_equal(result, {fock(): 1.0})

    def test_pair_annihilation_half(self):
        result = spex.apply_fermion_excitation(
            {fock(0, 1): 1.0}, spex.FermionTerm([], [0, 1], 1.0j), PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0, 1): S2, fock(): S2})

    def test_single_creation_full(self):
        result = spex.apply_fermion_excitation({fock(): 1.0}, spex.FermionTerm([0], [], 1.0j), PI)
        assert_state_dicts_almost_equal(result, {fock(0): 1.0})

    def test_single_annihilation_full(self):
        result = spex.apply_fermion_excitation({fock(0): 1.0}, spex.FermionTerm([], [0], 1.0j), PI)
        assert_state_dicts_almost_equal(result, {fock(): 1.0})


class TestComplexWeightExcitation(unittest.TestCase):

    def test_real_weight(self):
        # G = a†_0 a_1 + a†_1 a_0 (w = 1) on |10⟩ at θ=π
        result = spex.apply_fermion_excitation({fock(1): 1.0}, spex.FermionTerm([0], [1], 1.0), PI)
        assert_state_dicts_almost_equal(result, {fock(0): -1.0j})

    def test_real_weight_half(self):
        result = spex.apply_fermion_excitation({fock(1): 1.0}, spex.FermionTerm([0], [1], 1.0), 1.0)
        assert_state_dicts_almost_equal(
            result,
            {fock(1): complex(math.cos(0.5), 0.0), fock(0): complex(0.0, -math.sin(0.5))},
        )

    def test_double_weight(self):
        # |w| = 2 scales the angle by 2
        result = spex.apply_fermion_excitation({fock(1): 1.0}, spex.FermionTerm([0], [1], 2.0j), PI_2)
        assert_state_dicts_almost_equal(result, {fock(0): 1.0})

    def test_negative_weight(self):
        result = spex.apply_fermion_excitation({fock(1): 1.0}, spex.FermionTerm([0], [1], -1.0j), PI)
        assert_state_dicts_almost_equal(result, {fock(0): -1.0})


class TestEdgeCases(unittest.TestCase):

    def test_zero_weight(self):
        result = spex.apply_fermion_excitation({fock(1): 1.0}, spex.FermionTerm([0], [1], 0.0), PI)
        assert_state_dicts_almost_equal(result, {fock(1): 1.0})

    def test_zero_theta(self):
        result = spex.apply_fermion_excitation({fock(1): 1.0}, spex.FermionTerm([0], [1], 1.0j), 0.0)
        assert_state_dicts_almost_equal(result, {fock(1): 1.0})

    def test_nullspace_vacuum(self):
        result = spex.apply_fermion_excitation({fock(): 1.0}, spex.FermionTerm([0], [1], 1.0j), PI)
        assert_state_dicts_almost_equal(result, {fock(): 1.0})

    def test_nullspace_fully_occupied(self):
        result = spex.apply_fermion_excitation({fock(0, 1): 1.0}, spex.FermionTerm([0], [1], 1.0j), PI)
        assert_state_dicts_almost_equal(result, {fock(0, 1): 1.0})

    def test_self_pair_nullspace(self):
        result = spex.apply_fermion_excitation({fock(0): 1.0}, spex.FermionTerm([0], [0], 1.0j), PI)
        assert_state_dicts_almost_equal(result, {fock(0): 1.0})

    def test_empty_state_raises(self):
        with self.assertRaises(ValueError):
            spex.apply_fermion_excitation({}, spex.FermionTerm([0], [1], 1.0j), PI)
