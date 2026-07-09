import cmath
import math
import unittest

import spex_tequila as spex


def assertComplexAlmostEqual(actual, expected, places=11):
    diff = abs(complex(actual) - complex(expected))
    if diff > 10 ** (-places):
        raise AssertionError(
            f"Expected {complex(expected)}, got {complex(actual)}, diff={diff:.2e}"
        )

def assertStateDictsAlmostEqual(actual_dict, expected_dict, places=11):
    atol = 10 ** (-places)
    all_keys = set(actual_dict.keys()) | set(expected_dict.keys())
    for key in all_keys:
        val_actual = complex(actual_dict.get(key, 0.0 + 0.0j))
        val_expected = complex(expected_dict.get(key, 0.0 + 0.0j))
        diff = abs(val_actual - val_expected)
        if diff > atol:
            act_str = ", ".join(f"{k}: {v}" for k, v in actual_dict.items())
            exp_str = ", ".join(f"{k}: {v}" for k, v in expected_dict.items())
            raise AssertionError(
                f"\nState Vector mismatch at int basis |{key}> (0b{key:b}):\n"
                f"  Actual   : {val_actual}\n"
                f"  Expected : {val_expected}\n"
                f"  Diff     : {diff:.2e} (Tolerance: 1e-{places})\n"
                f"  Full Act : {{{act_str}}}\n"
                f"  Full Exp : {{{exp_str}}}"
            )


class TestExpectationValueTerm(unittest.TestCase):
    """Unit tests for expectation_value_fermionic_term"""

    def test_basic_matching(self):
        # ⟨01| a†_0 a_1 |10⟩ = (1+0j)
        phi = {0b01: 1.0}   # |01⟩
        psi = {0b10: 1.0}   # |10⟩
        term = spex.FermionTerm([0], [1], 1.0)
        result = spex.expectation_value_fermionic_term(phi, psi, term)
        assertComplexAlmostEqual(result, 1.0 + 0.0j)

    def test_complex_weight(self):
        # ⟨01| (2+3j)·a†_0 a_1 |10⟩ = (2+3j)
        phi = {0b01: 1.0}
        psi = {0b10: 1.0}
        term = spex.FermionTerm([0], [1], 2.0 + 3.0j)
        result = spex.expectation_value_fermionic_term(phi, psi, term)
        assertComplexAlmostEqual(result, 2.0 + 3.0j)

    def test_non_matching(self):
        # ⟨01| a†_0 a_1 |00⟩ = 0j  (a_1 can't act on empty orbital 1)
        phi = {0b01: 1.0}
        psi = {0b00: 1.0}
        term = spex.FermionTerm([0], [1], 1.0)
        result = spex.expectation_value_fermionic_term(phi, psi, term)
        assertComplexAlmostEqual(result, 0.0j)

    def test_asymmetric_pair_create(self):
        # a†_1 a†_0 |00⟩ = -|11⟩ ⇒ ⟨11| a†_1 a†_0 |00⟩ = -1
        phi = {0b11: 1.0}
        psi = {0b00: 1.0}
        term = spex.FermionTerm([0, 1], [], 1.0)
        result = spex.expectation_value_fermionic_term(phi, psi, term)
        assertComplexAlmostEqual(result, -1.0 + 0.0j)

    def test_asymmetric_pair_annihilate(self):
        # a_1 a_0 |11⟩ = +|00⟩ ⇒ ⟨00| a_1 a_0 |11⟩ = +1
        phi = {0b00: 1.0}
        psi = {0b11: 1.0}
        term = spex.FermionTerm([], [0, 1], 1.0)
        result = spex.expectation_value_fermionic_term(phi, psi, term)
        assertComplexAlmostEqual(result, 1.0 + 0.0j)

    def test_complex_asymmetric(self):
        # ⟨11| 1j·a†_1 a†_0 |00⟩ = 1j·(-1) = -1j
        phi = {0b11: 1.0}
        psi = {0b00: 1.0}
        term = spex.FermionTerm([0, 1], [], 1.0j)
        result = spex.expectation_value_fermionic_term(phi, psi, term)
        assertComplexAlmostEqual(result, -1.0j)

    def test_multi_term_with_phase(self):
        # ⟨011| a†_0 a_2 |110⟩ = -⟨011||011⟩ = -1
        phi = {0b011: 1.0}  # |011⟩ = 3
        psi = {0b110: 1.0}  # |110⟩ = 6
        term = spex.FermionTerm([0], [2], 1.0)
        result = spex.expectation_value_fermionic_term(phi, psi, term)
        assertComplexAlmostEqual(result, -1.0 + 0.0j)


class TestExpectationValueSum(unittest.TestCase):
    """Unit tests for expectation_value_fermionic"""

    def test_sum_two_terms(self):
        # ⟨01| a†_0 a_1 + 2·a†_1 a_0 |10⟩ = 1.0
        phi = {0b01: 1.0}
        psi = {0b10: 1.0}
        terms = [
            spex.FermionTerm([0], [1], 1.0),
            spex.FermionTerm([1], [0], 2.0),
        ]
        result = spex.expectation_value_fermionic(phi, psi, terms)
        assertComplexAlmostEqual(result, 1.0 + 0.0j)

    def test_cancellation(self):
        # ⟨01| a†_0 a_1 + (-1)·a†_0 a_1 |10⟩ = 0
        phi = {0b01: 1.0}
        psi = {0b10: 1.0}
        terms = [
            spex.FermionTerm([0], [1], 1.0),
            spex.FermionTerm([0], [1], -1.0),
        ]
        result = spex.expectation_value_fermionic(phi, psi, terms)
        assertComplexAlmostEqual(result, 0.0j)

    def test_empty_list(self):
        phi = {0b00: 1.0}
        psi = {0b00: 1.0}
        result = spex.expectation_value_fermionic(phi, psi, [])
        assertComplexAlmostEqual(result, 0.0j)

    def test_constant_term(self):
        # ⟨ψ| 5.0·I |ψ⟩ = 5.0
        phi = {0b00: 1.0}
        psi = {0b00: 1.0}
        terms = [spex.FermionTerm([], [], 5.0)]
        result = spex.expectation_value_fermionic(phi, psi, terms)
        assertComplexAlmostEqual(result, 5.0 + 0.0j)

    def test_complex_constant_term(self):
        # ⟨ψ| (3+4j)·I |ψ⟩ = 3+4j
        phi = {0b01: 1.0}
        psi = {0b01: 1.0}
        terms = [spex.FermionTerm([], [], 3.0 + 4.0j)]
        result = spex.expectation_value_fermionic(phi, psi, terms)
        assertComplexAlmostEqual(result, 3.0 + 4.0j)


S2 = 1.0 / math.sqrt(2.0)
PI_2 = math.pi / 2.0
PI = math.pi


class TestAsymmetricExcitation(unittest.TestCase):
    """Generators with |creation_idx| ≠ |annihilation_idx|"""

    def test_pair_creation_full(self):
        # G = i(a†_0 a†_1 - a_1 a_0), |00⟩, θ=π → -|11⟩
        res = spex.apply_fermion_excitation(
            {0: 1.0}, spex.FermionTerm([0, 1], [], 1.0j), PI)
        assertStateDictsAlmostEqual(res, {3: -1.0})

    def test_pair_creation_half(self):
        # |00⟩, θ=π/2 → cos|00⟩ + sin(-|11⟩) = S2|00⟩ - S2|11⟩
        res = spex.apply_fermion_excitation(
            {0: 1.0}, spex.FermionTerm([0, 1], [], 1.0j), PI_2)
        assertStateDictsAlmostEqual(res, {0: S2, 3: -S2})

    def test_pair_annihilation_full(self):
        # G = i(a_1 a_0 - a†_0 a†_1), |11⟩, θ=π → |00⟩
        res = spex.apply_fermion_excitation(
            {3: 1.0}, spex.FermionTerm([], [0, 1], 1.0j), PI)
        assertStateDictsAlmostEqual(res, {0: 1.0})

    def test_pair_annihilation_half(self):
        # |11⟩, θ=π/2 → S2|11⟩ + S2|00⟩
        res = spex.apply_fermion_excitation(
            {3: 1.0}, spex.FermionTerm([], [0, 1], 1.0j), PI_2)
        assertStateDictsAlmostEqual(res, {3: S2, 0: S2})

    def test_single_creation_full(self):
        # G = i(a†_0 - a_0), |0⟩, θ=π → |1⟩
        res = spex.apply_fermion_excitation(
            {0: 1.0}, spex.FermionTerm([0], [], 1.0j), PI)
        assertStateDictsAlmostEqual(res, {1: 1.0})

    def test_single_annihilation_full(self):
        # G = i(a_0 - a†_0), |1⟩, θ=π → |0⟩
        res = spex.apply_fermion_excitation(
            {1: 1.0}, spex.FermionTerm([], [0], 1.0j), PI)
        assertStateDictsAlmostEqual(res, {0: 1.0})


class TestComplexWeightExcitation(unittest.TestCase):
    """Generators with w ≠ 1j"""

    def test_real_weight(self):
        # G = a†_0 a_1 + a†_1 a_0  (w=1), |10⟩, θ=π
        res = spex.apply_fermion_excitation(
            {2: 1.0}, spex.FermionTerm([0], [1], 1.0), PI)
        assertStateDictsAlmostEqual(res, {1: -1.0j})

    def test_real_weight_half(self):
        res = spex.apply_fermion_excitation(
            {2: 1.0}, spex.FermionTerm([0], [1], 1.0), 1.0)
        expected_orig = complex(math.cos(0.5), 0.0)
        expected_partner = complex(0.0, -math.sin(0.5))
        assertStateDictsAlmostEqual(res, {2: expected_orig, 1: expected_partner})

    def test_double_weight(self):
        # w=2j, |w|=2, |10⟩, θ=π/2 → full transfer → |01⟩
        res = spex.apply_fermion_excitation(
            {2: 1.0}, spex.FermionTerm([0], [1], 2.0j), PI_2)
        assertStateDictsAlmostEqual(res, {1: 1.0})

    def test_negative_weight(self):
        res = spex.apply_fermion_excitation(
            {2: 1.0}, spex.FermionTerm([0], [1], -1.0j), PI)
        assertStateDictsAlmostEqual(res, {1: -1.0})


class TestEdgeCases(unittest.TestCase):
    """Edge cases for apply_fermion_excitation"""

    def test_zero_weight(self):
        res = spex.apply_fermion_excitation(
            {2: 1.0}, spex.FermionTerm([0], [1], 0.0), PI)
        assertStateDictsAlmostEqual(res, {2: 1.0})

    def test_zero_theta(self):
        res = spex.apply_fermion_excitation(
            {2: 1.0}, spex.FermionTerm([0], [1], 1.0j), 0.0)
        assertStateDictsAlmostEqual(res, {2: 1.0})

    def test_nullspace_vacuum(self):
        res = spex.apply_fermion_excitation(
            {0: 1.0}, spex.FermionTerm([0], [1], 1.0j), PI)
        assertStateDictsAlmostEqual(res, {0: 1.0})

    def test_nullspace_fully_occupied(self):
        res = spex.apply_fermion_excitation(
            {3: 1.0}, spex.FermionTerm([0], [1], 1.0j), PI)
        assertStateDictsAlmostEqual(res, {3: 1.0})

    def test_self_pair_nullspace(self):
        res = spex.apply_fermion_excitation(
            {1: 1.0}, spex.FermionTerm([0], [0], 1.0j), PI)
        assertStateDictsAlmostEqual(res, {1: 1.0})

    def test_empty_state_raises(self):
        with self.assertRaises(Exception):
            spex.apply_fermion_excitation(
                {}, spex.FermionTerm([0], [1], 1.0j), PI)


if __name__ == "__main__":
    unittest.main(verbosity=2)
