import cmath
import math
import unittest

import spex_tequila as spex

S2 = 1.0 / math.sqrt(2.0)  # ~0.7071067811865476
PI_2 = -math.pi / 2.0
PI = -math.pi


def fock(*orbitals):
    """Map orbital indices -> integer bitmask (bitstring).
    e.g., fock(0,2) -> 0b...00101
    """
    return sum(1 << orb for orb in orbitals)


class TestQuantumEngineeringFermionEngine(unittest.TestCase):
    """Tests for fermionic excitations"""

    def assertStateDictsAlmostEqual(
            self, actual_dict, expected_dict, places=11
    ):
        """Assert two sparse state dicts nearly equal."""
        atol = 10 ** (-places)
        all_keys = set(actual_dict.keys()) | set(expected_dict.keys())

        for key in all_keys:
            val_actual = complex(actual_dict.get(key, 0.0 + 0.0j))
            val_expected = complex(expected_dict.get(key, 0.0 + 0.0j))
            diff = abs(val_actual - val_expected)

            if diff > atol:
                act_str = ", ".join(f"{k}: {v}" for k, v in actual_dict.items())
                exp_str = ", ".join(
                    f"{k}: {v}" for k, v in expected_dict.items()
                )
                self.fail(
                    f"\nState Vector mismatch at int basis |{key}> (0b{key:b}):\n"
                    f"  Actual   : {val_actual}\n"
                    f"  Expected : {val_expected}\n"
                    f"  Diff     : {diff:.2e} (Tolerance: 1e-{places})\n"
                    f"  Full Act : {{{act_str}}}\n"
                    f"  Full Exp : {{{exp_str}}}"
                )

    # GROUP 1: Sanity

    def test_excite_from_empty_orbital(self):
        # |...0001> (orb0). try 1 -> 2
        res = spex.apply_fermion_excitation(
            {fock(0): 1.0}, [1], [2], 0.25
        )
        self.assertStateDictsAlmostEqual(res, {fock(0): 1.0})

    def test_pauli_blocking(self):
        # |...0011> (0,1). try 0 -> 1
        res = spex.apply_fermion_excitation(
            {fock(0, 1): 1.0}, [0], [1], PI_2
        )
        self.assertStateDictsAlmostEqual(res, {fock(0, 1): 1.0})

    def test_the_absolute_vacuum(self):
        # vacuum 0b0000
        res = spex.apply_fermion_excitation(
            {0: 1.0}, [0], [1], PI_2
        )
        self.assertStateDictsAlmostEqual(res, {0: 1.0})

    # GROUP 2: Single excitations

    def test_single_excitation_50_50_split(self):
        # 0 -> 1 at π/4 => (|01>+|10>)/√2
        res = spex.apply_fermion_excitation(
            {fock(0): 1.0}, [0], [1], PI_2
        )
        self.assertStateDictsAlmostEqual(
            res, {fock(0): S2, fock(1): S2}  # {1: S2, 2: S2}
        )

    def test_single_excitation_full_transfer(self):
        # 0 -> 1 at π/2 => full transfer
        res = spex.apply_fermion_excitation(
            {fock(0): 1.0}, [0], [1], PI
        )
        self.assertStateDictsAlmostEqual(
            res, {fock(1): 1.0}  # {2: 1.0}
        )

    def test_single_deexcitation(self):
        # 1 -> 0 reverse
        res = spex.apply_fermion_excitation(
            {fock(1): 1.0}, [0], [1], -PI_2
        )
        self.assertStateDictsAlmostEqual(res, {fock(1): S2, fock(0): S2})

    # GROUP 3: Parity (JW)

    def test_parity_long_jump_over_empty_orbital(self):
        # 0 -> 2 over empty 1 (parity +)
        res = spex.apply_fermion_excitation(
            {fock(0): 1.0}, [0], [2], PI_2
        )
        self.assertStateDictsAlmostEqual(res, {fock(0): S2, fock(2): S2})

    def test_parity_long_jump_over_occupied_orbital(self):
        # 0 -> 2 over occupied 1 (parity -)
        res = spex.apply_fermion_excitation(
            {fock(0, 1): 1.0}, [0], [2], PI_2
        )
        self.assertStateDictsAlmostEqual(
            res, {fock(0, 1): S2, fock(1, 2): -S2}  # {3: S2, 6: -S2}
        )

    # GROUP 4: Double excitations

    def test_double_excitation_standard(self):
        # (0,1) -> (2,3)
        res = spex.apply_fermion_excitation(
            {fock(0, 1): 1.0}, [0, 1], [2, 3], PI_2
        )
        self.assertStateDictsAlmostEqual(
            res, {fock(0, 1): S2, fock(2, 3): -S2}  # {3: S2, 12: S2}
        )

    def test_double_excitation_parity_trap(self):
        # (0,1) -> (2,4) with orb3 occupied (0b01011->0b11100)
        res = spex.apply_fermion_excitation(
            {fock(0, 1, 3): 1.0}, [0, 1], [2, 4], PI_2
        )
        self.assertStateDictsAlmostEqual(
            res, {fock(0, 1, 3): S2, fock(2, 3, 4): S2}
        )

    # GROUP 5: Superposition & interference

    def test_superposition_partial_branch_hit(self):
        # 0.8|001> + 0.6|010>, swap 0 -> 2
        initial = {fock(0): 0.8, fock(1): 0.6}
        res = spex.apply_fermion_excitation(
            initial, [0], [2], PI
        )
        self.assertStateDictsAlmostEqual(res, {fock(2): 0.8, fock(1): 0.6})

    def test_destructive_quantum_interference(self):
        # (|01>+|10>)/√2, rotate 0->1 -> |10>
        initial = {fock(0): S2, fock(1): S2}
        res = spex.apply_fermion_excitation(
            initial, [0], [1], PI_2
        )
        self.assertStateDictsAlmostEqual(res, {fock(1): 1.0})

    # GROUP 6: Complex amplitudes

    def test_complex_phase_preservation(self):
        res = spex.apply_fermion_excitation({fock(0): 1.0j}, [0], [1], PI_2)
        self.assertStateDictsAlmostEqual(
            res, {fock(0): S2 * 1.0j, fock(1): S2 * 1.0j}
        )

    def test_complex_interference(self):
        initial = {fock(0): S2, fock(1): S2 * 1.0j}
        res = spex.apply_fermion_excitation(
            initial, [0], [1], PI_2
        )
        self.assertStateDictsAlmostEqual(
            res, {fock(0): (0.5 - 0.5j), fock(1): (0.5 + 0.5j)}
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)