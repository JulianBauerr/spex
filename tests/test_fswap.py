import math
import unittest

import numpy as np

import spex_tequila as spex

S2 = 1.0 / math.sqrt(2.0)
PI_2 = math.pi / 2.0


def assertStateDictsAlmostEqual(actual, expected, places=11):
    atol = 10 ** (-places)
    keys = set(actual.keys()) | set(expected.keys())
    for key in keys:
        va = complex(actual.get(key, 0.0j))
        ve = complex(expected.get(key, 0.0j))
        diff = abs(va - ve)
        if diff > atol:
            raise AssertionError(
                f"Mismatch at |{key:04b}⟩: got {va}, expected {ve}, diff={diff:.2e}"
            )


class TestFSwap(unittest.TestCase):
    """Unit tests for apply_fswap"""

    def test_identity_same_index(self):
        """fSWAP on the same index leaves the state unchanged."""
        state = {0b1010: 1.0}
        result = spex.apply_fswap(state, 1, 1)
        assertStateDictsAlmostEqual(result, state)

    def test_00_unchanged(self):
        """|00⟩ in modes (0,1) stays |00⟩."""
        state = {0b00: 1.0}
        result = spex.apply_fswap(state, 0, 1)
        assertStateDictsAlmostEqual(result, state)

    def test_01_swapped(self):
        """|01⟩ → |10⟩ in modes (0,1)."""
        state = {0b01: 1.0}  # bit 0=1, bit 1=0
        result = spex.apply_fswap(state, 0, 1)
        assertStateDictsAlmostEqual(result, {0b10: 1.0})  # bit 0=0, bit 1=1

    def test_10_swapped(self):
        """|10⟩ → |01⟩ in modes (0,1)."""
        state = {0b10: 1.0}  # bit 0=0, bit 1=1
        result = spex.apply_fswap(state, 0, 1)
        assertStateDictsAlmostEqual(result, {0b01: 1.0})  # bit 0=1, bit 1=0

    def test_11_negative(self):
        """|11⟩ → -|11⟩ in modes (0,1)."""
        state = {0b11: 1.0}
        result = spex.apply_fswap(state, 0, 1)
        assertStateDictsAlmostEqual(result, {0b11: -1.0})

    def test_self_inverse(self):
        """Applying fSWAP twice recovers the original state."""
        state = {0b00: 1.0, 0b11: 1.0}
        once = spex.apply_fswap(state, 0, 2)
        twice = spex.apply_fswap(once, 0, 2)
        assertStateDictsAlmostEqual(twice, state)

    def test_higher_orbitals_unaffected(self):
        """Bits outside the swapped modes are unchanged."""
        state = {0b1101: 1.0}  # bits: 3=1,2=1,1=0,0=1
        result = spex.apply_fswap(state, 1, 2)  # swap bits 1,2
        # bit 1 (0) and bit 2 (1) differ → swap → bit 1=1, bit 2=0
        # bits 3 and 0 unchanged
        assertStateDictsAlmostEqual(result, {0b1011: 1.0})

    def test_superposition(self):
        """fSWAP on a superposition state."""
        state = {0b01: 1.0, 0b10: 1.0}  # |01⟩ + |10⟩
        result = spex.apply_fswap(state, 0, 1)
        # Both states just swap into each other
        assertStateDictsAlmostEqual(result, {0b10: 1.0, 0b01: 1.0})

    def test_superposition_with_phase(self):
        """fSWAP on a superposition with a |11⟩ component."""
        state = {0b01: 1.0, 0b11: 1.0}  # |01⟩ + |11⟩
        result = spex.apply_fswap(state, 0, 1)
        # |01⟩ → |10⟩, |11⟩ → -|11⟩
        assertStateDictsAlmostEqual(result, {0b10: 1.0, 0b11: -1.0})


class TestFSwapExcitationEquivalence(unittest.TestCase):
    """Cross-validate fSWAP by permuting excitation targets.

    Applying fSWAP(i,j), then an excitation k→j, then fSWAP(i,j) again
    should be equivalent to a direct excitation k→i.
    """

    def setUp(self):
        # Use a 4-qubit system with modes 1 and 2 swapped
        self.i, self.j, self.k = 1, 2, 0
        # Start with orbital 0 and 1 occupied (bits 0,1 set)
        self.initial = {0b0011: 1.0}
        self.theta = PI_2

    def _excitation(self, state, src, dst):
        """Apply excitation src→dst with w=1j."""
        return spex.apply_fermion_excitation(
            state, spex.FermionTerm([dst], [src], 1.0j), self.theta
        )

    def test_excitation_equivalence(self):
        """fSWAP(1,2) + excitation(0→2) + fSWAP(1,2) = excitation(0→1)."""
        # Case A: direct excitation 0→1
        expected = self._excitation(self.initial, self.k, self.i)

        # Case B: fSWAP(1,2) + excitation 0→2 + fSWAP(1,2)
        tmp = spex.apply_fswap(self.initial, self.i, self.j)
        tmp = self._excitation(tmp, self.k, self.j)
        result = spex.apply_fswap(tmp, self.i, self.j)

        assertStateDictsAlmostEqual(result, expected)

    def test_three_way_equivalence(self):
        """Same equivalence holds for other source-destination pairs."""
        # Use modes 0 and 2 swapped, excitation from 1→0 vs 1→2
        initial = {0b0111: 1.0}  # bits 0,1,2 occupied
        tmp = spex.apply_fswap(initial, 0, 2)
        tmp = self._excitation(tmp, 1, 2)  # k=1, j=2
        result = spex.apply_fswap(tmp, 0, 2)
        expected = self._excitation(initial, 1, 0)  # k=1, i=0
        assertStateDictsAlmostEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
