import math
import unittest

import spex_tequila as spex

S2 = 1.0 / math.sqrt(2.0)  # ~0.7071067811865476
PI = math.pi
PI_2 = math.pi / 2.0
PI_4 = math.pi / 4.0


def fock(*orbitals):
    """Map orbital indices -> integer bitmask (bitstring).

    e.g., fock(0, 2) -> 0b...00101
    """
    return sum(1 << orb for orb in orbitals)


class TestAbstractGenerator(unittest.TestCase):
    """Unit tests for `apply_abstract_generator`.

    `apply_abstract_generator` applies a sum of fermionic excitation terms
    exp(-i*theta * sum_k (w_k A_k + conj(w_k) A_k^dag)) to a state, where the
    terms are assumed to act on mutually orthogonal orbitals. Under that
    constraint it is implemented as the sequential application of
    `apply_fermion_excitation` for each term.
    """

    def assertStateDictsAlmostEqual(self, actual_dict, expected_dict, places=11):
        """Assert two sparse state dicts (int basis -> complex coeff) nearly equal."""
        atol = 10 ** (-places)
        all_keys = set(actual_dict.keys()) | set(expected_dict.keys())

        for key in all_keys:
            val_actual = complex(actual_dict.get(key, 0.0 + 0.0j))
            val_expected = complex(expected_dict.get(key, 0.0 + 0.0j))
            diff = abs(val_actual - val_expected)

            if diff > atol:
                act_str = ", ".join(f"{k}: {v}" for k, v in actual_dict.items())
                exp_str = ", ".join(f"{k}: {v}" for k, v in expected_dict.items())
                self.fail(
                    f"\nState Vector mismatch at int basis |{key}> (0b{key:b}):\n"
                    f"  Actual   : {val_actual}\n"
                    f"  Expected : {val_expected}\n"
                    f"  Diff     : {diff:.2e} (Tolerance: 1e-{places})\n"
                    f"  Full Act : {{{act_str}}}\n"
                    f"  Full Exp : {{{exp_str}}}"
                )

    # ------------------------------------------------------------------
    # Group 1: Input validation / degenerate inputs
    # ------------------------------------------------------------------

    def test_empty_state_raises_error(self):
        """Passing an empty state with a non-empty term list must raise."""
        term = spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=1.0j)

        with self.assertRaises(ValueError):
            spex.apply_abstract_generator({}, [term], PI_2)

    def test_empty_state_without_terms_is_noop(self):
        """An empty state with an empty term list is a no-op (no raise)."""
        result = spex.apply_abstract_generator({}, [], PI_2)
        self.assertEqual(result, {})

    def test_empty_terms_is_noop(self):
        """An empty term list leaves the state unchanged."""
        state = {fock(0, 1): 1.0 + 0.0j}
        result = spex.apply_abstract_generator(state, [], PI_2)
        self.assertStateDictsAlmostEqual(result, state)

    def test_zero_theta_returns_original_state(self):
        """A rotation by theta = 0 leaves the state unchanged."""
        state = {fock(1): 1.0 + 0.0j}
        term = spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=1.0j)

        result = spex.apply_abstract_generator(state, [term], 0.0)
        self.assertStateDictsAlmostEqual(result, state)

    def test_zero_weight_returns_original_state(self):
        """A term with weight = 0 leaves the state unchanged."""
        state = {fock(1): 1.0 + 0.0j}
        term = spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=0.0j)

        result = spex.apply_abstract_generator(state, [term], PI_2)
        self.assertStateDictsAlmostEqual(result, state)

    # ------------------------------------------------------------------
    # Group 2: Consistency with apply_fermion_excitation
    # ------------------------------------------------------------------

    def test_single_term_matches_fermion_excitation(self):
        """One term through the generator must equal apply_fermion_excitation."""
        state = {fock(0): 0.8, fock(1): 0.6}
        term = spex.FermionTerm(creation_idx=[0], annihilation_idx=[2], weight=1.0j)

        for theta in (0.1, PI_4, PI_2, 1.3, -PI_2):
            expected = spex.apply_fermion_excitation(state, term, theta)
            result = spex.apply_abstract_generator(state, [term], theta)
            self.assertStateDictsAlmostEqual(result, expected)

    def test_orthogonal_terms_match_sequential(self):
        """Disjoint-orbital terms must match sequential application exactly."""
        state = {fock(0, 1): 1.0 + 0.0j}
        term1 = spex.FermionTerm(creation_idx=[0], annihilation_idx=[2], weight=1.0j)
        term2 = spex.FermionTerm(creation_idx=[1], annihilation_idx=[3], weight=1.0j)

        theta = PI_4
        expected = spex.apply_fermion_excitation(state, term1, theta)
        expected = spex.apply_fermion_excitation(expected, term2, theta)

        result = spex.apply_abstract_generator(state, [term1, term2], theta)
        self.assertStateDictsAlmostEqual(result, expected)

    def test_three_terms_match_sequential(self):
        """Three pairwise-disjoint terms match sequential application."""
        state = {fock(0, 1, 2): 1.0 + 0.0j}
        terms = [
            spex.FermionTerm(creation_idx=[0], annihilation_idx=[3], weight=1.0j),
            spex.FermionTerm(creation_idx=[1], annihilation_idx=[4], weight=1.0j),
            spex.FermionTerm(creation_idx=[2], annihilation_idx=[5], weight=1.0j),
        ]

        expected = state
        for term in terms:
            expected = spex.apply_fermion_excitation(expected, term, PI_4)

        result = spex.apply_abstract_generator(state, terms, PI_4)
        self.assertStateDictsAlmostEqual(result, expected)

    def test_superposition_input_matches_sequential(self):
        """Superposition input is handled identically to sequential application."""
        state = {fock(0): 0.8, fock(1): 0.6}
        term1 = spex.FermionTerm(creation_idx=[0], annihilation_idx=[2], weight=1.0j)
        term2 = spex.FermionTerm(creation_idx=[1], annihilation_idx=[3], weight=1.0j)

        theta = 0.7
        expected = spex.apply_fermion_excitation(state, term1, theta)
        expected = spex.apply_fermion_excitation(expected, term2, theta)

        result = spex.apply_abstract_generator(state, [term1, term2], theta)
        self.assertStateDictsAlmostEqual(result, expected)

    # ------------------------------------------------------------------
    # Group 3: Known analytic results
    # ------------------------------------------------------------------

    def test_single_excitation_50_50_split(self):
        """0 -> 1 at pi/2 gives the balanced superposition (|0> - |1>)/sqrt(2)."""
        result = spex.apply_abstract_generator(
            {fock(0): 1.0},
            [spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=1.0j)],
            PI_2,
        )
        self.assertStateDictsAlmostEqual(result, {fock(0): S2, fock(1): -S2})

    def test_single_excitation_full_transfer(self):
        """0 -> 1 at pi gives full population transfer with a -1 phase."""
        result = spex.apply_abstract_generator(
            {fock(0): 1.0},
            [spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=1.0j)],
            PI,
        )
        self.assertStateDictsAlmostEqual(result, {fock(1): -1.0})

    def test_orthogonal_full_transfer(self):
        """Two orthogonal excitations at pi fully transfer both particles."""
        state = {fock(0, 1): 1.0}
        term1 = spex.FermionTerm(creation_idx=[0], annihilation_idx=[2], weight=1.0j)
        term2 = spex.FermionTerm(creation_idx=[1], annihilation_idx=[3], weight=1.0j)

        result = spex.apply_abstract_generator(state, [term1, term2], PI)
        self.assertStateDictsAlmostEqual(result, {fock(2, 3): 1.0})

    # ------------------------------------------------------------------
    # Group 4: Nullspace (terms that cannot act)
    # ------------------------------------------------------------------

    def test_nullspace_vacuum(self):
        """Annihilating from the vacuum is a no-op (term lies in the nullspace)."""
        state = {fock(): 1.0}  # vacuum
        term = spex.FermionTerm(creation_idx=[2], annihilation_idx=[0], weight=1.0j)

        result = spex.apply_abstract_generator(state, [term], PI)
        self.assertStateDictsAlmostEqual(result, state)

    def test_nullspace_pauli_blocking(self):
        """Creating into an already-occupied orbital is blocked (no-op)."""
        state = {fock(0, 1): 1.0}  # orbitals 0 and 1 occupied
        # a^dag_1 a_0 tries to create in occupied orbital 1 -> nullspace
        term = spex.FermionTerm(creation_idx=[1], annihilation_idx=[0], weight=1.0j)

        result = spex.apply_abstract_generator(state, [term], PI_2)
        self.assertStateDictsAlmostEqual(result, state)

    # ------------------------------------------------------------------
    # Group 5: Parity / Jordan-Wigner sign rules
    # ------------------------------------------------------------------

    def test_parity_jump_over_empty_orbital(self):
        """0 -> 2 over an empty orbital carries no parity sign flip."""
        result = spex.apply_abstract_generator(
            {fock(0): 1.0},
            [spex.FermionTerm(creation_idx=[0], annihilation_idx=[2], weight=1.0j)],
            PI_2,
        )
        self.assertStateDictsAlmostEqual(result, {fock(0): S2, fock(2): -S2})

    def test_parity_jump_over_occupied_orbital(self):
        """0 -> 2 over an occupied orbital flips sign (fermionic parity)."""
        result = spex.apply_abstract_generator(
            {fock(0, 1): 1.0},
            [spex.FermionTerm(creation_idx=[0], annihilation_idx=[2], weight=1.0j)],
            PI_2,
        )
        self.assertStateDictsAlmostEqual(result, {fock(0, 1): S2, fock(1, 2): S2})

    # ------------------------------------------------------------------
    # Group 6: Complex amplitudes, phases and weight scaling
    # ------------------------------------------------------------------

    def test_complex_input_amplitude_preserved(self):
        """A complex input amplitude is propagated through the rotation."""
        result = spex.apply_abstract_generator(
            {fock(0): 1.0j},
            [spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=1.0j)],
            PI_2,
        )
        self.assertStateDictsAlmostEqual(result, {fock(0): S2 * 1.0j, fock(1): -S2 * 1.0j})

    def test_negative_theta_reverses_rotation(self):
        """A negative theta reverses the sign of the transferred amplitude."""
        result = spex.apply_abstract_generator(
            {fock(0): 1.0},
            [spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=1.0j)],
            -PI_2,
        )
        self.assertStateDictsAlmostEqual(result, {fock(0): S2, fock(1): S2})

    def test_negative_weight_equivalent_to_negative_angle(self):
        """A negative imaginary weight equals the same rotation at -theta."""
        state = {fock(0): 1.0}
        result_neg_weight = spex.apply_abstract_generator(
            state, [spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=-1.0j)], PI_2
        )
        result_neg_theta = spex.apply_abstract_generator(
            state, [spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=1.0j)], -PI_2
        )
        self.assertStateDictsAlmostEqual(result_neg_weight, result_neg_theta)
        self.assertStateDictsAlmostEqual(result_neg_weight, {fock(0): S2, fock(1): S2})

    def test_weight_scales_rotation_angle(self):
        """A weight of c*1j scales the effective rotation angle to c*theta."""
        state = {fock(0): 1.0}
        result_scaled = spex.apply_abstract_generator(
            state, [spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=0.8j)], 0.5
        )
        result_rescaled = spex.apply_abstract_generator(
            state, [spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=1.0j)], 0.4
        )
        self.assertStateDictsAlmostEqual(result_scaled, result_rescaled)

    def test_complex_weight_matches_sequential(self):
        """A genuinely complex weight is handled identically to the single term."""
        state = {fock(0): 1.0}
        term = spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=0.6 + 0.8j)
        theta = 0.7

        result = spex.apply_abstract_generator(state, [term], theta)
        expected = spex.apply_fermion_excitation(state, term, theta)
        self.assertStateDictsAlmostEqual(result, expected)

    def test_unitarity_preserved(self):
        """The total norm is preserved for arbitrary weights and angles."""
        state = {fock(0): 0.6, fock(1): 0.8}
        norm = sum(abs(v) ** 2 for v in state.values())

        for weight in (1.0j, -0.5j, 0.3 + 0.4j, 1.0):
            for theta in (0.3, PI_4, PI_2, PI):
                term = spex.FermionTerm(creation_idx=[0], annihilation_idx=[2], weight=weight)
                result = spex.apply_abstract_generator(state, [term], theta)
                result_norm = sum(abs(v) ** 2 for v in result.values())
                self.assertAlmostEqual(result_norm, norm, places=10)

    # ------------------------------------------------------------------
    # Group 7: Non-orthogonal terms (documented sequential behavior)
    # ------------------------------------------------------------------

    def test_overlapping_terms_are_order_dependent(self):
        """Overlapping terms do not commute, so the order of application matters."""
        state = {fock(0): 1.0}
        term_a = spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=1.0j)
        term_b = spex.FermionTerm(creation_idx=[0], annihilation_idx=[2], weight=1.0j)

        forward = spex.apply_abstract_generator(state, [term_a, term_b], PI_2)
        reversed_ = spex.apply_abstract_generator(state, [term_b, term_a], PI_2)

        self.assertNotEqual(forward, reversed_)

        # Each ordering still equals the corresponding sequential application.
        seq_forward = spex.apply_fermion_excitation(state, term_a, PI_2)
        seq_forward = spex.apply_fermion_excitation(seq_forward, term_b, PI_2)
        self.assertStateDictsAlmostEqual(forward, seq_forward)


if __name__ == "__main__":
    unittest.main(verbosity=2)
