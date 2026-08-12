import unittest
import spex_tequila as spex

from tests.helpers import S2, PI, PI_2, PI_4, assert_state_dicts_almost_equal, fock


class TestAbstractGenerator(unittest.TestCase):

    def test_empty_state_raises_error(self):
        term = spex.FermionTerm([0], [1], 1.0j)
        with self.assertRaises(ValueError):
            spex.apply_abstract_generator({}, [term], PI_2)

    def test_empty_state_without_terms_is_noop(self):
        self.assertEqual(spex.apply_abstract_generator({}, [], PI_2), {})

    def test_empty_terms_is_noop(self):
        state = {fock(0, 1): 1.0}
        assert_state_dicts_almost_equal(spex.apply_abstract_generator(state, [], PI_2), state)

    def test_zero_theta_returns_original_state(self):
        state = {fock(1): 1.0}
        term = spex.FermionTerm([0], [1], 1.0j)
        assert_state_dicts_almost_equal(spex.apply_abstract_generator(state, [term], 0.0), state)

    def test_zero_weight_returns_original_state(self):
        state = {fock(1): 1.0}
        term = spex.FermionTerm([0], [1], 0.0j)
        assert_state_dicts_almost_equal(spex.apply_abstract_generator(state, [term], PI_2), state)

    def test_single_term_matches_fermion_excitation(self):
        state = {fock(0): 0.8, fock(1): 0.6}
        term = spex.FermionTerm([0], [2], 1.0j)
        for theta in (0.1, PI_4, PI_2, 1.3, -PI_2):
            expected = spex.apply_fermion_excitation(state, term, theta)
            assert_state_dicts_almost_equal(
                spex.apply_abstract_generator(state, [term], theta), expected
            )

    def test_orthogonal_terms_match_sequential(self):
        state = {fock(0, 1): 1.0}
        terms = [spex.FermionTerm([0], [2], 1.0j), spex.FermionTerm([1], [3], 1.0j)]

        expected = state
        for term in terms:
            expected = spex.apply_fermion_excitation(expected, term, PI_4)

        assert_state_dicts_almost_equal(
            spex.apply_abstract_generator(state, terms, PI_4), expected
        )

    def test_three_terms_match_sequential(self):
        state = {fock(0, 1, 2): 1.0}
        terms = [
            spex.FermionTerm([0], [3], 1.0j),
            spex.FermionTerm([1], [4], 1.0j),
            spex.FermionTerm([2], [5], 1.0j),
        ]

        expected = state
        for term in terms:
            expected = spex.apply_fermion_excitation(expected, term, PI_4)

        assert_state_dicts_almost_equal(
            spex.apply_abstract_generator(state, terms, PI_4), expected
        )

    def test_superposition_input_matches_sequential(self):
        state = {fock(0): 0.8, fock(1): 0.6}
        terms = [spex.FermionTerm([0], [2], 1.0j), spex.FermionTerm([1], [3], 1.0j)]

        expected = state
        for term in terms:
            expected = spex.apply_fermion_excitation(expected, term, 0.7)

        assert_state_dicts_almost_equal(
            spex.apply_abstract_generator(state, terms, 0.7), expected
        )

    def test_single_excitation_50_50_split(self):
        result = spex.apply_abstract_generator(
            {fock(0): 1.0}, [spex.FermionTerm([0], [1], 1.0j)], PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0): S2, fock(1): -S2})

    def test_single_excitation_full_transfer(self):
        result = spex.apply_abstract_generator(
            {fock(0): 1.0}, [spex.FermionTerm([0], [1], 1.0j)], PI
        )
        assert_state_dicts_almost_equal(result, {fock(1): -1.0})

    def test_orthogonal_full_transfer(self):
        terms = [spex.FermionTerm([0], [2], 1.0j), spex.FermionTerm([1], [3], 1.0j)]
        result = spex.apply_abstract_generator({fock(0, 1): 1.0}, terms, PI)
        assert_state_dicts_almost_equal(result, {fock(2, 3): 1.0})

    def test_nullspace_vacuum(self):
        state = {fock(): 1.0}
        result = spex.apply_abstract_generator(state, [spex.FermionTerm([2], [0], 1.0j)], PI)
        assert_state_dicts_almost_equal(result, state)

    def test_nullspace_pauli_blocking(self):
        state = {fock(0, 1): 1.0}
        result = spex.apply_abstract_generator(state, [spex.FermionTerm([1], [0], 1.0j)], PI_2)
        assert_state_dicts_almost_equal(result, state)

    def test_parity_jump_over_empty_orbital(self):
        result = spex.apply_abstract_generator(
            {fock(0): 1.0}, [spex.FermionTerm([0], [2], 1.0j)], PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0): S2, fock(2): -S2})

    def test_parity_jump_over_occupied_orbital(self):
        result = spex.apply_abstract_generator(
            {fock(0, 1): 1.0}, [spex.FermionTerm([0], [2], 1.0j)], PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0, 1): S2, fock(1, 2): S2})

    def test_complex_input_amplitude_preserved(self):
        result = spex.apply_abstract_generator(
            {fock(0): 1.0j}, [spex.FermionTerm([0], [1], 1.0j)], PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0): S2 * 1.0j, fock(1): -S2 * 1.0j})

    def test_negative_theta_reverses_rotation(self):
        result = spex.apply_abstract_generator(
            {fock(0): 1.0}, [spex.FermionTerm([0], [1], 1.0j)], -PI_2
        )
        assert_state_dicts_almost_equal(result, {fock(0): S2, fock(1): S2})

    def test_negative_weight_equivalent_to_negative_angle(self):
        state = {fock(0): 1.0}
        neg_weight = spex.apply_abstract_generator(
            state, [spex.FermionTerm([0], [1], -1.0j)], PI_2
        )
        neg_theta = spex.apply_abstract_generator(
            state, [spex.FermionTerm([0], [1], 1.0j)], -PI_2
        )
        assert_state_dicts_almost_equal(neg_weight, neg_theta)

    def test_weight_scales_rotation_angle(self):
        state = {fock(0): 1.0}
        scaled = spex.apply_abstract_generator(
            state, [spex.FermionTerm([0], [1], 0.8j)], 0.5
        )
        rescaled = spex.apply_abstract_generator(
            state, [spex.FermionTerm([0], [1], 1.0j)], 0.4
        )
        assert_state_dicts_almost_equal(scaled, rescaled)

    def test_complex_weight_matches_sequential(self):
        state = {fock(0): 1.0}
        term = spex.FermionTerm([0], [1], 0.6 + 0.8j)
        expected = spex.apply_fermion_excitation(state, term, 0.7)
        assert_state_dicts_almost_equal(
            spex.apply_abstract_generator(state, [term], 0.7), expected
        )

    def test_unitarity_preserved(self):
        state = {fock(0): 0.6, fock(1): 0.8}
        norm = sum(abs(v) ** 2 for v in state.values())
        for weight in (1.0j, -0.5j, 0.3 + 0.4j, 1.0):
            for theta in (0.3, PI_4, PI_2, PI):
                result = spex.apply_abstract_generator(
                    state, [spex.FermionTerm([0], [2], weight)], theta
                )
                result_norm = sum(abs(v) ** 2 for v in result.values())
                self.assertAlmostEqual(result_norm, norm, places=10)

    def test_overlapping_terms_are_order_dependent(self):
        state = {fock(0): 1.0}
        term_a = spex.FermionTerm([0], [1], 1.0j)
        term_b = spex.FermionTerm([0], [2], 1.0j)

        forward = spex.apply_abstract_generator(state, [term_a, term_b], PI_2)
        reversed_ = spex.apply_abstract_generator(state, [term_b, term_a], PI_2)

        self.assertNotEqual(forward, reversed_)

        sequential = spex.apply_fermion_excitation(state, term_a, PI_2)
        sequential = spex.apply_fermion_excitation(sequential, term_b, PI_2)
        assert_state_dicts_almost_equal(forward, sequential)
