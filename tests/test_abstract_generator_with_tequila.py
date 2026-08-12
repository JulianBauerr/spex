import math
import warnings

import numpy as np
import pytest
import tequila as tq

import spex_tequila as spex


def assert_states_match(tequila_wfn, spex_state, atol=1e-7):
    """Assert that a Tequila wavefunction and a Spex state dict agree on all amplitudes.

    Compares the union of basis states in both directions, treating missing
    entries as zero amplitude.
    """
    tq_state = {int(k): complex(v) for k, v in tequila_wfn.items()}
    all_keys = set(tq_state) | set(spex_state)

    for basis in all_keys:
        tq_amp = tq_state.get(basis, 0.0j)
        spx_amp = spex_state.get(basis, 0.0j)
        assert abs(tq_amp - spx_amp) <= atol, (
            f"Amplitude mismatch at basis |{basis}> (0b{basis:b}): "
            f"Spex={spx_amp}, Tequila={tq_amp}"
        )


@pytest.fixture(scope="module")
def h2_molecule():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tq.Molecule("H 0 0 0\nH 0 0 1", "sto-3g")


def _tequila_circuit(molecule, init_orbitals, gates):
    """Simulate X(init_orbitals) followed by a list of (i, j, angle) excitation gates."""
    U = tq.gates.X(list(init_orbitals))
    for i, j, angle in gates:
        U += molecule.make_excitation_gate(indices=[(i, j)], angle=angle)
    return tq.simulate(U)


class TestAbstractGeneratorTequila:
    """Cross-validate `apply_abstract_generator` against Tequila circuits.

    Spex's `FermionTerm(creation_idx=[i], annihilation_idx=[j], weight=1.0j)`
    corresponds to Tequila's `FermionicExcitation(target=(i, j))`, i.e. the
    excitation i -> j, at the same angle theta.
    """

    @pytest.mark.parametrize("theta", [0.5, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3])
    def test_orthogonal_generator_match(self, h2_molecule, theta):
        """Two disjoint single excitations applied as a sum must match Tequila."""
        # HF-like state: orbitals 0 and 1 occupied -> |...0011>
        spex_state = {0b0011: 1.0 + 0.0j}

        term1 = spex.FermionTerm(creation_idx=[0], annihilation_idx=[2], weight=1.0j)
        term2 = spex.FermionTerm(creation_idx=[1], annihilation_idx=[3], weight=1.0j)

        final_spex_state = spex.apply_abstract_generator(spex_state, [term1, term2], theta)

        tequila_wfn = _tequila_circuit(
            h2_molecule, [0, 1], [(0, 2, theta), (1, 3, theta)]
        )

        assert_states_match(tequila_wfn, final_spex_state)

    @pytest.mark.parametrize("theta", [0.5, np.pi / 2, np.pi])
    def test_single_term_matches_tequila(self, h2_molecule, theta):
        """A single-term generator reduces to one Tequila excitation gate."""
        spex_state = {0b0001: 1.0 + 0.0j}  # orbital 0 occupied

        term = spex.FermionTerm(creation_idx=[0], annihilation_idx=[1], weight=1.0j)
        final_spex_state = spex.apply_abstract_generator(spex_state, [term], theta)

        tequila_wfn = _tequila_circuit(h2_molecule, [0], [(0, 1, theta)])

        assert_states_match(tequila_wfn, final_spex_state)

    def test_scaled_imaginary_weights_match(self, h2_molecule):
        """Non-unit imaginary weights scale the rotation angle (|w| * theta)."""
        theta = 0.25
        weight1 = 0.8j
        weight2 = -0.6j

        spex_state = {0b0011: 1.0 + 0.0j}

        term1 = spex.FermionTerm(creation_idx=[0], annihilation_idx=[2], weight=weight1)
        term2 = spex.FermionTerm(creation_idx=[1], annihilation_idx=[3], weight=weight2)

        final_spex_state = spex.apply_abstract_generator(spex_state, [term1, term2], theta)

        # A negative imaginary weight corresponds to a negated rotation angle.
        tequila_wfn = _tequila_circuit(
            h2_molecule, [0, 1], [(0, 2, 0.8 * theta), (1, 3, -0.6 * theta)]
        )

        assert_states_match(tequila_wfn, final_spex_state)

    def test_parity_sign_over_occupied_orbital(self, h2_molecule):
        """Excitation 0 -> 2 jumping over occupied orbital 1 flips the sign."""
        theta = np.pi / 2
        spex_state = {0b0011: 1.0 + 0.0j}  # orbitals 0 and 1 occupied

        term = spex.FermionTerm(creation_idx=[0], annihilation_idx=[2], weight=1.0j)
        final_spex_state = spex.apply_abstract_generator(spex_state, [term], theta)

        tequila_wfn = _tequila_circuit(h2_molecule, [0, 1], [(0, 2, theta)])

        assert_states_match(tequila_wfn, final_spex_state)

    def test_double_excitation_match(self, h2_molecule):
        """A multi-orbital (double) excitation term is forwarded correctly."""
        theta = 0.4
        spex_state = {0b0011: 1.0 + 0.0j}  # orbitals 0 and 1 occupied

        # (0, 1) -> (2, 3)
        term = spex.FermionTerm(creation_idx=[0, 1], annihilation_idx=[2, 3], weight=1.0j)
        final_spex_state = spex.apply_abstract_generator(spex_state, [term], theta)

        # A single multi-orbital gate (both index pairs in one call), in the
        # order Tequila expects for the same sign.
        U = tq.gates.X([0, 1]) + h2_molecule.make_excitation_gate(
            indices=[(1, 3), (0, 2)], angle=theta
        )
        tequila_wfn = tq.simulate(U)

        assert_states_match(tequila_wfn, final_spex_state)

    def test_full_transfer_two_terms(self, h2_molecule):
        """Two orthogonal pi-rotations fully transfer both particles."""
        spex_state = {0b0011: 1.0 + 0.0j}

        term1 = spex.FermionTerm(creation_idx=[0], annihilation_idx=[2], weight=1.0j)
        term2 = spex.FermionTerm(creation_idx=[1], annihilation_idx=[3], weight=1.0j)

        final_spex_state = spex.apply_abstract_generator(spex_state, [term1, term2], np.pi)

        tequila_wfn = _tequila_circuit(
            h2_molecule, [0, 1], [(0, 2, np.pi), (1, 3, np.pi)]
        )

        assert_states_match(tequila_wfn, final_spex_state)
