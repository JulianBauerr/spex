import numpy as np
import pytest
import tequila as tq

import spex_tequila as spex

from tests.helpers import assert_states_match, fock


@pytest.fixture(scope="module")
def h2_molecule():
    return tq.Molecule("H 0 0 0\nH 0 0 1", "sto-3g", units="angstrom")


def _tequila_circuit(molecule, init_orbitals, gates):
    U = tq.gates.X(list(init_orbitals))
    for i, j, angle in gates:
        U += molecule.make_excitation_gate(indices=[(i, j)], angle=angle)
    return tq.simulate(U)


class TestAbstractGeneratorTequila:

    @pytest.mark.parametrize("theta", [0.5, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3])
    def test_orthogonal_generator_match(self, h2_molecule, theta):
        terms = [spex.FermionTerm([0], [2], 1.0j), spex.FermionTerm([1], [3], 1.0j)]
        result = spex.apply_abstract_generator({fock(0, 1): 1.0}, terms, theta)
        wfn = _tequila_circuit(h2_molecule, [0, 1], [(0, 2, theta), (1, 3, theta)])
        assert_states_match(wfn, result)

    @pytest.mark.parametrize("theta", [0.5, np.pi / 2, np.pi])
    def test_single_term_matches_tequila(self, h2_molecule, theta):
        term = spex.FermionTerm([0], [1], 1.0j)
        result = spex.apply_abstract_generator({fock(0): 1.0}, [term], theta)
        wfn = _tequila_circuit(h2_molecule, [0], [(0, 1, theta)])
        assert_states_match(wfn, result)

    def test_scaled_imaginary_weights_match(self, h2_molecule):
        theta = 0.25
        terms = [spex.FermionTerm([0], [2], 0.8j), spex.FermionTerm([1], [3], -0.6j)]
        result = spex.apply_abstract_generator({fock(0, 1): 1.0}, terms, theta)
        wfn = _tequila_circuit(h2_molecule, [0, 1], [(0, 2, 0.8 * theta), (1, 3, -0.6 * theta)])
        assert_states_match(wfn, result)

    def test_parity_sign_over_occupied_orbital(self, h2_molecule):
        theta = np.pi / 2
        term = spex.FermionTerm([0], [2], 1.0j)
        result = spex.apply_abstract_generator({fock(0, 1): 1.0}, [term], theta)
        wfn = _tequila_circuit(h2_molecule, [0, 1], [(0, 2, theta)])
        assert_states_match(wfn, result)

    def test_double_excitation_match(self, h2_molecule):
        theta = 0.4
        term = spex.FermionTerm([0, 1], [2, 3], 1.0j)
        result = spex.apply_abstract_generator({fock(0, 1): 1.0}, [term], theta)
        # A single multi-orbital gate, with reversed pairs for the same sign.
        U = tq.gates.X([0, 1]) + h2_molecule.make_excitation_gate(
            indices=[(1, 3), (0, 2)], angle=theta
        )
        assert_states_match(tq.simulate(U), result)

    def test_full_transfer_two_terms(self, h2_molecule):
        terms = [spex.FermionTerm([0], [2], 1.0j), spex.FermionTerm([1], [3], 1.0j)]
        result = spex.apply_abstract_generator({fock(0, 1): 1.0}, terms, np.pi)
        wfn = _tequila_circuit(h2_molecule, [0, 1], [(0, 2, np.pi), (1, 3, np.pi)])
        assert_states_match(wfn, result)
