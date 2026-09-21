import numpy as np
import pytest
import tequila as tq

import spex_tequila as spex

from tests.helpers import assert_states_match, fock


@pytest.fixture(scope="module")
def h2_molecule():
    return tq.Molecule("H 0 0 0\nH 0 0 1", "sto-3g", units="angstrom")


@pytest.fixture(scope="module")
def big_molecule():
    return tq.Molecule("\n".join(f"H 0 0 {i}" for i in range(14)), "sto-3g", units="angstrom")


class TestSpexExcitations:
    @pytest.mark.parametrize("theta", [np.pi, np.pi / 2, np.pi / 4, -np.pi / 3])
    def test_fermion_excitation_adjacent(self, h2_molecule, theta):
        fe = h2_molecule.make_excitation_gate(indices=[(0, 1)], angle="a")
        tq_wfn = tq.simulate(tq.gates.X(0) + fe, variables={"a": theta})

        result = spex.apply_fermion_excitation(
            {fock(0): 1.0}, spex.FermionTerm([0], [1], 1.0j), theta
        )

        assert_states_match(tq_wfn, result)

    @pytest.mark.parametrize("theta", [np.pi, np.pi / 2])
    def test_fermion_excitation_jump_occupied(self, h2_molecule, theta):
        fe = h2_molecule.make_excitation_gate(indices=[(0, 2)], angle="a")
        tq_wfn = tq.simulate(tq.gates.X([0, 1]) + fe, variables={"a": theta})

        result = spex.apply_fermion_excitation(
            {fock(0, 1): 1.0}, spex.FermionTerm([0], [2], 1.0j), theta
        )

        assert_states_match(tq_wfn, result)

    def test_fermion_excitation_jump_empty(self, h2_molecule):
        theta = np.pi / 2
        fe = h2_molecule.make_excitation_gate(indices=[(0, 2)], angle="a")
        tq_wfn = tq.simulate(tq.gates.X(0) + fe, variables={"a": theta})

        result = spex.apply_fermion_excitation(
            {fock(0): 1.0}, spex.FermionTerm([0], [2], 1.0j), theta
        )

        assert_states_match(tq_wfn, result)

    @pytest.mark.parametrize("theta", [np.pi / 4, np.pi / 2])
    def test_multi_fermion_excitation_standard(self, h2_molecule, theta):
        fe = h2_molecule.make_excitation_gate(indices=[(1, 3), (0, 2)], angle="a")
        tq_wfn = tq.simulate(tq.gates.X([0, 1]) + fe, variables={"a": theta})

        result = spex.apply_fermion_excitation(
            {fock(0, 1): 1.0}, spex.FermionTerm([0, 1], [2, 3], 1.0j), theta
        )

        assert_states_match(tq_wfn, result)

    @pytest.mark.parametrize("theta", [np.pi / 4, np.pi / 2])
    def test_multi_fermion_excitation_parity_trap(self, big_molecule, theta):
        fe = big_molecule.make_excitation_gate(indices=[(1, 4), (0, 2)], angle="a")
        tq_wfn = tq.simulate(tq.gates.X([0, 1, 3]) + fe, variables={"a": theta})

        result = spex.apply_fermion_excitation(
            {fock(0, 1, 3): 1.0}, spex.FermionTerm([0, 1], [2, 4], 1.0j), theta
        )

        assert_states_match(tq_wfn, result)

    @pytest.mark.parametrize("theta", [np.pi / 2, np.pi / 4])
    def test_excitation_to_already_occupied(self, h2_molecule, theta):
        fe = h2_molecule.make_excitation_gate(indices=[(0, 2)], angle="a")
        tq_wfn = tq.simulate(tq.gates.X([0, 2]) + fe, variables={"a": theta})

        result = spex.apply_fermion_excitation(
            {fock(0, 2): 1.0}, spex.FermionTerm([0], [2], 1.0j), theta
        )

        assert_states_match(tq_wfn, result)

    @pytest.mark.parametrize("theta", [np.pi / 4, np.pi / 2])
    def test_partial_occupation_multi_excitation(self, h2_molecule, theta):
        fe = h2_molecule.make_excitation_gate(indices=[(1, 3), (0, 2)], angle="a")
        tq_wfn = tq.simulate(tq.gates.X([0, 1, 2]) + fe, variables={"a": theta})

        result = spex.apply_fermion_excitation(
            {fock(0, 1, 2): 1.0}, spex.FermionTerm([0, 1], [2, 3], 1.0j), theta
        )

        assert_states_match(tq_wfn, result)

    def test_number_excitations_self_pairs(self, h2_molecule):
        theta = np.pi / 2
        for initial, term in [
            ({fock(0): 1.0}, spex.FermionTerm([0], [0], 1.0j)),
            ({fock(0, 1): 1.0}, spex.FermionTerm([0, 1], [0, 1], 1.0j)),
            ({fock(0, 1): 1.0}, spex.FermionTerm([0, 1], [2, 1], 1.0j)),
        ]:
            result = spex.apply_fermion_excitation(initial, term, theta)
            assert_states_match(initial, result)
