import spex_tequila as spex
import pytest
import tequila as tq
import numpy as np


def assert_states_match(tequila_wfn, spex_state, atol=1e-5):
    tq_state = dict(tequila_wfn.items())

    # Check amplitudes
    for basis_state, tq_amp in tq_state.items():
        spex_amp = spex_state.get(basis_state, 0.0)
        assert np.isclose(spex_amp, np.real(tq_amp), atol=atol), \
            f"Amplitude mismatch at basis state {basis_state}: Spex={spex_amp}, Tequila={np.real(tq_amp)}"

    # Check extra states
    for basis_state, spex_amp in spex_state.items():
        if abs(spex_amp) > atol:
            tq_amp = tq_state.get(basis_state, 0.0)
            assert np.isclose(spex_amp, np.real(tq_amp), atol=atol), \
                f"Spex produced unexpected amplitude at {basis_state}: {spex_amp}"

@pytest.fixture
def big_molecule():
    """14H -> 28 spin-orbitals"""
    return tq.Molecule("\n".join([f"H 0 0 {i}" for i in range(14)]), "sto-3g")

def str_to_int(bitstring):
    """reverse bitstring -> int"""
    return int(bitstring[::-1], 2)

@pytest.fixture
def h2_molecule():
    return tq.Molecule("H 0 0 0\nH 0 0 1", "sto-3g")

class TestSpexExcitations:
    @pytest.mark.parametrize("theta", [np.pi, np.pi/2, np.pi/4, -np.pi/3])
    def test_qubit_excitation_adjacent(self, theta):
        """|10> -> |01>"""
        U0 = tq.gates.X(0)
        QE = tq.gates.QubitExcitation(target=[0, 1], angle="a")
        tq_wfn = tq.simulate(U0 + QE, variables={"a": theta})

        # 1 -> |001>
        initial_state = {1: 1.0}
        spex_result = spex.apply_qubit_excitation(initial_state, [0], [1], theta)

        assert_states_match(tq_wfn, spex_result)

    @pytest.mark.parametrize("theta", [np.pi, np.pi/2])
    def test_qubit_excitation_jump(self, theta):
        """|110> -> |011>"""
        U0 = tq.gates.X([0, 1])
        QE = tq.gates.QubitExcitation(target=[0, 2], angle="a")
        tq_wfn = tq.simulate(U0 + QE, variables={"a": theta})

        # 3 -> |011>
        initial_state = {3: 1.0}
        spex_result = spex.apply_qubit_excitation(initial_state, [0], [2], theta)

        assert_states_match(tq_wfn, spex_result)

    @pytest.mark.parametrize("theta", [np.pi, np.pi/2, np.pi/4, -np.pi/3])
    def test_fermion_excitation_adjacent(self, h2_molecule, theta):
        """|10> -> |01>"""
        U0 = tq.gates.X(0)
        FE = h2_molecule.make_excitation_gate(indices=[(0, 1)], angle="a")
        tq_wfn = tq.simulate(U0 + FE, variables={"a": theta})

        initial_state = {1: 1.0}
        spex_result = spex.apply_fermion_excitation(initial_state, [0], [1], theta)

        assert_states_match(tq_wfn, spex_result)

    @pytest.mark.parametrize("theta", [np.pi, np.pi/2])
    def test_fermion_excitation_jump_occupied(self, h2_molecule, theta):
        """|110> -> -|011>"""
        U0 = tq.gates.X([0, 1])
        FE = h2_molecule.make_excitation_gate(indices=[(0, 2)], angle="a")
        tq_wfn = tq.simulate(U0 + FE, variables={"a": theta})

        initial_state = {3: 1.0}
        spex_result = spex.apply_fermion_excitation(initial_state, [0], [2], theta)

        assert_states_match(tq_wfn, spex_result)

    @pytest.mark.parametrize("theta", [np.pi/2])
    def test_fermion_excitation_jump_empty(self, h2_molecule, theta):
        """|100> -> |001> (no sign flip)"""
        U0 = tq.gates.X(0)
        FE = h2_molecule.make_excitation_gate(indices=[(0, 2)], angle="a")
        tq_wfn = tq.simulate(U0 + FE, variables={"a": theta})

        initial_state = {1: 1.0}
        spex_result = spex.apply_fermion_excitation(initial_state, [0], [2], theta)

        assert_states_match(tq_wfn, spex_result)

    @pytest.mark.parametrize("theta", [np.pi/4, np.pi/2])
    def test_multi_fermion_excitation_standard(self, h2_molecule, theta):
        """(0,1) -> (2,3)"""
        # occupy 0,1 -> |...0011>
        U0 = tq.gates.X([0, 1])
        # reverse pairs for spex sign
        FE = h2_molecule.make_excitation_gate(indices=[(1, 3), (0, 2)], angle="a")
        tq_wfn = tq.simulate(U0 + FE, variables={"a": theta})

        initial_state = {(1 << 0) + (1 << 1): 1.0}  # |...0011> -> 3
        spex_result = spex.apply_fermion_excitation(initial_state, [0, 1], [2, 3], theta)

        assert_states_match(tq_wfn, spex_result)

    @pytest.mark.parametrize("theta", [np.pi/4, np.pi/2])
    def test_multi_fermion_excitation_parity_trap(self, big_molecule, theta):
        """(0,1) -> (2,4) with orbital 3 occupied"""
        # occupy 0,1,3 -> |...01011>
        U0 = tq.gates.X([0, 1, 3])
        # reverse pairs for spex sign
        FE = big_molecule.make_excitation_gate(indices=[(1, 4), (0, 2)], angle="a")
        tq_wfn = tq.simulate(U0 + FE, variables={"a": theta})

        # |...01011> -> 11
        initial_state = {(1 << 0) + (1 << 1) + (1 << 3): 1.0}
        spex_result = spex.apply_fermion_excitation(initial_state, [0, 1], [2, 4], theta)

        assert_states_match(tq_wfn, spex_result)


