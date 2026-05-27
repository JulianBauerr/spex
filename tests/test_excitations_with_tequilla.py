import spex_tequila as spex
import pytest
import numpy as np
import tequila as tq

def assert_states_match(tequila_wfn, spex_state, atol=1e-5):
    tq_state = dict(tequila_wfn.items())

    # Check all states and amplitudes
    for basis_state, tq_amp in tq_state.items():
        spex_amp = spex_state.get(basis_state, 0.0)
        assert np.isclose(spex_amp, np.real(tq_amp), atol=atol), \
            f"Amplitude mismatch at basis state {basis_state}: Spex={spex_amp}, Tequila={np.real(tq_amp)}"

    # Check for extra states
    for basis_state, spex_amp in spex_state.items():
        if abs(spex_amp) > atol:
            tq_amp = tq_state.get(basis_state, 0.0)
            assert np.isclose(spex_amp, np.real(tq_amp), atol=atol), \
                f"Spex produced unexpected amplitude at {basis_state}: {spex_amp}"

@pytest.fixture
def h2_molecule():
    return tq.Molecule("H 0 0 0\nH 0 0 1", "sto-3g")

class TestSpexExcitations:
    @pytest.mark.parametrize("theta", [np.pi, np.pi/2, np.pi/4, -np.pi/3])
    def test_qubit_excitation_adjacent(self, theta):
        """Test qubit excitation: |10> -> |01>"""
        U0 = tq.gates.X(0)
        QE = tq.gates.QubitExcitation(target=[0, 1], angle="a")
        tq_wfn = tq.simulate(U0 + QE, variables={"a": theta})

        # State 1 = 001
        initial_state = {1: 1.0}
        spex_result = spex.apply_qubit_excitation(initial_state, [0], [1], theta)

        assert_states_match(tq_wfn, spex_result)

    @pytest.mark.parametrize("theta", [np.pi, np.pi/2])
    def test_qubit_excitation_jump(self, theta):
        """Test qubit excitation: |110> -> |011>"""
        U0 = tq.gates.X([0, 1])
        QE = tq.gates.QubitExcitation(target=[0, 2], angle="a")
        tq_wfn = tq.simulate(U0 + QE, variables={"a": theta})

        # State 3 = 011
        initial_state = {3: 1.0}
        spex_result = spex.apply_qubit_excitation(initial_state, [0], [2], theta)

        assert_states_match(tq_wfn, spex_result)

    @pytest.mark.parametrize("theta", [np.pi, np.pi/2, np.pi/4, -np.pi/3])
    def test_fermion_excitation_adjacent(self, h2_molecule, theta):
        """Test fermionic: |10> -> |01>"""
        U0 = tq.gates.X(0)
        FE = h2_molecule.make_excitation_gate(indices=[(0, 1)], angle="a")
        tq_wfn = tq.simulate(U0 + FE, variables={"a": theta})

        initial_state = {1: 1.0}
        spex_result = spex.apply_fermion_excitation(initial_state, [0], [1], theta)

        assert_states_match(tq_wfn, spex_result)

    @pytest.mark.parametrize("theta", [np.pi, np.pi/2])
    def test_fermion_excitation_jump_occupied(self, h2_molecule, theta):
        """apply a -1 phase. State: |110> -> -|011>"""
        U0 = tq.gates.X([0, 1])
        FE = h2_molecule.make_excitation_gate(indices=[(0, 2)], angle="a")
        tq_wfn = tq.simulate(U0 + FE, variables={"a": theta})

        initial_state = {3: 1.0}
        spex_result = spex.apply_fermion_excitation(initial_state, [0], [2], theta)

        assert_states_match(tq_wfn, spex_result)

    @pytest.mark.parametrize("theta", [np.pi/2])
    def test_fermion_excitation_jump_empty(self, h2_molecule, theta):
        """The phase should not flip here. State: |100> -> |001>"""
        U0 = tq.gates.X(0)
        FE = h2_molecule.make_excitation_gate(indices=[(0, 2)], angle="a")
        tq_wfn = tq.simulate(U0 + FE, variables={"a": theta})

        initial_state = {1: 1.0}
        spex_result = spex.apply_fermion_excitation(initial_state, [0], [2], theta)

        assert_states_match(tq_wfn, spex_result)