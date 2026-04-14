import pytest
import spex_tequila as spex
import numpy as np

def test_pauli_exclusion():
    """Checks that creating two electrons in the same spot results in zero."""
    # Start with |0>
    state, phase = spex.apply_fe_creation(0, 5)
    assert state == (1 << 5)

    # Try creating at index 5 again
    state_fail, phase_fail = spex.apply_fe_creation(state, 5)
    assert phase_fail == 0j
    assert state_fail == 0

def test_annihilation_empty():
    """Checks that annihilating an electron where none exists results in zero."""
    state_fail, phase_fail = spex.apply_fe_annihilation(0, 2)
    assert phase_fail == 0j

def test_anticommutation_creation():
    """
    Checks CAR: {ai_dag, aj_dag} = 0
    ai_dag * aj_dag = -aj_dag * ai_dag
    """
    initial_state = 0  # |0>
    i, j = 1, 3

    # Path 1: Create i then j
    s_i, p_i = spex.apply_fe_creation(initial_state, i)
    s_ij, p_ij = spex.apply_fe_creation(s_i, j)
    total_phase_1 = p_i * p_ij

    # Path 2: Create j then i
    s_j, p_j = spex.apply_fe_creation(initial_state, j)
    s_ji, p_ji = spex.apply_fe_creation(s_j, i)
    total_phase_2 = p_j * p_ji

    assert s_ij == s_ji
    assert total_phase_1 == -total_phase_2

def test_anticommutation_mixed():
    """
    Checks {ai, aj_dag} = delta_ij
    i != j, ai * aj_dag = -aj_dag * ai
    """
    initial_state = 1 << 2  # |00100>
    i, j = 2, 4             # Annihilate at 2, Create at 4

    # Path 1: Annihilate 2, then Create 4
    s1, p1 = spex.apply_fe_annihilation(initial_state, i)
    s12, p12 = spex.apply_fe_creation(s1, j)

    # Path 2: Create 4, then Annihilate 2
    s2, p2 = spex.apply_fe_creation(initial_state, j)
    s21, p21 = spex.apply_fe_annihilation(s2, i)

    assert s12 == s21
    assert p1 * p12 == -(p2 * p21)