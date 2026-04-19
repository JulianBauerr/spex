import pytest
import spex_tequila as spex
import numpy as np

K_IDX = [0, 1]
L_IDX = [2, 3]

def test_forward_swap():
    # State: 0011 -> 1100 (c = 0)
    initial_state = {0b0011: 1.0}
    theta = np.pi # cos(pi/2) = 0, sin(pi/2) = 1

    result = spex.apply_qubit_excitation(initial_state, K_IDX, L_IDX, theta)

    assert np.isclose(result[0b0011], 0.0)

    assert 0b1100 in result
    assert np.isclose(result[0b1100], 1.0)

def test_reverse_swap():
    """Test a valid excitation where 1s move from l to k (Sign should be -1)."""
    # State: 1100 -> 0011 (c = -1)
    initial_state = {0b1100: 1.0}
    theta = np.pi

    result = spex.apply_qubit_excitation(initial_state, K_IDX, L_IDX, theta)

    assert np.isclose(result[0b1100], 0.0)

    # New state should have amplitude -1.0 (1.0 * sin(pi/2) * -1)
    assert 0b0011 in result
    assert np.isclose(result[0b0011], -1.0)

def test_nullspace_empty_and_full():
    """Test states where all bits are 0 or all bits are 1."""
    initial_state = {
        0b0000: 0.5,
        0b1111: 0.5
    }
    theta = np.pi

    result = spex.apply_qubit_excitation(initial_state, K_IDX, L_IDX, theta)

    assert np.isclose(result[0b0000], 0.5)
    assert np.isclose(result[0b1111], 0.5)

    # No new states should have been created
    assert len(result) == 2

def test_nullspace_edge_case_1001():
    # State: 1001 -> should be skipped
    initial_state = {0b1001: 1.0}
    theta = np.pi

    result = spex.apply_qubit_excitation(initial_state, K_IDX, L_IDX, theta)

    assert np.isclose(result[0b1001], 1.0)
    assert 0b0110 not in result

def test_theta_zero():
    """Test that applying a 0 degree angle does not change the state."""
    initial_state = {0b0011: 1.0}
    theta = 0.0

    result = spex.apply_qubit_excitation(initial_state, K_IDX, L_IDX, theta)

    assert np.isclose(result[0b0011], 1.0)

    assert 0b1100 in result
    assert np.isclose(result[0b1100], 0.0)