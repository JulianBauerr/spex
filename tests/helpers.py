import math

S2 = 1.0 / math.sqrt(2.0)
PI = math.pi
PI_2 = math.pi / 2.0
PI_4 = math.pi / 4.0


def fock(*orbitals):
    """Integer basis state for the Fock state with the given orbitals occupied."""
    return sum(1 << orb for orb in orbitals)


def assert_complex_almost_equal(actual, expected, places=11):
    diff = abs(complex(actual) - complex(expected))
    if diff > 10 ** (-places):
        raise AssertionError(f"Expected {expected}, got {actual} (diff {diff:.2e})")


def assert_state_dicts_almost_equal(actual, expected, places=11):
    atol = 10 ** (-places)
    for key in set(actual) | set(expected):
        va = complex(actual.get(key, 0.0j))
        ve = complex(expected.get(key, 0.0j))
        diff = abs(va - ve)
        if diff > atol:
            raise AssertionError(f"Mismatch at |{key:b}>: got {va}, expected {ve} (diff {diff:.2e})")


def assert_states_match(tequila_wfn, spex_state, atol=1e-7):
    tq_state = {int(k): complex(v) for k, v in tequila_wfn.items()}
    for key in set(tq_state) | set(spex_state):
        tq_amp = tq_state.get(key, 0.0j)
        spx_amp = spex_state.get(key, 0.0j)
        diff = abs(tq_amp - spx_amp)
        if diff > atol:
            raise AssertionError(f"Mismatch at |{key:b}>: Spex={spx_amp}, Tequila={tq_amp}")
