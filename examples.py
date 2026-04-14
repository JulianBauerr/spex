import spex_tequila as spex
import math

def main():
    # the right start is i=0
    state = 0b1010101110010011
    i = 5
    val = spex.apply_Fe(state, i)
    # expected {1010101110110011, -1.0}
    print(f"{val[0]:016b} {val[1]}")

if __name__ == "__main__":
    main()


def ol_example():
    # Example 1: Parsing a Pauli String
    print("\nExample 1: Parsing a Pauli String")
    pauli_string = "X(0)Y(1)Z(2)"
    pauli_map = spex.parse_pauli_string(pauli_string)
    print(f"Pauli String: {pauli_string}")
    print("Parsed Pauli Map:")
    for qubit, op in pauli_map.items():
        print(f"  Qubit {qubit}: {op}")
    print("-" * 50)

    # Example 2: Applying a Pauli Operator to a Basis State
    print("Example 2: Applying a Pauli Operator to a Basis State")
    pauli_string = "X(0)Y(1)"
    basis_state = 0b01  # |01⟩
    new_state, phase = spex.apply_Pk(pauli_string, basis_state)
    print(f"Applying Pauli String '{pauli_string}' to Basis State |{basis_state:02b}⟩")
    print(f"New Basis State: |{new_state:02b}⟩")
    print(f"Phase Factor: {phase}")
    print("-" * 50)

    # Example 3: Initializing the Zero State
    print("Example 3: Initializing the Zero State")
    num_qubits = 2
    zero_state = spex.initialize_zero_state(num_qubits)
    print(f"Initialized |0>^{num_qubits} State:")
    print(spex.state_to_string(zero_state, num_qubits))
    print("-" * 50)

    # Example 4: Creating Quantum States
    print("Example 4: Creating Quantum States")
    # Define |00⟩ and |11⟩ with amplitudes 1.0 and -1.0 respectively
    psi = {0b00: 1.0, 0b11: -1.0}
    phi = {0b00: 0.5, 0b11: 0.5}
    print("State |ψ⟩:")
    print(spex.state_to_string(psi, num_qubits))
    print("State |φ⟩:")
    print(spex.state_to_string(phi, num_qubits))
    print("-" * 50)

    # Example 5: Computing Inner Product
    print("Example 5: Computing Inner Product")
    inner_prod = spex.inner_product(psi, phi)
    print(f"Inner Product ⟨ψ|φ⟩: {inner_prod}")
    print("-" * 50)

    # Example 6: Defining Hamiltonian H
    print("Example 6: Defining Hamiltonian H")
    # H = X(0) + (-0.5)Z(1)
    H_terms = [("X(0)", 1.0), ("Z(1)", -0.5)]
    print("Hamiltonian H Terms:")
    for term, coeff in H_terms:
        print(f"  {coeff} * {term}")
    print("-" * 50)

    # Example 7: Computing Expectation Value ⟨φ|H|ψ⟩
    print("Example 7: Computing Expectation Value ⟨φ|H|ψ⟩")
    expectation = spex.expectation_value(phi, psi, H_terms)
    print(f"Expectation Value ⟨φ|H|ψ⟩: {expectation}")
    print("-" * 50)

    # Example 8: Computing Expectation Value in Parallel
    print("Example 8: Computing Expectation Value in Parallel")
    expectation_parallel = spex.expectation_value_parallel(phi, psi, H_terms)
    print(f"Expectation Value ⟨φ|H|ψ⟩ (Parallel): {expectation_parallel}")
    print("-" * 50)

    # Example 9: Applying an Exponential Pauli Operator
    print("Example 9: Applying an Exponential Pauli Operator")
    pauli_string = "X(0)"
    theta = math.pi / 2  # 90 degrees
    new_state = spex.apply_exp_pauli(pauli_string, theta, psi)
    print(f"Applying e^(-i*{theta}) * {pauli_string} to |ψ⟩")
    print("New State |ψ'>:")
    print(spex.state_to_string(new_state, num_qubits))
    print("-" * 50)

    # Example 10: Applying a Sequence of Exponential Pauli Operators
    print("Example 10: Applying a Sequence of Exponential Pauli Operators")
    U_terms = [
        spex.ExpPauliTerm("X(0)", math.pi / 2),
        spex.ExpPauliTerm("Y(1)", math.pi / 4)
    ]
    final_state = spex.apply_U(U_terms, spex.initialize_zero_state(num_qubits))
    print("Applying Sequence of Operators:")
    for term in U_terms:
        print(f"  e^(-i*{term.angle}) * {term.pauli_string}")
    print("Final State after Applying U:")
    print(spex.state_to_string(final_state, num_qubits))
    print("-" * 50)

    # Example 11: Displaying a Quantum State
    print("Example 11: Displaying a Quantum State")
    print("State |ψ⟩:")
    print(spex.state_to_string(psi, num_qubits))
    print("State |φ⟩:")
    print(spex.state_to_string(phi, num_qubits))
    print("Final State |ψ'> after Applying U:")
    print(spex.state_to_string(final_state, num_qubits))
    print("-" * 50)