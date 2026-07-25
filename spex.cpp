#include <unordered_map>
#include <complex>
#include <string>
#include <cmath>
#include <cstdint>
#include <vector>
#include <regex>
#include <thread>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "include/unordered_dense.h"

// Namespace for Pybind11
namespace py = pybind11;

// Type definition for the quantum state
using State = ankerl::unordered_dense::map<uint64_t, std::complex<double>>;

constexpr bool TINY_RESIDUAL_CLEANUP = true;
constexpr double TINY_RESIDUAL_CLEANUP_TOL = 1e-12;

/**
 * Structure to define one term of a general fermionic operator.
 *
 * Spex applies annihilation in vector order, then creation in vector order.
 * The represented operator is: a†_{c[-1]}...a†_{c[0]} · a_{a[-1]}...a_{a[0]}.
 */
struct FermionTerm {
    std::vector<int> creation_idx;
    std::vector<int> annihilation_idx;
    std::complex<double> weight = 1.0;

    /**
     * Return a Fermion string representation.
     *
     * This is the same representation as Googles OpenFermion uses
     */
    std::string to_fermion_string() const {
        std::ostringstream os;
        bool first = true;
        auto add = [&](const std::string &s) {
            if (!first) os << ' ';
            os << s;
            first = false;
        };
        for (auto it = creation_idx.rbegin(); it != creation_idx.rend(); ++it)
            add(std::to_string(*it) + "^");
        for (auto it = annihilation_idx.rbegin(); it != annihilation_idx.rend(); ++it)
            add(std::to_string(*it));
        return os.str();
    }
};

/**
 * Parse a fermion operator string into a FermionTerm.
 *
 * Accepts strings like "0^ 1" or "2^ 3^ 0 1".
 * Operators without "^" are annihilation; those with "^" are creation.
 * The result matches spex's convention (spex applies in vector order,
 * so index vectors preserve the string's left-to-right order).
 */
FermionTerm parse_fermion_string(const std::string &s, std::complex<double> weight = 1.0) {
    std::vector<int> creation, annihilation;
    std::istringstream iss(s);
    std::string token;
    while (iss >> token) {
        if (token.size() >= 2 && token.back() == '^') {
            token.pop_back();
            creation.push_back(std::stoi(token));
        } else {
            annihilation.push_back(std::stoi(token));
        }
    }
    return {creation, annihilation, weight};
}


/**
 * Structure to represent an exponential Pauli term like "X(0)Y(1)Z(2)" and an angle θ.
 */
struct ExpPauliTerm {
    ankerl::unordered_dense::map<int, char> pauli_map; 
    double angle;

    ExpPauliTerm() : pauli_map(), angle(0.0) {}

    void set_angle(double new_angle) {
        angle = new_angle;
    }
};

/**
 * Applies a Pauli operator to a basis state, returning the new basis state and the associated phase factor.
 *
 * @param pauli_map A map from qubit indices to Pauli operators ('I', 'X', 'Y', 'Z').
 * @param basis_state The basis state represented as an unsigned 64-bit integer.
 * @return A pair containing the new basis state and the phase factor (std::complex<double>).
 */
std::pair<uint64_t, std::complex<double>> apply_Pk(
    const ankerl::unordered_dense::map<int, char>& pauli_map, uint64_t basis_state) {
    uint64_t new_basis_state = basis_state;
    std::complex<double> phase = 1.0;

    for (const auto& [qubit, p] : pauli_map) {
        if (qubit < 0 || qubit >= 64) {
            throw std::out_of_range("Qubit index out of valid range (0-63): " + std::to_string(qubit));
        }

        bool bit = (basis_state >> qubit) & 1ULL; // get the state of the qubit

        switch (p) {
            case 'I':
                // Identity operator: do nothing
                break;
            case 'X':
                // Flip the qubit
                new_basis_state ^= (1ULL << qubit);
                break;
            case 'Y':
                // Flip the qubit and add phase factor
                new_basis_state ^= (1ULL << qubit);
                phase *= bit ? std::complex<double>(0, -1) : std::complex<double>(0, 1);
                break;
            case 'Z':
                // Add phase factor if the qubit is in state |1⟩
                if (bit) {
                    phase *= -1.0;
                }
                break;
            default:
                throw std::invalid_argument(std::string("Invalid Pauli operator: ") + p);
        }
    }
    return {new_basis_state, phase};
}

/**
 * Computes the inner product ⟨ψ|φ⟩ of two quantum states.
 *
 * @param psi The first quantum state (bra), represented as a State.
 * @param phi The second quantum state (ket), represented as a State.
 * @return The inner product ⟨ψ|φ⟩ as a std::complex<double>.
 */
std::complex<double> inner_product(const State& psi, const State& phi) {
    std::complex<double> result = 0.0;

    if (psi.empty() || phi.empty()) {
        throw std::invalid_argument("Quantum states cannot be empty.");
    }

    // Determine the smaller state for optimization
    const State* smaller_state = (psi.size() < phi.size()) ? &psi : &phi;
    const State* larger_state = (smaller_state == &psi) ? &phi : &psi;

    // Iterate over the basis states of the smaller state
    for (const auto& [basis_state, coeff_smaller] : *smaller_state) {
        auto it = larger_state->find(basis_state);
        if (it != larger_state->end()) {
            result += std::conj(coeff_smaller) * it->second; // we always need to conjugate the psi state. In some cases with the swapping we conjugate the wrong state.
        }
    }

    return result;
}

/**
 * Computes the expectation value ⟨φ|H|ψ⟩ for a given Hamiltonian H represented as a sum of Pauli terms.
 *
 * @param phi The bra state ⟨φ|, represented as a State.
 * @param psi The ket state |ψ⟩, represented as a State.
 * @param H_terms A vector of pairs, each containing a Pauli string and its coefficient in H.
 * @return The expectation value ⟨φ|H|ψ⟩ as a std::complex<double>.
 */
std::complex<double> expectation_value(
    const State& phi, const State& psi,
    const std::vector<std::pair<ExpPauliTerm, std::complex<double>>>& H_terms) {
    if (psi.empty() || phi.empty()) {
        throw std::invalid_argument("Quantum states cannot be empty.");
    }

    if (H_terms.empty()) {
        throw std::invalid_argument("Hamiltonian terms cannot be empty.");
    }

    std::complex<double> result = 0.0;
    State psi_k;

    for (const auto& [term, coeff] : H_terms) {
        if (coeff == 0.0) {
            continue; // Skip zero coefficients
        }
        
        psi_k.clear();

        // Apply Pk to ψ to obtain ψk
        for (const auto& [basis_state, amplitude] : psi) {
            auto [new_basis_state, phase] = apply_Pk(term.pauli_map, basis_state);
            psi_k[new_basis_state] += phase * amplitude;
        }
        // Compute ⟨φ|ψk⟩
        std::complex<double> inner_prod = inner_product(phi, psi_k);
        // Accumulate c_k * ⟨φ|ψk⟩
        result += coeff * inner_prod;
    }
    return result;
}

/**
 * Computes the expectation value ⟨φ|H|ψ⟩ in parallel.
 *
 * @param phi The bra state ⟨φ|, represented as a State.
 * @param psi The ket state |ψ⟩, represented as a State.
 * @param H_terms A vector of pairs, each containing a Pauli string and its coefficient in H.
 * @return The expectation value ⟨φ|H|ψ⟩ as a std::complex<double>.
 *
 * Note: This function parallelizes over the terms in H for performance on large Hamiltonians.
 */
std::complex<double> expectation_value_parallel(
    const State& phi, const State& psi,
    const std::vector<std::pair<ExpPauliTerm, std::complex<double>>>& H_terms,
    int user_threads = -1) {
    if (user_threads == 1 || user_threads == 0) {
        return expectation_value(phi, psi, H_terms);
    }

    if (psi.empty() || phi.empty()) {
        throw std::invalid_argument("Quantum states cannot be empty.");
    }

    if (H_terms.empty()) {
        throw std::invalid_argument("Hamiltonian terms cannot be empty.");
    }

    std::complex<double> result = 0.0;

    // Determine the number of threads to use
    unsigned int num_threads = (user_threads < 0)
                               ? (std::thread::hardware_concurrency() - 2)
                               : static_cast<unsigned int>(user_threads);
    if (num_threads == 0) {
        num_threads = 4;
    }

    size_t total_terms = H_terms.size();
    size_t chunk_size = (total_terms + num_threads - 1) / num_threads;

    // Vector to hold partial results
    std::vector<std::complex<double>> partial_results(num_threads, 0.0);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Worker function for each thread
    auto worker = [&](size_t start_idx, size_t end_idx, unsigned int thread_id) {
        std::complex<double> local_result = 0.0;
        State psi_k;

        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            const auto& [term, coeff] = H_terms[idx];
            if (coeff == 0.0) continue;
            psi_k.clear();

            // Apply Pk to ψ to obtain ψk
            for (const auto& [basis_state, amplitude] : psi) {
                auto [new_basis_state, phase] = apply_Pk(term.pauli_map, basis_state);
                psi_k[new_basis_state] += phase * amplitude;
            }

            // Accumulate c_k * ⟨φ|ψk⟩
            local_result += coeff * inner_product(phi, psi_k);
        }
        partial_results[thread_id] = local_result;
    };

    // Launch threads
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, total_terms);
        if (start_idx >= end_idx) break;
        threads.emplace_back(worker, start_idx, end_idx, t);
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Combine partial results
    for (const auto& partial : partial_results) {
        result += partial;
    }

    return result;
}

/**
 * Applies the exponential of a Pauli operator e^{-iθP} to a quantum state |ψ⟩.
 *
 * @param pauli_string The Pauli operator P in string format, e.g., "X(0)Y(1)".
 * @param theta The angle θ in radians.
 * @param state The input quantum state |ψ⟩, represented as a State.
 * @return The new quantum state after applying e^{-iθP}|ψ⟩.
 */
State apply_exp_pauli(const ExpPauliTerm& term, const State& state, double threshold) {
    if (state.empty()) {
        throw std::invalid_argument("Input quantum state cannot be empty.");
    }

    double theta = term.angle;
    if (theta == 0.0) return state; // No change if theta is zero

    State new_state;
    new_state.reserve(state.size() * 2);
    
    const double cos_theta = std::cos(theta / 2.0);
    const double sin_theta = std::sin(theta / 2.0);
    bool use_threshold = (threshold >= 0.0);

    // Compute P|ψ⟩ and update new_state
    for (const auto& [basis_state, amplitude] : state) {
        auto [new_basis_state, phase] = apply_Pk(term.pauli_map, basis_state);

        const auto amp1 = cos_theta * amplitude;
        const auto amp2 = (-std::complex<double>(0, sin_theta)) * phase * amplitude;

        if (use_threshold) {
            if (std::abs(amp1) >= threshold) {
                new_state[basis_state] += amp1;
            }
            if (std::abs(amp2) >= threshold) {
                new_state[new_basis_state] += amp2;
            }
        } else {
            new_state[basis_state] += amp1;
            new_state[new_basis_state] += amp2;
        }
    }

    if (use_threshold) {
        for (auto it = new_state.begin(); it != new_state.end(); ) {
            if (std::abs(it->second) < threshold) {
                it = new_state.erase(it);
            } else {
                ++it;
            }
        }
    }
    return new_state;
}

/**
 * Applies a sequence of exponential Pauli operators to an initial quantum state.
 *
 * @param U_terms A vector of ExpPauliTerm, each containing a Pauli string and an angle θ.
 * @param initial_state The initial quantum state, represented as a State.
 * @return The final quantum state after applying all exponential operators.
 */
State apply_U(const std::vector<ExpPauliTerm>& U_terms, const State& initial_state, double threshold) {
    if (initial_state.empty()) {
        throw std::invalid_argument("Initial quantum state cannot be empty.");
    }

    State state = initial_state;
    for (const auto& term : U_terms) {
        state = apply_exp_pauli(term, state, threshold);
    }
    return state;
}

/**
 * Initializes the zero state |0⟩^n for a given number of qubits.
 *
 * @param num_qubits The number of qubits n.
 * @return The zero quantum state |0⟩^n, represented as a State.
 */
State initialize_zero_state(int num_qubits) {
    if (num_qubits <= 0 || num_qubits > 64) {
        throw std::invalid_argument("Number of qubits must be between 1 and 64.");
    }

    // Basis state |0...0⟩ corresponds to the basis state with index 0
    State state;
    state.emplace(0ULL, 1.0);
    return state;
}

/**
 * Converts a quantum state to a string representation for display or debugging.
 *
 * @param state The quantum state to convert, represented as a State.
 * @param num_qubits The number of qubits in the state.
 * @return A string representation of the quantum state.
 */
std::string state_to_string(const State& state, int num_qubits) {
    if (num_qubits <= 0 || num_qubits > 64) {
        throw std::invalid_argument("Number of qubits must be between 1 and 64.");
    }
    
    std::string s;
    for (const auto& [basis_state, coeff] : state) {
        s += "|";
        for (int i = num_qubits - 1; i >= 0; --i) {
            s += ((basis_state >> i) & 1ULL) ? "1" : "0";
        }
        s += "⟩: " + std::to_string(coeff.real()) +
             (coeff.imag() >= 0 ? "+" : "") + std::to_string(coeff.imag()) + "i\n";
    }
    return s;
}

State apply_qubit_excitation(const State& state, const std::vector<int>& k,
    const std::vector<int>& l, const double theta) {
    // Checks
    if (state.empty())
        throw std::invalid_argument("A state cannot be empty");
    if (k.size() != l.size())
        throw std::invalid_argument("The orbital sets must have the same length");

    // Initials
    State new_state;
    new_state.reserve(state.size() * 2);
    // Loop through state
    for (const auto& [basis_state, coeff] : state) {
        // Check for P0
        bool k_val = (basis_state >> k[0]) & 1ULL;
        bool skip = false;
        for (int i = 0; i < k.size(); i++) {
            if (((basis_state >> k[i]) & 1ULL) != k_val) skip = true;
            if (((basis_state >> l[i]) & 1ULL) == k_val) skip = true;
        }
        if (skip) {
            new_state[basis_state] += coeff;
        } else {
            // Implement excitation
            std::complex<double> c_ibs = coeff * std::cos(theta / 2);
            uint64_t partner_basis_state = basis_state;
            for (int i = 0; i < k.size(); i++) {
                partner_basis_state ^= (1ULL << k[i]);
                partner_basis_state ^= (1ULL << l[i]);
            }
            int c = (basis_state >> k[0]) & 1ULL ? 1 : -1;
            std::complex<double> c_ipbs = c * std::sin(theta/2) * coeff;

            new_state[basis_state] += c_ibs;
            new_state[partner_basis_state] += c_ipbs;
        }
    }
    // Prune tiny residual amplitudes introduced by floating point arithmetic
    for (auto it = new_state.begin(); it != new_state.end(); ) {
        if (std::abs(it->second) < TINY_RESIDUAL_CLEANUP_TOL) it = new_state.erase(it);
        else ++it;
    }
    return new_state;
}


/**
 * Apply the fermionic SWAP (fSWAP) operator on oribtal i and j.
 */
State apply_fswap(const State& state, int i, int j) {
    if (i == j) return state;
    if (state.empty())
        throw std::invalid_argument("A state cannot be empty");

    const uint64_t mask_i = 1ULL << i;
    const uint64_t mask_j = 1ULL << j;

    State new_state;
    new_state.reserve(state.size());

    for (const auto& [basis_state, coeff] : state) {
        bool bit_i = (basis_state >> i) & 1ULL;
        bool bit_j = (basis_state >> j) & 1ULL;

        if (bit_i == bit_j) {
            // |00⟩ → |00⟩, |11⟩ → -|11⟩
            double phase = bit_i ? -1.0 : 1.0;
            new_state[basis_state] += phase * coeff;
        } else {
            // |01⟩ ↔ |10⟩, swap the two bits
            uint64_t swapped = basis_state ^ mask_i ^ mask_j;
            new_state[swapped] += coeff;
        }
    }

    // Prune tiny residuals
    if (TINY_RESIDUAL_CLEANUP) {
        for (auto it = new_state.begin(); it != new_state.end(); ) {
            if (std::abs(it->second) < TINY_RESIDUAL_CLEANUP_TOL)
                it = new_state.erase(it);
            else ++it;
        }
    }
    return new_state;
}

/**
 * Applies the unitary U = exp(-i(θ/2) * G) generated by the Hermitian operator
 * G = w·A + w̄·A†  where A = Π a†_{cᵢ} Π a_{aⱼ} and w is any complex weight.
 *
 * When |w|=0 the state is returned unchanged.
 */
State apply_fermion_excitation(const State& state, const FermionTerm& term, double theta) {
    if (state.empty())
        throw std::invalid_argument("A state cannot be empty");

    const std::complex<double> w = term.weight;
    const double abs_w = std::abs(w);
    if (abs_w == 0.0) return state;

    const double cos2 = std::cos(abs_w * theta / 2.0);
    const double sin2 = std::sin(abs_w * theta / 2.0);

    const std::complex<double> i_unit(0.0, 1.0);
    const std::complex<double> factor_a  = -i_unit * w / abs_w;
    const std::complex<double> factor_ad = -i_unit * std::conj(w) / abs_w;

    State new_state;
    new_state.reserve(state.size() * 2);

    for (const auto& [basis_state, coeff] : state) {
        // ---- Nullspace check ----
        bool a_acts  = true;  // A|i⟩ ≠ 0  (all creation empty, all annihilation occupied)
        bool ad_acts = true;  // A†|i⟩ ≠ 0 (all creation occupied, all annihilation empty)

        for (int idx : term.creation_idx) {
            bool bit = (basis_state >> idx) & 1ULL;
            if (bit) a_acts = false;   // creation idx occupied → A can't create
            else ad_acts = false;      // creation idx empty → A† can't annihilate
        }
        for (int idx : term.annihilation_idx) {
            bool bit = (basis_state >> idx) & 1ULL;
            if (bit) ad_acts = false;  // annihilation idx occupied → A† can't create
            else a_acts = false;       // annihilation idx empty → A can't annihilate
        }

        if (!a_acts && !ad_acts) {
            // Nullspace: just copy the basis state unchanged
            new_state[basis_state] += coeff;
            continue;
        }

        // Active space: apply rotation
        int phase = 1;
        uint64_t new_basis_state = basis_state;

        const std::vector<int>& annihilate_idx = a_acts ? term.annihilation_idx : term.creation_idx;
        const std::vector<int>& create_idx     = a_acts ? term.creation_idx : term.annihilation_idx;

        // Apply annihilation
        for (int idx : annihilate_idx) {
            uint64_t mask = (1ULL << idx) - 1;
            if (__builtin_popcountll(new_basis_state & mask) % 2 != 0) {
                phase = -phase;
            }
            new_basis_state ^= (1ULL << idx);
        }

        // Apply creation
        for (int idx : create_idx) {
            uint64_t mask = (1ULL << idx) - 1;
            if (__builtin_popcountll(new_basis_state & mask) % 2 != 0) {
                phase = -phase;
            }
            new_basis_state ^= (1ULL << idx);
        }

        // Update amplitudes
        std::complex<double> factor = a_acts ? factor_a : factor_ad;

        new_state[basis_state]       += coeff * cos2;
        new_state[new_basis_state]   += factor * std::complex<double>(phase, 0.0) * sin2 * coeff;
    }

    // Prune tiny residual amplitudes
    if (TINY_RESIDUAL_CLEANUP) {
        for (auto it = new_state.begin(); it != new_state.end(); ) {
            if (std::abs(it->second) < TINY_RESIDUAL_CLEANUP_TOL) it = new_state.erase(it);
            else ++it;
        }
    }
    return new_state;
}

std::complex<double> expectation_value_fermionic_term(const State &phi, const State &psi,
                                                      const FermionTerm &term) {
    // Init
    std::complex<double> term_overlap = 0.0;
    //Loop
    for (const auto& [basis_state_psi, coeff_psi] : psi) {
        uint64_t new_basis_state = basis_state_psi;
        int phase = 1;
        bool skip = false;

        // apply Annihilation
        for (int idx : term.annihilation_idx) {
            // Check of the orbital is already empty
            if ((new_basis_state >> idx & 1ULL) == 0) {
                skip = true;
                break;
            }

            uint64_t mask = (1ULL << idx) - 1;
            if (__builtin_popcountll(new_basis_state & mask) % 2 != 0) {
                phase = -phase;
            }
            new_basis_state ^= (1ULL << idx);
        }
        if (skip) continue;

        // apply Creation
        for (int idx : term.creation_idx) {
            // Check of the orbital is already occupied
            if ((new_basis_state >> idx & 1ULL) == 1) {
                skip = true;
                break;
            }
            uint64_t mask = (1ULL << idx) - 1;
            if (__builtin_popcountll(new_basis_state & mask) % 2 != 0) {
                phase = -phase;
            }
            new_basis_state ^= (1ULL << idx);
        }
        if (skip) continue;

        // Compute overlap
        auto it = phi.find(new_basis_state);
        if (it != phi.end()) {
            std::complex<double> coeff_phi = std::conj(it->second);
            term_overlap += coeff_phi * (term.weight * std::complex<double>(phase, 0.0) * coeff_psi);
        }
    }

    return term_overlap;
}

std::complex<double> expectation_value_fermionic(const State &phi, const State &psi,
                                                 const std::vector<FermionTerm> &operator_sum) {
    std::complex<double> result = 0.0;

    for (const auto &term: operator_sum) {
        result += expectation_value_fermionic_term(phi, psi, term);
    }
    return result;
}

namespace pybind11 {
namespace detail {
template <class Key, class T, class Hash, class KeyEqual, class Allocator>
struct type_caster<ankerl::unordered_dense::map<Key, T, Hash, KeyEqual, Allocator>>
    : map_caster<ankerl::unordered_dense::map<Key, T, Hash, KeyEqual, Allocator>, Key, T>
{};

} 
} 

// Bindings for Pybind11
PYBIND11_MODULE(spex_tequila, p) {
    p.doc() = "Expectation value computation module on sparse pauli states for tequila, implemented in C++ using Pybind11";

    // Expose ExpPauliTerm structure
    py::class_<ExpPauliTerm>(p, "ExpPauliTerm")
        .def(py::init<>())
        .def_readwrite("pauli_map", &ExpPauliTerm::pauli_map)
        .def_readwrite("angle", &ExpPauliTerm::angle)
        .def("set_angle", &ExpPauliTerm::set_angle,
             "Set the angle of this ExpPauliTerm");

    // Expose apply_Pk function
    p.def("apply_Pk", &apply_Pk,
          "Apply a Pauli operator to a basis state",
          py::arg("pauli_map"), py::arg("basis_state"));

    // Expose inner_product function
    p.def("inner_product", &inner_product,
          "Compute the inner product of two quantum states",
          py::arg("psi"), py::arg("phi"));


    // Expose expectation_value function
    p.def("expectation_value", &expectation_value,
          "Compute the expectation value ⟨φ|H|ψ⟩",
          py::arg("phi"), py::arg("psi"), py::arg("H_terms"));

    // Expose expectation_value_parallel function
    p.def("expectation_value_parallel", &expectation_value_parallel,
          "Compute the expectation value ⟨φ|H|ψ⟩ (parallelized)",
          py::arg("phi"), py::arg("psi"), py::arg("H_terms"), py::arg("num_threads") = -1,
          py::call_guard<py::gil_scoped_release>()
    );

    // Expose apply_exp_pauli function
    p.def("apply_exp_pauli", &apply_exp_pauli,
          "Apply e^{-iθP} to a quantum state",
          py::arg("term"), py::arg("state"), py::arg("threshold"));

    // Expose apply_U function
    p.def("apply_U", &apply_U,
          "Apply a sequence of exponential Pauli operators to a quantum state",
          py::arg("U_terms"), py::arg("initial_state"), py::arg("threshold"));

    // Expose initialize_zero_state function
    p.def("initialize_zero_state", &initialize_zero_state,
          "Initialize the |0⟩ state",
          py::arg("num_qubits"));

    // Expose state_to_string function
    p.def("state_to_string", &state_to_string,
          "Convert a quantum state to a string representation",
          py::arg("state"), py::arg("num_qubits"));

    // Expose FermionTerm structure
    py::class_<FermionTerm>(p, "FermionTerm")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&, const std::vector<int>&, std::complex<double>>(),
             py::arg("creation_idx"), py::arg("annihilation_idx"), py::arg("weight"))
        .def_readwrite("creation_idx", &FermionTerm::creation_idx)
        .def_readwrite("annihilation_idx", &FermionTerm::annihilation_idx)
        .def_readwrite("weight", &FermionTerm::weight)
        .def("to_fermion_string", &FermionTerm::to_fermion_string,
             "Return the Fermion string representation");

    // Standalone parser: Fermion string → FermionTerm
    p.def("parse_fermion_string", &parse_fermion_string,
          "Parse a Fermion operator string into a FermionTerm",
          py::arg("s"), py::arg("weight") = 1.0);

    // Expose expectation_value_fermionic_term function
    p.def("expectation_value_fermionic_term", &expectation_value_fermionic_term,
          "Compute ⟨φ|term|ψ⟩ for a single FermionTerm",
          py::arg("phi"), py::arg("psi"), py::arg("term"));

    // Expose expectation_value_fermionic function
    p.def("expectation_value_fermionic", &expectation_value_fermionic,
          "Compute ⟨φ|O|ψ⟩ for a sum of FermionTerms",
          py::arg("phi"), py::arg("psi"), py::arg("operator_sum"));

    // Expose apply_qubit_excitation function
    p.def("apply_qubit_excitation", &apply_qubit_excitation,
        "",
        py::arg("state"), py::arg("k_vector"), py::arg("l_vector"), py::arg("theta"));

    // Expose apply_fswap function
    p.def("apply_fswap", &apply_fswap,
        "Apply the fSWAP operator on modes i, j to a quantum state",
        py::arg("state"), py::arg("i"), py::arg("j"));

    // Expose apply_fermion_excitation function (generalized)
    p.def("apply_fermion_excitation", &apply_fermion_excitation,
        "Apply exp(-i*θ/2 * (w·A + w̄·A†)) to |ψ⟩ where A = Π a†_c Π a_a",
        py::arg("state"), py::arg("term"), py::arg("theta"));
}
