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
State apply_fermionic_excitation(const State& state, const FermionTerm& term, double theta) {
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

/**
 * Applies the Generator to the input state under the constraint that all orbitals are orthogonal to each other.
 * Apply exp(-i*θ * Σ_k (w_k·A_k + w̄_k·A_k†)) to |ψ⟩ - orbitals must be orthogonal
 * @param state input state
 * @param terms Generator
 * @param theta rotation
 * @return output state
 */
State apply_abstract_generator(const State& state, const std::vector<FermionTerm>& terms, double theta) {
    State current_state = state;

    for (const auto& term : terms) {
        current_state = apply_fermionic_excitation(current_state, term, theta);
    }

    return current_state;
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

    // Expose apply_fswap function
    p.def("apply_fswap", &apply_fswap,
        "Apply the fSWAP operator on modes i, j to a quantum state",
        py::arg("state"), py::arg("i"), py::arg("j"));

    // Expose apply_fermion_excitation function (generalized)
    p.def("apply_fermion_excitation", &apply_fermionic_excitation,
        "Apply exp(-i*θ/2 * (w·A + w̄·A†)) to |ψ⟩ where A = Π a†_c Π a_a",
        py::arg("state"), py::arg("term"), py::arg("theta"));

    // Expose apply_abstract_generator function (sum of fermionic terms, exact)
    p.def("apply_abstract_generator", &apply_abstract_generator,
          "Apply exp(-i*θ * Σ_k (w_k·A_k + w̄_k·A_k†)) to |ψ⟩ - orbitals must be orthogonal",
          py::arg("state"), py::arg("terms"), py::arg("theta"));
}
