import spex_tequila as spex

from tequila import TequilaException, QubitWaveFunction, QubitHamiltonian
from tequila.quantumchemistry.chemistry_tools import NBodyTensor
from tequila.quantumchemistry import qc_base
from tequila.utils.bitstrings import BitNumbering, reverse_int_bits
from numpy import eye, ndarray, array, complex128, real
from openfermion import FermionOperator
from numbers import Number
from typing import Union, List, Callable


_ZERO_TOL = 1e-12


class SpexExpval:
    def __init__(
        self,
        operator: Union[str, FermionOperator, List[FermionOperator]] = None,
        **kwargs,
    ):
        self.operator = None
        self.hamiltonian: List[spex.FermionTerm] = []
        self._init_state_bra: dict = None
        self._init_state_ket: dict = None
        self._name = None

        # Molecule Data
        self.norb: int = None
        self.n_alpha: int = None
        self.n_beta: int = None
        self.core_energy: float = 0.0
        self.one_body_integrals: ndarray = None
        self.two_body_integrals: ndarray = None
        self.mo_coeff: ndarray = None
        self.spin: float = 0
        self.symmetry = None
        self.atom = None
        self.basis = None
        self.active_space = None

        if 'H' in kwargs:
            if operator is not None:
                raise TequilaException('Two operators provided?')
            operator = kwargs.pop('H')
        if 'mol' in kwargs:
            if 'molecule' in kwargs and kwargs['molecule']:
                raise TequilaException('Two molecules provided?')
            kwargs['molecule'] = kwargs.pop('mol')

        if 'molecule' in kwargs and kwargs['molecule']:
            molecule = kwargs.pop('molecule')
            if isinstance(molecule, qc_base.QuantumChemistryBase):
                self.mo_coeff = molecule.integral_manager.orbital_coefficients
                c, h, g = molecule.get_integrals()
                g = g.reorder('chem').elems
                self.core_energy = c
                self.one_body_integrals = h
                self.two_body_integrals = g
                self.norb = molecule.n_orbitals
                self.spin = (molecule.parameters.multiplicity - 1) / 2
                self.n_alpha = int((molecule.n_electrons + 2 * self.spin) // 2)
                self.n_beta = int((molecule.n_electrons - 2 * self.spin) // 2)
                self.symmetry = getattr(molecule, 'point_group', None)
                self.atom = molecule.parameters.get_geometry()
                self.basis = molecule.parameters.basis_set
                self.active_space = [i.idx_total for i in molecule.integral_manager.active_orbitals]
            else:
                raise TequilaException(f"No molecule type {type(molecule).__name__} supported")
        elif 'integral_manager' in kwargs and 'parameters' in kwargs:
            integral = kwargs.pop('integral_manager')
            params = kwargs.pop('parameters')
            self.mo_coeff = integral.orbital_coefficients
            c, h, g = integral.get_integrals()
            g = g.reorder('chem').elems
            self.core_energy = c
            self.one_body_integrals = h
            self.two_body_integrals = g
            self.norb = len(integral.active_orbitals)
            self.spin = (params.multiplicity - 1) / 2
            n_elec = getattr(params, 'total_n_electrons', None)
            if n_elec is None:
                n_elec = getattr(params, 'n_electrons', None)
            if n_elec is None:
                raise TequilaException('No manner of defining the amount of electrons provided')
            self.n_alpha = int((n_elec + 2 * self.spin) // 2)
            self.n_beta = int((n_elec - 2 * self.spin) // 2)
            self.atom = params.get_geometry()
            self.basis = params.basis_set
            self.active_space = [i.idx_total for i in integral.active_orbitals]
            if 'point_group' in kwargs:
                self.symmetry = kwargs.pop('point_group')
            elif 'symmetry' in kwargs:
                self.symmetry = kwargs.pop('symmetry')
        else:
            int1e = None
            int2e = None
            e_core = None
            mo_coeff = None
            n_elec = None
            n_alpha = None
            n_beta = None
            spin = 0
            point_group = None
            if "int1e" in kwargs:
                int1e = kwargs.pop('int1e')
            elif "one_body_integrals" in kwargs:
                int1e = kwargs.pop('one_body_integrals')
            elif "h" in kwargs:
                int1e = kwargs.pop('h')
            if 'int2e' in kwargs:
                int2e = kwargs.pop('int2e')
            elif 'two_body_integrals' in kwargs:
                int2e = kwargs.pop('two_body_integrals')
            elif 'g' in kwargs:
                int2e = kwargs.pop('g')
            if isinstance(int2e, NBodyTensor):
                int2e = int2e.reorder('chem').elems
            if 'e_core' in kwargs:
                e_core = kwargs.pop('e_core')
            elif 'constant_term' in kwargs:
                e_core = kwargs.pop('constant_term')
            elif 'constant' in kwargs:
                e_core = kwargs.pop('constant')
            elif 'c' in kwargs:
                e_core = kwargs.pop('c')
            else:
                e_core = 0.0
            if 'mo_coeff' in kwargs:
                mo_coeff = kwargs.pop('mo_coeff')
            elif 'orbital_coefficients' in kwargs:
                mo_coeff = kwargs.pop('orbital_coefficients')
            if 'spin' in kwargs:
                spin = kwargs.pop('spin')
            elif 'multiplicity' in kwargs:
                spin = (kwargs.pop('multiplicity') - 1) / 2
            if int1e is None:
                raise TequilaException('Not enough molecular data provided')
            if mo_coeff is None:
                mo_coeff = eye(len(int1e))
            if 'n_elec' in kwargs:
                n_elec = kwargs.pop('n_elec')
                n_alpha = int((n_elec + 2 * spin) // 2)
                n_beta = int((n_elec - 2 * spin) // 2)
            elif 'n_electrons' in kwargs:
                n_elec = kwargs.pop('n_electrons')
                n_alpha = int((n_elec + 2 * spin) // 2)
                n_beta = int((n_elec - 2 * spin) // 2)
            elif 'n_alpha' in kwargs and 'n_beta' in kwargs:
                n_alpha = kwargs.pop('n_alpha')
                n_beta = kwargs.pop('n_beta')
                n_elec = n_alpha + n_beta
            else:
                raise TequilaException('No manner of defining the amount of electrons provided')
            if n_alpha is None and n_elec is not None:
                n_alpha = int(n_elec // 2)
                n_beta = int(n_elec - n_alpha)
            if 'point_group' in kwargs:
                point_group = kwargs.pop('point_group')
            elif 'symmetry' in kwargs:
                point_group = kwargs.pop('symmetry')
            self.core_energy = e_core
            self.one_body_integrals = array(int1e, dtype=complex128)
            self.two_body_integrals = array(int2e, dtype=complex128) if int2e is not None else None
            self.mo_coeff = array(mo_coeff, dtype=complex128)
            self.norb = len(int1e)
            self.spin = spin
            self.n_alpha = n_alpha
            self.n_beta = n_beta
            self.symmetry = point_group
            self.atom = None
            self.basis = None
            self.active_space = [*range(len(int1e))]

        if any(i is None for i in [self.two_body_integrals, self.one_body_integrals, self.n_alpha, self.n_beta]):
            raise TequilaException('Not enough molecular data provided')

        self.hamiltonian = self._build_hamiltonian()

        occupied = [i for i in range(self.n_alpha)] + [self.norb + i for i in range(self.n_beta)]
        hf_state = {sum(1 << o for o in occupied): 1.0}
        self._init_state_bra = hf_state
        self._init_state_ket = hf_state

        if 'name' in kwargs:
            self._name = kwargs.pop('name')
        else:
            self._name = 'Expectation Value'
        if operator is None:
            operator = 'H'
        if isinstance(operator, str) and operator == 'I':
            self._name = 'Transition Element'
        self.operator = self.build_operator(operator)

    def _build_hamiltonian(self) -> List[spex.FermionTerm]:
        # Integrals in chemists notation g[p,q,r,s]=(pq|rs); spatial orbital p maps to
        # spin orbitals p (alpha) and p+norb (beta) in up-then-down ordering.
        norb = self.norb
        terms: List[spex.FermionTerm] = []
        if self.core_energy is not None and abs(self.core_energy) > _ZERO_TOL:
            terms.append(spex.FermionTerm([], [], complex(self.core_energy)))
        if self.one_body_integrals is not None:
            h = array(self.one_body_integrals, dtype=complex128)
            for p in range(norb):
                for q in range(norb):
                    if abs(h[p, q]) < _ZERO_TOL:
                        continue
                    terms.append(spex.FermionTerm([p], [q], h[p, q]))
                    terms.append(spex.FermionTerm([p + norb], [q + norb], h[p, q]))
        if self.two_body_integrals is not None:
            g = array(self.two_body_integrals, dtype=complex128)
            for p in range(norb):
                for q in range(norb):
                    for r in range(norb):
                        for s in range(norb):
                            w = 0.5 * g[p, r, q, s]
                            if abs(w) < _ZERO_TOL:
                                continue
                            terms.append(spex.FermionTerm([q, p], [r, s], w))
                            terms.append(spex.FermionTerm([q + norb, p + norb], [r + norb, s + norb], w))
                            terms.append(spex.FermionTerm([q + norb, p], [r, s + norb], w))
                            terms.append(spex.FermionTerm([q, p + norb], [r + norb, s], w))
        return terms

    @property
    def init_state(self) -> tuple:
        return self.__civect_to_qwvf(self._init_state_bra), self.__civect_to_qwvf(self._init_state_ket)

    @init_state.setter
    def init_state(self, init_state: QubitWaveFunction):
        self._init_state_bra = self.__qwvf_to_civect(init_state)
        self._init_state_ket = self.__qwvf_to_civect(init_state)

    def __civect_to_qwvf(self, state_dict: dict) -> QubitWaveFunction:
        return _civect_to_qwvf(state_dict, 2 * self.norb)

    def __qwvf_to_civect(self, wvf: QubitWaveFunction) -> dict:
        return _qwvf_to_civect(wvf, 2 * self.norb)

    def build_operator(self, operator: Union[str, FermionOperator, QubitHamiltonian] = None) -> Union[None, Callable, List[spex.FermionTerm]]:
        # Supported: "I"/"H", openfermion FermionOperator, tequila QubitHamiltonian (callable on state dict)

        def from_string(operator: str) -> Union[Callable, List[spex.FermionTerm]]:
            if operator.upper() == "I":
                return lambda x: x
            elif operator.upper() == "H":
                return self.hamiltonian
            else:
                raise TequilaException(f"No operator str {operator} supported on SpexExpval")

        if isinstance(operator, str):
            operator = from_string(operator)
        elif isinstance(operator, Number):
            operator = [spex.FermionTerm([], [], complex(operator))]
        elif isinstance(operator, FermionOperator):
            # openfermion's terms dict already merges identical terms
            terms: List[spex.FermionTerm] = []
            for term, weight in operator.terms.items():
                if abs(weight) < _ZERO_TOL:
                    continue
                if len(term) == 0:
                    terms.append(spex.FermionTerm([], [], weight))
                    continue
                # openfermion terms act right-to-left; spex applies annihilation then creation
                # in vector order, so both lists are the reversed tuple order.
                creation = [i for i, a in term if a == 1]
                annihilation = [i for i, a in term if a == 0]
                # openfermion numbers spin-orbitals interleaved (even=alpha, odd=beta);
                # spex works internally in up-then-down ordering, remap the indices.
                creation = [i // 2 + (i % 2) * self.norb for i in creation]
                annihilation = [i // 2 + (i % 2) * self.norb for i in annihilation]
                terms.append(spex.FermionTerm(list(reversed(creation)), list(reversed(annihilation)), weight))
            operator = terms
        elif isinstance(operator, QubitHamiltonian):
            qubit_operator = operator

            def f(state_dict):
                wvf = self.__civect_to_qwvf(state_dict)
                wvf = wvf.apply_qubitoperator(qubit_operator)
                return self.__qwvf_to_civect(wvf)

            operator = f
        else:
            raise TequilaException(f"No operator {type(operator).__name__} supported")

        return operator

    def __call__(self, variables=None, *args, **kwargs) -> float:
        return self.simulate(variables=variables)

    def simulate(self, variables=None) -> float:
        state = self._init_state_ket
        if callable(self.operator):
            state = self.operator(state)
        if isinstance(self.operator, list):
            result = spex.expectation_value_fermionic(state, state, self.operator)
        else:
            result = spex.expectation_value_fermionic(state, state, [spex.FermionTerm([], [], 1.0)])
        return float(real(result))


def _qwvf_to_civect(wvf: QubitWaveFunction, n_qubits: int) -> dict:
    wvf.n_qubits = n_qubits
    result = {}
    for bs, coeff in wvf.items():
        key = int(bs)
        if wvf.numbering != BitNumbering.LSB:
            key = reverse_int_bits(key, n_qubits)
        if abs(coeff) > _ZERO_TOL:
            result[key] = complex(coeff)
    return result


def _civect_to_qwvf(state_dict: dict, n_qubits: int) -> QubitWaveFunction:
    wvf = QubitWaveFunction(n_qubits=n_qubits, numbering=BitNumbering.LSB)
    wvf._state = {
        int(k): complex(v) for k, v in (state_dict or {}).items() if abs(v) > _ZERO_TOL
    }
    return wvf
