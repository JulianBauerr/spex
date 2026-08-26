"""Expectation value wrapper <bra|O|ket> around the spex_tequila C++ simulator; states are sparse dict[int, complex]."""
import spex_tequila as spex

from sunrise.fermionic_operations.circuit import FCircuit

from tequila import TequilaException, QubitWaveFunction, Variable, QubitHamiltonian
from tequila.objective.objective import Variables
from tequila.quantumchemistry.chemistry_tools import NBodyTensor
from tequila.quantumchemistry import qc_base
from tequila.utils.bitstrings import BitNumbering, reverse_int_bits
from numpy import eye, ndarray, array, complex128, real, einsum, argwhere
from pyscf.gto import Mole
from pyscf.scf import RHF
from openfermion import FermionOperator
from typing import Union, List, Callable


_ZERO_TOL = 1e-12


def _restore_eri(mf) -> ndarray:
    # pyscf stores ERIs packed in 2D; ao2mo.restore recovers the 4-index (pq|rs) AO tensor.
    from pyscf import ao2mo

    eri2d = mf._eri
    if eri2d is None:
        eri2d = mf.with_df.get_eri() if getattr(mf, 'with_df', None) is not None else None
        if eri2d is None:
            raise TequilaException('No two-electron integrals available on the SCF object')
    if eri2d.ndim != 4:
        nao = mf.mol.nao_nr()
        return array(ao2mo.restore(1, eri2d, nao), dtype=complex128)
    return array(eri2d, dtype=complex128)


class SpexExpval:
    def __init__(
        self,
        bra: Union['FCircuit', None] = None,
        ket: Union['FCircuit', None] = None,
        operator: Union[str, FermionOperator, List[FermionOperator]] = None,
        backend_kwargs: dict = {},
        *args,
        **kwargs,
    ):
        self.operator = None
        self.hamiltonian: List[spex.FermionTerm] = []
        self._init_state_bra: dict = None
        self._init_state_ket: dict = None
        self._ket = None
        self._bra = None
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

        if 'circuit' in kwargs:
            if ket is not None:
                raise TequilaException('Two circuits provided?')
            ket = kwargs.pop('circuit')
        if 'U' in kwargs:
            if ket is not None:
                raise TequilaException('Two circuits provided?')
            ket = kwargs.pop('U')
        if 'H' in kwargs:
            if operator is not None:
                raise TequilaException('Two operators provided?')
            operator = kwargs.pop('H')
        if 'mol' in kwargs:
            if 'molecule' in kwargs and kwargs['molecule']:
                raise TequilaException('Two molecules provided?')
            kwargs['molecule'] = kwargs.pop('mol')

        run_hf = (bra is None or bra.initial_state is None) and (ket is None or ket.initial_state is None)

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
            elif isinstance(molecule, Mole):
                mf = RHF(mol=molecule)
                mf.kernel()
                self.mo_coeff = array(mf.mo_coeff, dtype=complex128)
                self.core_energy = float(mf.energy_nuc())
                self.one_body_integrals = einsum('pi,pq,qj->ij', self.mo_coeff, array(mf.get_hcore(), dtype=complex128), self.mo_coeff)
                eri_ao = _restore_eri(mf)
                self.two_body_integrals = einsum(
                    'pqrs,pi,qj,rk,sl->ijkl', eri_ao, *[self.mo_coeff] * 4
                )
                self.norb = self.mo_coeff.shape[1]
                nelec = molecule.nelec
                if isinstance(nelec, (tuple, list)):
                    self.n_alpha, self.n_beta = int(nelec[0]), int(nelec[1])
                else:
                    self.n_alpha = int(nelec // 2)
                    self.n_beta = int(nelec - self.n_alpha)
                self.spin = molecule.spin / 2
                self.symmetry = getattr(molecule, 'symmetry', None)
                self.atom = getattr(molecule, 'atom', None)
                self.basis = getattr(molecule, 'basis', None)
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
            elif ket is not None and ket.initial_state is not None:
                if isinstance(ket.initial_state._state, dict):
                    n_elec = bin([*ket.initial_state._state.keys()][0])[2:].count('1')
                else:
                    n_elec = bin(argwhere(ket.initial_state._state > 1.e-6)[0][0])[2:].count('1')
                n_alpha = int((n_elec + 2 * spin) // 2)
                n_beta = int((n_elec - 2 * spin) // 2)
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

        if run_hf:
            occupied = [i for i in range(self.n_alpha)] + [self.norb + i for i in range(self.n_beta)]
            hf_state = {sum(1 << o for o in occupied): 1.0}
            self._init_state_bra = hf_state
            self._init_state_ket = hf_state

        if ket is not None:
            self.ket = ket
        if bra is not None:
            self.bra = bra
        if 'name' in kwargs:
            self._name = kwargs.pop('name')
        else:
            self._name = 'Expectation Value' if self.is_diagonal else 'Transition Value'
        if isinstance(operator, str) and operator == 'I':
            self._name = 'Transition Element'
        if operator is not None:
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
                            w = 0.5 * g[p, q, r, s]
                            if abs(w) < _ZERO_TOL:
                                continue
                            terms.append(spex.FermionTerm([q, p], [r, s], w))
                            terms.append(spex.FermionTerm([q + norb, p + norb], [r + norb, s + norb], w))
                            terms.append(spex.FermionTerm([q + norb, p], [r, s + norb], w))
                            terms.append(spex.FermionTerm([q, p + norb], [r + norb, s], w))
        return terms

    @property
    def bra(self) -> 'FCircuit':
        return self._bra

    @bra.setter
    def bra(self, bra):
        if not isinstance(bra, FCircuit):
            raise TypeError(f"FCircuit expected, received {type(bra).__name__}")
        if bra.initial_state is not None:
            self._init_state_bra = self.__qwvf_to_civect(bra.initial_state)
        bra = bra.to_upthendown(self.norb)
        self._bra = bra

    @property
    def ket(self) -> 'FCircuit':
        return self._ket

    @ket.setter
    def ket(self, ket):
        if not isinstance(ket, FCircuit):
            raise TypeError(f"FCircuit expected, received {type(ket).__name__}")
        if ket.initial_state is not None:
            self._init_state_ket = self.__qwvf_to_civect(ket.initial_state)
        ket = ket.to_upthendown(self.norb)
        self._ket = ket

    @property
    def variables(self) -> List[Variable]:
        bra = self.bra.variables if self.bra is not None else []
        ket = self.ket.variables if self.ket is not None else []
        return bra + ket

    @property
    def init_state(self) -> tuple:
        return self.__civect_to_qwvf(self._init_state_bra), self.__civect_to_qwvf(self._init_state_ket)

    @init_state.setter
    def init_state(self, init_state: QubitWaveFunction):
        self._init_state_bra = self.__qwvf_to_civect(init_state)
        self._init_state_ket = self.__qwvf_to_civect(init_state)

    @property
    def is_diagonal(self):
        return self.bra is None or self.ket is None or self.bra == self.ket

    @property
    def U(self):
        return self.ket if self.is_diagonal else [self.bra, self.ket]

    def __civect_to_qwvf(self, state_dict: dict) -> QubitWaveFunction:
        wvf = QubitWaveFunction(n_qubits=2 * self.norb, numbering=BitNumbering.LSB)
        wvf._state = {
            int(k): complex(v) for k, v in (state_dict or {}).items() if abs(v) > _ZERO_TOL
        }
        return wvf

    def __qwvf_to_civect(self, wvf: QubitWaveFunction) -> dict:
        wvf.n_qubits = 2 * self.norb
        result = {}
        for bs, coeff in wvf.items():
            key = int(bs)
            if wvf.numbering != BitNumbering.LSB:
                key = reverse_int_bits(key, 2 * self.norb)
            if abs(coeff) > _ZERO_TOL:
                result[key] = complex(coeff)
        return result

    def build_operator(self, operator: Union[str, FermionOperator, QubitHamiltonian] = None) -> Union[None, Callable, List[spex.FermionTerm]]:
        # Supported: "I"/"H", openfermion FermionOperator, tequila QubitHamiltonian (callable on state dict)

        def from_string(operator: str) -> Union[Callable, List[spex.FermionTerm]]:
            if operator.upper() == "I":
                return lambda x: x
            elif operator.upper() == "H":
                return self.hamiltonian
            else:
                raise TequilaException(f"No operator str {operator} supported on Spex BraKet")

        if isinstance(operator, str):
            operator = from_string(operator)
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

    def __call__(self, variables: Union[list, dict, Variables] = {}, *args, **kwargs) -> float:
        return self.simulate(variables=variables)

    def simulate(self, variables: Union[list, dict, Variables] = None) -> float:
        variables = {} if variables is None else variables
        if isinstance(variables, Variables):
            variables = variables.store
        if not isinstance(variables, dict):
            raise TequilaException(f'variables must be a dict or tequila Variables, received {type(variables).__name__}')

        # Normalize keys: Variable objects and their names are both accepted.
        dvars = {}
        for k, val in variables.items():
            name = getattr(k, 'name', k)
            dvars[name] = val
            if isinstance(k, Variable):
                dvars[k] = val

        check_variables = {k: (k in dvars or getattr(k, 'name', None) in dvars) for k in self.extract_variables()}
        if not all(check_variables.values()):
            missing = [k for k, v in check_variables.items() if not v]
            raise TequilaException(f'Objective did not receive all variables: {variables} given, missing {missing}')

        ket_state = self._apply_circuit(self._init_state_ket, self.ket, dvars)
        if self.is_diagonal:
            bra_state = ket_state
        else:
            bra_state = self._apply_circuit(self._init_state_bra, self.bra, dvars)

        if isinstance(self.operator, list):
            result = spex.expectation_value_fermionic(bra_state, ket_state, self.operator)
        elif callable(self.operator):
            result = spex.inner_product(bra_state, self.operator(ket_state))
        else:
            result = spex.inner_product(bra_state, ket_state)  # no operator: plain <bra|ket>
        return float(real(result))

    def extract_variables(self) -> List[Variable]:
        variables_bra = []
        variables_ket = []
        uniques_bra = []
        if self.bra is not None:
            variables_bra = self.bra.extract_variables()
        if self.ket is not None:
            variables_ket = self.ket.extract_variables()
        for v in variables_bra:
            if v not in variables_ket:
                uniques_bra.append(v)
        return uniques_bra + variables_ket

    def _apply_circuit(self, state: dict, circuit, dvars: dict) -> dict:
        # each gate maps to FermionTerm(creation_idx=[from...], annihilation_idx=[to...], 1j)
        state = {} if state is None else state
        result = {int(k): complex(v) for k, v in state.items() if abs(v) > _ZERO_TOL}
        if circuit is None:
            return result
        gates = circuit.gates
        for gate in gates:
            indices = gate.indices
            parameter = gate.variables
            if not indices:
                continue
            if isinstance(parameter, Variable):
                name = getattr(parameter, 'name', None)
                if name is not None and name in dvars:
                    value = dvars[name]
                elif parameter in dvars:
                    value = dvars[parameter]
                else:
                    value = parameter.map_variables(dvars)
                if isinstance(value, Variable):
                    value = value.map_variables(dvars)
                theta = float(value)
            else:
                theta = float(parameter)
            for term_pairs in indices:
                creation = [p[0] for p in term_pairs]
                annihilation = [p[1] for p in term_pairs]
                term = spex.FermionTerm(creation, annihilation, 1.0j)
                result = spex.apply_abstract_generator(result, [term], theta)
        return result
