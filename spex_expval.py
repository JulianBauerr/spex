"""
spex_expval.py
==============

Expectation value wrapper around the ``spex_tequila`` C++ fermionic simulator.

Drop-in replacement for ``ffsim_expval.FFS_EXPVAL`` from the sunrise project:
identical constructor signature, kwargs aliases, properties and methods, but all
fermionic operations are evaluated with ``spex_tequila`` on sparse states.

Internal state representation is a sparse ``dict[int, complex]`` mapping Fock
bitstrings to amplitudes. Spin-orbital ordering is up-then-down: spatial orbital
``p`` corresponds to the spin orbitals ``2p`` (alpha) and ``2p+1`` (beta).
"""
import spex_tequila as spex

try:
    from ..fermionic_operations.circuit import FCircuit
except ImportError:  # standalone usage outside the sunrise package
    FCircuit = None

from tequila import (
    TequilaException,
    Molecule,
    QubitWaveFunction,
    simulate,
    Variable,
    Objective,
    assign_variable,
    QubitHamiltonian,
)
from tequila.objective.objective import Variables, FixedVariable
from tequila.quantumchemistry.chemistry_tools import NBodyTensor
from tequila.quantumchemistry import qc_base
from tequila.utils.bitstrings import BitString, BitNumbering, reverse_int_bits
from numbers import Number
from numpy import (
    ceil,
    argwhere,
    pi,
    prod,
    eye,
    zeros,
    isclose,
    allclose,
    ndarray,
    array,
    complex128,
    real,
    vdot,
    einsum,
)
from pyscf.gto import Mole
from pyscf.scf import RHF
from openfermion import FermionOperator
from copy import deepcopy
from typing import Union, List, Tuple, Callable
from collections import defaultdict


def _restore_eri(mf) -> ndarray:
    """
    Recover the 4-index (pq|rs) AO two-electron tensor from a pyscf SCF
    object. pyscf stores the ERIs in a packed 2D form, so the tensor has to
    be restored with ao2mo before being transformed to the MO basis.
    """
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


def _is_circuit_like(x) -> bool:
    """Duck-typed circuit check used when FCircuit is unavailable."""
    return hasattr(x, 'gates') or (
        hasattr(x, '__iter__') and not isinstance(x, (str, bytes, dict))
    )


def map_variables(x: Union[Variable, Objective], dvariables: dict):
    """Map variables of a tequila Variable or Objective onto the given values."""
    if isinstance(x, Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x, Objective):
        x = simulate(x, dvariables)
    return x


class SpexExpval:
    """Expectation value ``<bra|O|ket>`` evaluated with the spex_tequila C++ backend."""

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

        # Molecular data
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
            circuit = kwargs['circuit']
            kwargs.pop('circuit')
            if ket is not None:
                raise TequilaException('Two circuits provided?')
            else:
                ket = circuit
        if 'U' in kwargs:
            U = kwargs['U']
            kwargs.pop('U')
            if ket is not None:
                raise TequilaException('Two circuits provided?')
            else:
                ket = U
        if 'H' in kwargs:
            H = kwargs['H']
            kwargs.pop('H')
            if operator is not None:
                raise TequilaException('Two operators provided?')
            else:
                operator = H
        if 'mol' in kwargs:
            if 'molecule' in kwargs and kwargs['molecule']:
                raise TequilaException("Two molecules provided?")
            kwargs['molecule'] = kwargs['mol']
            kwargs.pop('mol')

        run_hf = (bra is None or getattr(bra, 'initial_state', None) is None) and (ket is None or getattr(ket, 'initial_state', None) is None)

        if 'molecule' in kwargs and kwargs['molecule']:
            molecule = kwargs['molecule']
            kwargs.pop('molecule')
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
                self.one_body_integrals = einsum(
                    'pi,pq,qj->ij', self.mo_coeff, array(mf.get_hcore(), dtype=complex128), self.mo_coeff
                )
                eri_ao = _restore_eri(mf)
                self.two_body_integrals = einsum(
                    'pqrs,pi,qj,rk,sl->ijkl',
                    eri_ao,
                    self.mo_coeff,
                    self.mo_coeff,
                    self.mo_coeff,
                    self.mo_coeff,
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
            integral = kwargs['integral_manager']
            params = kwargs['parameters']
            kwargs.pop('integral_manager')
            kwargs.pop('parameters')
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
                self.symmetry = kwargs['point_group']
                kwargs.pop('point_group')
            elif 'symmetry' in kwargs:
                self.symmetry = kwargs['symmetry']
                kwargs.pop('symmetry')
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
                int1e = kwargs['int1e']
                kwargs.pop('int1e')
            elif "one_body_integrals" in kwargs:
                int1e = kwargs['one_body_integrals']
                kwargs.pop('one_body_integrals')
            elif "h" in kwargs:
                int1e = kwargs['h']
                kwargs.pop('h')
            if 'int2e' in kwargs:
                int2e = kwargs['int2e']
                kwargs.pop('int2e')
            elif 'two_body_integrals' in kwargs:
                int2e = kwargs['two_body_integrals']
                kwargs.pop('two_body_integrals')
            elif 'g' in kwargs:
                int2e = kwargs['g']
                kwargs.pop('g')
            if isinstance(int2e, NBodyTensor):
                int2e = int2e.reorder('chem').elems
            if 'e_core' in kwargs:
                e_core = kwargs['e_core']
                kwargs.pop('e_core')
            elif 'constant_term' in kwargs:
                e_core = kwargs['constant_term']
                kwargs.pop('constant_term')
            elif 'constant' in kwargs:
                e_core = kwargs['constant']
                kwargs.pop('constant')
            elif 'c' in kwargs:
                e_core = kwargs['c']
                kwargs.pop('c')
            else:
                e_core = 0.0
            if 'mo_coeff' in kwargs:
                mo_coeff = kwargs['mo_coeff']
                kwargs.pop('mo_coeff')
            elif 'orbital_coefficients' in kwargs:
                mo_coeff = kwargs['orbital_coefficients']
                kwargs.pop('orbital_coefficients')
            if 'spin' in kwargs:
                spin = kwargs['spin']
                kwargs.pop('spin')
            elif 'multiplicity' in kwargs:
                spin = (kwargs['multiplicity'] - 1) / 2
                kwargs.pop('multiplicity')
            if int1e is None:
                raise TequilaException('Not enough molecular data provided')
            if mo_coeff is None:
                mo_coeff = eye(len(int1e))
            if 'n_elec' in kwargs:
                n_elec = kwargs['n_elec']
                kwargs.pop('n_elec')
                n_alpha = int((n_elec + 2 * spin) // 2)
                n_beta = int((n_elec - 2 * spin) // 2)
            elif 'n_electrons' in kwargs:
                n_elec = kwargs['n_electrons']
                kwargs.pop('n_electrons')
                n_alpha = int((n_elec + 2 * spin) // 2)
                n_beta = int((n_elec - 2 * spin) // 2)
            elif 'n_alpha' in kwargs and 'n_beta' in kwargs:
                n_alpha = kwargs['n_alpha']
                n_beta = kwargs['n_beta']
                kwargs.pop('n_alpha')
                kwargs.pop('n_beta')
                n_elec = n_alpha + n_beta
            elif ket is not None and ket.initial_state is not None:
                wvf = ket.initial_state
                n_elec = None
                for bs, _ in wvf.items():
                    n_elec = bin(int(bs))[2:].count('1')
                    break
                if n_elec is None:
                    raise TequilaException('No manner of defining the amount of electrons provided')
                n_alpha = int((n_elec + 2 * spin) // 2)
                n_beta = int((n_elec - 2 * spin) // 2)
            else:
                raise TequilaException('No manner of defining the amount of electrons provided')
            if n_alpha is None and n_elec is not None:
                n_alpha = int(n_elec // 2)
                n_beta = int(n_elec - n_alpha)
            if 'point_group' in kwargs:
                point_group = kwargs['point_group']
                kwargs.pop('point_group')
            elif 'symmetry' in kwargs:
                point_group = kwargs['symmetry']
                kwargs.pop('symmetry')
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

        if any(
            i is None for i in [self.two_body_integrals, self.one_body_integrals, self.n_alpha, self.n_beta]
        ):
            raise TequilaException('Not enough molecular data provided')

        self.hamiltonian = self._build_hamiltonian()

        if run_hf:
            occupied = [2 * i for i in range(self.n_alpha)] + [2 * i + 1 for i in range(self.n_beta)]
            hf_state = {sum(1 << o for o in occupied): 1.0}
            self._init_state_bra = hf_state
            self._init_state_ket = hf_state

        if ket is not None:
            self.ket = ket
        if bra is not None:
            self.bra = bra
        if 'name' in kwargs:
            self._name = kwargs['name']
        else:
            self._name = 'Expectation Value' if self.is_diagonal else 'Transition Value'
        if isinstance(operator, str) and operator == 'I':
            self._name = 'Transition Element'
        if operator is not None:
            self.operator = self.build_operator(operator)

    def _build_hamiltonian(self) -> List[spex.FermionTerm]:
        """
        Build the molecular Hamiltonian as a list of spex.FermionTerm.

        The integrals are expected in chemists notation with
        ``g[p, q, r, s] = (pq|rs)``. Spin-orbital mapping is up-then-down:
        spatial orbital ``p`` maps to the spin orbitals ``2p`` (alpha) and
        ``2p+1`` (beta).

        One-body part:   h[p, q] a^dag_{p,s} a_{q,s}
        Two-body part:   0.5 * g[p, q, r, s] a^dag_{p,s} a^dag_{q,t} a_{s,t} a_{r,s}

        In spex convention the creation/annihilation lists are reversed with
        respect to the mathematical operator order, so the two-body term reads
        ``FermionTerm([2q+t, 2p+s], [2r+s, 2s+t], 0.5*g[p, q, r, s])``.
        """
        terms: List[spex.FermionTerm] = []
        if self.core_energy is not None and abs(self.core_energy) > 1e-12:
            terms.append(spex.FermionTerm([], [], complex(self.core_energy)))
        if self.one_body_integrals is not None:
            h = array(self.one_body_integrals, dtype=complex128)
            for p in range(self.norb):
                for q in range(self.norb):
                    if abs(h[p, q]) < 1e-12:
                        continue
                    terms.append(spex.FermionTerm([2 * p], [2 * q], h[p, q]))
                    terms.append(spex.FermionTerm([2 * p + 1], [2 * q + 1], h[p, q]))
        if self.two_body_integrals is not None:
            g = array(self.two_body_integrals, dtype=complex128)
            for p in range(self.norb):
                for q in range(self.norb):
                    for r in range(self.norb):
                        for s in range(self.norb):
                            w = 0.5 * g[p, q, r, s]
                            if abs(w) < 1e-12:
                                continue
                            # same spin (alpha-alpha / beta-beta)
                            terms.append(spex.FermionTerm([2 * q, 2 * p], [2 * r, 2 * s], w))
                            terms.append(spex.FermionTerm([2 * q + 1, 2 * p + 1], [2 * r + 1, 2 * s + 1], w))
                            # mixed spins (alpha-beta / beta-alpha)
                            terms.append(spex.FermionTerm([2 * q + 1, 2 * p], [2 * r, 2 * s + 1], w))
                            terms.append(spex.FermionTerm([2 * q, 2 * p + 1], [2 * r + 1, 2 * s], w))
        return terms

    @property
    def bra(self) -> 'FCircuit':
        """
        Excitation operators applied to the bra.
        """
        return self._bra

    @bra.setter
    def bra(self, bra):
        '''
        Expected FCircuit.

        When FCircuit is available (sunrise package) a strict isinstance check
        is enforced. In standalone mode (FCircuit is None) any object with the
        circuit interface (gates, or iterable of gates) is accepted via
        duck-typing.
        '''
        if FCircuit is not None:
            if not isinstance(bra, FCircuit):
                raise TypeError(f"FCircuit expected, received {type(bra).__name__}")
        elif not _is_circuit_like(bra):
            raise TypeError(f"FCircuit expected, received {type(bra).__name__}")
        if getattr(bra, 'initial_state', None) is not None:
            self.init_state_bra = bra.initial_state
        if hasattr(bra, 'to_upthendown'):
            bra = bra.to_upthendown(self.norb)
        self._bra = bra

    @property
    def ket(self) -> 'FCircuit':
        """
        Excitation operators applied to the ket.
        """
        return self._ket

    @ket.setter
    def ket(self, ket):
        '''
        Expected FCircuit

        When FCircuit is available (sunrise package) a strict isinstance check
        is enforced. In standalone mode (FCircuit is None) any object with the
        circuit interface (gates, or iterable of gates) is accepted via
        duck-typing.
        '''
        if FCircuit is not None:
            if not isinstance(ket, FCircuit):
                raise TypeError(f"FCircuit expected, received {type(ket).__name__}")
        elif not _is_circuit_like(ket):
            raise TypeError(f"FCircuit expected, received {type(ket).__name__}")
        if getattr(ket, 'initial_state', None) is not None:
            self.init_state_ket = ket.initial_state
        if hasattr(ket, 'to_upthendown'):
            ket = ket.to_upthendown(self.norb)
        self._ket = ket

    @property
    def variables_bra(self) -> List[Variable]:
        """Tequila Circuit Bra variables."""
        return self.bra.variables if self.bra is not None else []

    @property
    def variables_ket(self) -> List[Variable]:
        """Tequila circuit Ket parameters."""
        return self.ket.variables if self.ket is not None else []

    @property
    def variables(self) -> List[Variable]:
        """Tequila circuit variables."""
        return self.variables_bra + self.variables_ket

    @property
    def params_bra(self) -> List[Variable]:
        """Parameters of the bra circuit."""
        return self.variables_bra

    @property
    def params_ket(self) -> List[Variable]:
        """Parameters of the ket circuit."""
        return self.variables_ket

    @property
    def init_state_bra(self) -> QubitWaveFunction:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.
        """
        return self.__civect_to_qwvf(self._init_state_bra)

    @init_state_bra.setter
    def init_state_bra(self, init_state_bra: QubitWaveFunction):
        self._init_state_bra = self.__qwvf_to_civect(init_state_bra)

    @property
    def init_state_ket(self) -> QubitWaveFunction:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.
        """
        return self.__civect_to_qwvf(self._init_state_ket)

    @init_state_ket.setter
    def init_state_ket(self, init_state_ket: QubitWaveFunction):
        self._init_state_ket = self.__qwvf_to_civect(init_state_ket)

    @property
    def init_state(self) -> Tuple[QubitWaveFunction, QubitWaveFunction]:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.
        """
        return self.init_state_bra, self.init_state_ket

    @init_state.setter
    def init_state(self, init_state: QubitWaveFunction):
        self.init_state_bra = init_state
        self.init_state_ket = init_state

    def __str__(self):
        res = ''
        if self.is_diagonal:
            if self.ket is not None:
                res += f"{self._name} with indices: {self.ket.extract_indices()} with variables {self.params_ket}"
            elif self.bra is not None:
                res += f"{self._name} with indices: {self.bra.extract_indices()} with variables {self.params_bra}"
            else:
                res += str(self._name)
        else:
            res += f"{self._name} with Bra= {self.bra.extract_indices()} with variables {self.params_bra}\n"
            res += f"{len(self._name) * ' '} with Ket= {self.ket.extract_indices()} with variables {self.params_ket}"
        return res

    def __repr__(self):
        return self.__str__()

    @property
    def is_diagonal(self):
        if self.bra is None:
            return True
        if self.ket is None:
            return True
        if self.bra is not None and self.bra == self.ket:
            return True
        return False

    @property
    def U(self):
        'Dummy function to work with tequila Objectives'
        if self.is_diagonal:
            return self.ket
        else:
            return [self.bra, self.ket]

    def count_measurements(self) -> int:
        mes = 0
        if self.one_body_integrals is not None:
            mes += prod(self.one_body_integrals)
        if self.two_body_integrals is not None:
            mes += prod(self.two_body_integrals)
        if mes:
            return mes
        else:
            return 1  # single HF determinant

    def __civect_to_qwvf(self, state_dict: dict) -> QubitWaveFunction:
        """
        Convert a sparse CI vector (dict bitstring -> amplitude) to a tequila
        QubitWaveFunction in LSB numbering.
        """
        wvf = QubitWaveFunction(n_qubits=2 * self.norb, numbering=BitNumbering.LSB)
        # QubitWaveFunction has no public sparse constructor; for dense=False
        # its internal state is a plain dict keyed by basis-state ints in the
        # wavefunction's own numbering.
        wvf._state = {
            int(k): complex(v) for k, v in (state_dict or {}).items() if abs(v) > 1e-12
        }
        return wvf

    def __qwvf_to_civect(self, wvf: QubitWaveFunction) -> dict:
        """Convert a tequila QubitWaveFunction to a sparse CI vector in LSB numbering."""
        wvf.n_qubits = 2 * self.norb
        result = {}
        for bs, coeff in wvf.items():
            key = int(bs)
            if wvf.numbering != BitNumbering.LSB:
                key = reverse_int_bits(key, 2 * self.norb)
            if abs(coeff) > 1e-8:
                result[key] = complex(coeff)
        return result

    def build_operator(self, operator: Union[str, FermionOperator, QubitHamiltonian] = None) -> Union[None, Callable, List[spex.FermionTerm]]:
        '''
        Build the expectation value operator.

        Supported inputs:
          - "I": the identity operator (returned as a callable)
          - "H": the molecular Hamiltonian (returned as a list of spex.FermionTerm)
          - an openfermion FermionOperator (converted to a list of spex.FermionTerm)
          - a tequila QubitHamiltonian (returned as a callable acting on the sparse state dict)

        Even if it is accepted a QubitHamiltonian, we disencourage its use here
        for fermionic states limitations. It will be applied to the CI vector
        and keeps these results which don't leave the CI vector.
        '''

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
                if abs(weight) < 1e-12:
                    continue
                if len(term) == 0:
                    terms.append(spex.FermionTerm([], [], weight))
                    continue
                # openfermion terms act right-to-left on the ket (rightmost
                # tuple first); spex applies annihilation in vector order and
                # then creation in vector order, so both lists are the
                # reversed tuple order.
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
        """
        Evaluate the expectation value for the given variables.

        :param variables: dict mapping variable names to values, or a tequila
            Variables object. May be None when the objective has no variables.
        :return: the real part of the expectation value as a plain float.
        """
        if variables is None:
            variables = {}
        if isinstance(variables, Variables):
            variables = variables.store
        if not isinstance(variables, dict):
            raise TequilaException(
                f"variables must be a dict or tequila Variables, received {type(variables).__name__}"
            )

        # Normalize keys: Variable objects and their names are both accepted.
        dvars = {}
        for k, val in variables.items():
            name = getattr(k, 'name', k)
            dvars[name] = val
            if isinstance(k, Variable):
                dvars[k] = val

        check_variables = {k: self._variable_in(k, dvars) for k in self.extract_variables()}
        if not all(list(check_variables.values())):
            raise TequilaException(
                "Objective did not receive all variables:\n"
                "You gave\n"
                " {}\n"
                " but the objective depends on\n"
                " {}\n"
                " missing values for\n"
                " {}".format(variables, self.extract_variables(), [k for k, v in check_variables.items() if not v])
            )

        ket_state = self._apply_circuit(self._init_state_ket, self.ket, dvars)
        if self.is_diagonal:
            # diagonal case: the same (ket) state is used on both sides,
            # mirroring the reference ``vdot(init_state_ket, O @ init_state_ket)``
            bra_state = ket_state
        else:
            bra_state = self._apply_circuit(self._init_state_bra, self.bra, dvars)

        if isinstance(self.operator, list):
            result = spex.expectation_value_fermionic(bra_state, ket_state, self.operator)
        elif callable(self.operator):
            result = spex.inner_product(bra_state, self.operator(ket_state))
        else:
            # no operator: plain transition element <bra|ket>
            result = spex.inner_product(bra_state, ket_state)
        return float(real(result))

    def extract_variables(self) -> List[Variable]:
        """
        Extract all variables on which the objective depends
        :return: List of all Variables
        """
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

    @staticmethod
    def _variable_in(variable: Variable, dvars: dict) -> bool:
        if variable in dvars:
            return True
        name = getattr(variable, 'name', None)
        return name in dvars

    def _apply_circuit(self, state: dict, circuit, dvars: dict) -> dict:
        """
        Apply an FCircuit of fermionic excitations to a sparse state dict.

        Each gate is translated into a spex.FermionTerm with weight 1j:
        gate.indices == [(from, to), ...] maps to
        FermionTerm(creation_idx=[from...], annihilation_idx=[to...], 1j),
        matching make_excitation_gate(indices) (see the excitation test suite).
        """
        if state is None:
            state = {}
        result = {int(k): complex(v) for k, v in state.items() if abs(v) > 1e-12}
        if circuit is None:
            return result
        gates = getattr(circuit, 'gates', None)
        if gates is None:
            gates = circuit  # duck-typing fallback: the circuit itself is iterable
        for gate in gates:
            indices = getattr(gate, 'indices', None)
            parameter = getattr(gate, 'parameter', 0.0)
            if indices is None or len(indices) == 0:
                continue
            theta = self._resolve_parameter(parameter, dvars)
            term = spex.FermionTerm([i for i, _ in indices], [j for _, j in indices], 1.0j)
            result = spex.apply_abstract_generator(result, [term], theta)
        return result

    @staticmethod
    def _resolve_parameter(parameter, dvars: dict) -> float:
        """Resolve a gate parameter (Variable or number) to a numeric angle."""
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
            return float(value)
        return float(parameter)
