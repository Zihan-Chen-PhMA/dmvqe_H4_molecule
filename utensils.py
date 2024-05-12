from typing import Union, Sequence, Any, Tuple, List
from ansatz_circ import *
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np

from numpy import ndarray



def Circuit_generator(ansatz: AnsCirc, params: ndarray,
                     num_electrons: int) -> QuantumCircuit:
    qcircuit = QuantumCircuit(ansatz.num_qubits)
    for i in range(int(num_electrons/2)):
        qcircuit.initialize([1,0],i)
        qcircuit.initialize([1,0],i+ansatz.num_orbits)
    for i in range(int(num_electrons/2), ansatz.num_orbits):
        qcircuit.initialize([0,1],i)
        qcircuit.initialize([0,1],i+ansatz.num_orbits)
    # params = ParameterVector('θ', ansatz.num_params)
    for gate in ansatz.gate_sequences:
        # qcircuit = SingleGate_append(qcircuit, params, gate)
        if gate.gateType == 'H':
            qcircuit.h(gate.supp[0])
        elif gate.gateType == 'cx':
            qcircuit.cx(gate.supp[0], gate.supp[1])
        elif gate.gateType == 'Rxh':
            if gate.param_factor != None:
                qcircuit.rx(gate.param_factor*np.pi/2, gate.supp[0])
            else:
                qcircuit.rx(np.pi/2, gate.supp[0])
        elif gate.gateType == 'Rz':
            if gate.param_factor != None:
                qcircuit.rz(params[gate.paramId]*gate.param_factor,
                            gate.supp[0])
            else:
                qcircuit.rz(params[gate.paramId], gate.supp[0])
    return qcircuit
        
def Circuit_template_generator(ansatz: AnsCirc,
                     num_electrons: int) -> QuantumCircuit:
    qcircuit = QuantumCircuit(ansatz.num_qubits)
    for i in range(int(num_electrons/2)):
        qcircuit.initialize([1,0],i)
        qcircuit.initialize([1,0],i+ansatz.num_orbits)
    for i in range(int(num_electrons/2), ansatz.num_orbits):
        qcircuit.initialize([0,1],i)
        qcircuit.initialize([0,1],i+ansatz.num_orbits)
    params = ParameterVector('θ', ansatz.num_params)
    for gate in ansatz.gate_sequences:
        # qcircuit = SingleGate_append(qcircuit, params, gate)
        if gate.gateType == 'H':
            qcircuit.h(gate.supp[0])
        elif gate.gateType == 'cx':
            qcircuit.cx(gate.supp[0], gate.supp[1])
        elif gate.gateType == 'Rxh':
            if gate.param_factor != None:
                qcircuit.rx(gate.param_factor*np.pi/2, gate.supp[0])
            else:
                qcircuit.rx(np.pi/2, gate.supp[0])
        elif gate.gateType == 'Rz':
            if gate.param_factor != None:
                qcircuit.rz(params[gate.paramId]*gate.param_factor,
                            gate.supp[0])
            else:
                qcircuit.rz(params[gate.paramId], gate.supp[0])
    return qcircuit


def SingleGate_append(qcircuit: QuantumCircuit, params: ParameterVector,
                      gate: SingleGate) -> QuantumCircuit:
    if gate.gateType == 'H':
        qcircuit.h(gate.supp[0])
    elif gate.gateType == 'cx':
        qcircuit.cx(gate.supp[0], gate.supp[1])
    elif gate.gateType == 'Rxh':
        if gate.param_factor != None:
            qcircuit.rx(gate.param_factor*np.pi/2, gate.supp[0])
        else:
            qcircuit.rx(np.pi/2, gate.supp[0])
    elif gate.gateType == 'Rz':
        if gate.param_factor != None:
            qcircuit.rz(params[gate.paramId]*gate.param_factor,
                         gate.supp[0])
        else:
            qcircuit.rz(params[gate.paramId], gate.supp[0])
    return qcircuit



class Hamiltonian():

    def __init__(self, filename: str) -> None:
        self.hamiltonian = self._load_from_file(filename)
        self.num_qubits = int(np.log2(len(self.hamiltonian)))
        self.degeneracy: int = 0
        self.gs_energy = None
        self.gs_dm = None

        
    
    def _load_from_file(self, filename: str) -> ndarray:
        sigma_I = np.array([[1,0],[0,1]])
        sigma_x = np.array([[0,1],[1,0]])
        sigma_y = np.array([[0,-1j],[1j,0]])
        sigma_z = np.array([[1,0],[0,-1]])
        ham_from_file = np.loadtxt(filename)
        ret_mat = 0
        for row in ham_from_file:
            pauli_temp = 1
            row_flip = np.flip(row[1:])     # qiskit has a wierd order.
            for pauli_id in row_flip:
                if int(pauli_id) == 0:
                    pauli_temp = np.kron(pauli_temp, sigma_I)
                elif int(pauli_id) == 1:
                    pauli_temp = np.kron(pauli_temp, sigma_x)
                elif int(pauli_id) == 2:
                    pauli_temp = np.kron(pauli_temp, sigma_y)
                elif int(pauli_id) == 3:
                    pauli_temp = np.kron(pauli_temp, sigma_z)
                else:
                    raise ValueError('Unsupported pauli id.')
            ret_mat = ret_mat + row[0]*pauli_temp
        return ret_mat


    def groundstate_energy(self) -> float:
        eigvls = np.linalg.eigvalsh(self.hamiltonian)
        degn = 0 
        self.gs_energy = eigvls[0]
        for item in eigvls:
            if abs(item-self.gs_energy) < 10**(-12):
                degn = degn +1
        self.degeneracy = degn
        return self.gs_energy

    def groundstate_dm(self) -> ndarray:
        eigvls, eigvectors = np.linalg.eigh(self.hamiltonian)
        if self.degeneracy > 1:
            warnings.warn('Ground state degeneracy > 1.')
        gs_state = eigvectors[:,0]
        # gs_dm = np.matmul(np.transpose(np.conjugate(
        #                                 np.atleast_2d(gs_state))),
        #                   np.atleast_2d(gs_state))
        gs_dm = np.matmul(np.reshape(gs_state,(-1,1)),
                          np.conjugate(np.reshape(gs_state,(1,-1))))
        self.gs_dm = gs_dm/np.trace(gs_dm)
        return self.gs_dm
    
    
    def lower_energies(self, num_levels: int) -> float:
        eigvls = np.linalg.eigvalsh(self.hamiltonian)
        return eigvls[:num_levels]





