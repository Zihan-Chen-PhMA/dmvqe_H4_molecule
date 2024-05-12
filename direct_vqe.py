from qiskit import quantum_info
from qiskit import *
from qiskit import Aer
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit_aer.noise import (NoiseModel, QuantumError, 
        pauli_error, kraus_error,
        phase_amplitude_damping_error,depolarizing_error)
# from opt_einsum import contract
from scipy.optimize import minimize
from scipy import optimize
from qiskit.quantum_info import SparsePauliOp

from typing import Callable
from numpy import ndarray
from joblib import Parallel, delayed
import timeit
from ansatz_circ import *
from utensils import *
import copy as cp

def ansatz_full(ansatz: AnsCirc, params: ndarray, 
                num_electrons: int,
                error: QuantumError | None) -> ndarray:
    if num_electrons%2 == 1:
        raise ValueError('Odd number of electrons is not '+
                         'supported,')
    qcircuit = Circuit_generator(ansatz, params,
                                  num_electrons)
    # qcircuit.assign_parameters(params)
    # qcircuit.draw()
    qcircuit.save_density_matrix()
    if error != None:
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(error, ['rx','rz','h'])
        two_qubit_error = error.tensor(error) 
        noise.add_all_qubit_quantum_error(two_qubit_error,['cx'])  
    else:
        noise = None   
    sim_noise = AerSimulator(method='density_matrix',
                             noise_model=noise, device='GPU')
    bqc_noise = transpile(qcircuit, sim_noise)
    result_dm = sim_noise.run(bqc_noise).result().data().get(
                            'density_matrix').data
    return result_dm

def ansatz_from_template(ansatz: AnsCirc, params: ndarray, 
                num_electrons: int,
                qcircuit: QuantumCircuit,
                error: QuantumError | None) -> ndarray:
    if num_electrons%2 == 1:
        raise ValueError('Odd number of electrons is not '+
                         'supported,')
    # qcircuit = Circuit_generator(ansatz, params,
    #                               num_electrons)
    qc = cp.deepcopy(qcircuit)
    qc.assign_parameters(params, inplace = True)
    # qcircuit.draw()
    qc.save_density_matrix()
    if error != None:
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(error, ['rx','rz','h'])
        two_qubit_error = error.tensor(error) 
        noise.add_all_qubit_quantum_error(two_qubit_error,['cx'])  
    else:
        noise = None   
    sim_noise = AerSimulator(method='density_matrix',
                             noise_model=noise, device='GPU')
    bqc_noise = transpile(qc, sim_noise)
    result_dm = sim_noise.run(bqc_noise).result().data().get(
                            'density_matrix').data
    return result_dm



class Dir_VQE():

    def __init__(self, hamiltonian: Hamiltonian, 
                 num_electrons: int,
                 ansatz: AnsCirc, error: QuantumError | None) -> None:
        self.num_qubits = hamiltonian.num_qubits
        self.hamiltonian = hamiltonian
        self.num_electrons = num_electrons
        self.ansatz = ansatz
        self.qcircuit = Circuit_template_generator(self.ansatz, 
                                          self.num_electrons)
        self.error = error
        self.groundstate_energy = self.hamiltonian.groundstate_energy()
        self.groundstate_dm = self.hamiltonian.groundstate_dm()
        self.callback = self._callback() 
        self.obj_fun = self._obj_function_from_tmp()
        self.diff_err = 10**(-5)
        self.stop_threshold = 10**(-5)
        self.jac = self._jac()
        self.iter = 0
        self.iter_list = []
        self.xn_iter_list = []
        self.energy_iter_list = []
        self.fidelity_iter_list = []


    def run(self) -> tuple[list, float, float]:
        x0 = 2*np.pi*np.random.rand(self.ansatz.num_params)
        vqe_result: optimize.OptimizeResult = minimize(
            fun=self.obj_fun,
            x0=x0,
            method='L-BFGS-B',
            jac=self.jac,
            callback=self.callback,
            bounds=None, 
            options={'maxfun':800000,'maxiter':1200,'gtol':1e-12,
                     'ftol':1e-14,'eps':1e-6,'iprint':99})
        vqe_fidelity = self._result_fidelity(vqe_result.x)
        return [vqe_result.x, vqe_result.fun, vqe_fidelity]
    
    def _obj_function_from_tmp(self) -> Callable:
        def obj_function(params: ndarray) -> float:
            dm_ans = ansatz_from_template(self.ansatz, params,
                                 self.num_electrons, 
                                 self.qcircuit, self.error) 
            ret_nom = np.trace(np.matmul(self.hamiltonian.hamiltonian,
                                          dm_ans))
            ret_den = np.trace(dm_ans)
            ret = np.real(ret_nom/ret_den)
            return ret
        return obj_function


    def _obj_function(self) -> Callable:
        def obj_function(params: ndarray) -> float:
            dm_ans = ansatz_full(self.ansatz, params,
                                 self.num_electrons, self.error) 
            ret_nom = np.trace(np.matmul(self.hamiltonian.hamiltonian,
                                          dm_ans))
            ret_den = np.trace(dm_ans)
            ret = np.real(ret_nom/ret_den)
            # time_start = timeit.default_timer()
            # if self.iter>102:
            #     if abs(self.energy_iter_list[-1]-
            #            self.energy_iter_list[-100])<(self.stop_threshold*
            #                         abs(self.energy_iter_list[-1])):
            #         return np.zeros(len(params))
            # param_id_list = Reshape([i for i in range(len(params))],
            #                         1)      # 4 for dir
            # print(param_id_list)
            # diff_ndlist = Parallel(n_jobs=25)(
            #                 delayed(self._Diff_multiple)(params, row_id) 
            #                     for row_id in param_id_list)
            # time_end = timeit.default_timer()
            # print('gradient calc time is: ', time_end-time_start)
            # jac = np.ndarray.flatten(np.array(diff_ndlist))
            return ret
        return obj_function
    
    def _result_fidelity(self, params) -> float:
        vqe_dm = ansatz_full(self.ansatz, params,
                             self.num_electrons, self.error)
        ret_nom = np.trace(np.matmul(self.groundstate_dm, vqe_dm))
        ret_den = np.trace(vqe_dm)
        ret = np.real(ret_nom/ret_den)
        return ret

    def _callback(self) -> Callable:
        def callback(xk) -> None:
            self.iter = self.iter + 1
            self.iter_list.append(self.iter)
            self.xn_iter_list.append(xk)
            energy = self.obj_fun(xk)
            self.energy_iter_list.append(energy)
            fidelity = self._result_fidelity(xk)
            self.fidelity_iter_list.append(fidelity)
            with open('dir_log_temp.txt','a') as f:
                f.write("iter:")
                f.write(str(self.iter))
                f.write('   ')
                write_line_list = [energy, fidelity]
                write_line_arr = np.array(write_line_list)
                write_line_arr = np.array([write_line_arr])
                np.savetxt(f,write_line_arr)

        return callback

    def _jac(self) -> Callable:
        def jac(x) -> ndarray:
            time_start = timeit.default_timer()
            if self.iter>102:
                if abs(self.energy_iter_list[-1]-
                       self.energy_iter_list[-100])<(self.stop_threshold*
                                    abs(self.energy_iter_list[-1])):
                    return np.zeros(len(x))
            param_id_list = Reshape([i for i in range(len(x))],
                                    1)      # 4 for dir
            diff_ndlist = Parallel(n_jobs=25)(
                            delayed(self._Diff_multiple)(x, row_id) 
                                for row_id in param_id_list)
            time_end = timeit.default_timer()
            print('gradient calc time is: ', time_end-time_start)
            return np.ndarray.flatten(np.array(diff_ndlist))
        return jac
    
    def _Diff_multiple(self, params: ndarray, row_id: list) -> list:
        ret_list = []
        for id in row_id:
            params_up = params + 0
            params_up[id] = params[id] + self.diff_err/2
            params_down = params + 0
            params_down[id] = params[id] - self.diff_err/2
            diff_id = (self.obj_fun(params_up)-
                       self.obj_fun(params_down))/self.diff_err
            ret_list.append(diff_id)
        return ret_list




def Reshape(ls: list, num_col: int) -> list:
    ret_list = []
    for row in range(int(len(ls)/num_col)):
        ret_list.append(ls[:num_col])
        del ls[:num_col]
    if len(ls) > 0:
        ret_list.append(ls)
    return ret_list
        

