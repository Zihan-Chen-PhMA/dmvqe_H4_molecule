from abc import ABC
from typing import Union, Sequence, Any, Tuple, List
import warnings
from numpy import array

# from utensils import Is_strictly_sorted
from circuit_blocks import *
from itertools import combinations

class AnsCirc(ABC):

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self.num_orbits = None
        self.param_id_list_dict = {}
        self.num_params = None
        self.gate_sequences = []


class Uccsd_Spin_Sym(AnsCirc):

    def __init__(self, num_qubits: int) -> None:
        if num_qubits%2 != 0:
            raise ValueError("The number of qubits should "
                             + "be even.")
        self.num_qubits = num_qubits
        self.num_orbits = int(self.num_qubits/2)
        self.param_id_list_dict = self._param_dict()
        self.num_params = sum([len(self.param_id_list_dict[i]) for i in 
                               self.param_id_list_dict])
        self.gate_sequences = self._calc_gate_sequences()


    def _param_dict(self) -> dict:
        # Dict keeps insertion order since Python 3.7
        num_param_dict = {}
        param_id_list_dict = {}
        num_param_dict['d_up'] = len([i for i in combinations(
                                      [j for j in range(0,self.num_orbits)], 
                                      4)])
        num_param_dict['d_down'] = num_param_dict['d_up']
        num_param_dict['d_mix'] = len([i for i in combinations(
                                      [j for j in range(0,self.num_orbits)], 
                                      2)])**2
        num_param_dict['s_up'] = len([i for i in combinations(
                                      [j for j in range(0,self.num_orbits)], 
                                      2)])
        num_param_dict['s_down'] = num_param_dict['s_up']    
        param_num = 0   
        for item in num_param_dict:
            param_num = param_num + num_param_dict[item]
        param_id_list = [i for i in range(param_num)]
        for item in num_param_dict:
            param_id_list_temp = param_id_list[:num_param_dict[item]]
            param_id_list_dict[item] = param_id_list_temp
            del param_id_list[:num_param_dict[item]]
        return param_id_list_dict


    def circ_d_up(self) -> List[SingleGate]:
        param_id_list = self.param_id_list_dict['d_up']
        qubit_anchor_list = [i for i in combinations(
                            [j for j in range(0,self.num_orbits)], 4)] 
        gate_sequence = [] 
        for param_id, qubit_anchor in zip(param_id_list,qubit_anchor_list):
            qubit_supp_l = [i for i in range(qubit_anchor[0],
                                             qubit_anchor[1]+1)]
            qubit_supp_r = [i for i in range(qubit_anchor[2],
                                             qubit_anchor[3]+1)]
            oper_l = ['Z' for i in qubit_supp_l]
            oper_r = ['Z' for i in qubit_supp_r]
            pauli_list = ['X','Y']
            for oper_id in [0,1]:
                for group_id in [0,1]:
                    for pos_id in [0,1]:
                        oper_list_temp = [oper_l,oper_r]
                        oper_list_temp[0][0] = pauli_list[1-oper_id]
                        oper_list_temp[0][-1] = pauli_list[1-oper_id]
                        oper_list_temp[1][0] = pauli_list[1-oper_id]
                        oper_list_temp[1][-1] = pauli_list[1-oper_id]
                        oper_list_temp[group_id][-pos_id] = pauli_list[
                                                            oper_id]
                        gate_sequence.append(Pauli_rotation(
                            qubit_supp_l + qubit_supp_r,
                            oper_list_temp[0] + oper_list_temp[1],
                            param_id,
                            ((-1)**(oper_id+group_id+1))/4
                        ).gateSequence)
        ret_gate_sequence = []
        for row in gate_sequence:
            for gate in row:
                ret_gate_sequence.append(gate)
        return ret_gate_sequence



    def circ_d_down(self) -> List[SingleGate]:
        param_id_list = self.param_id_list_dict['d_down']
        qubit_anchor_list = [i for i in combinations(
                            [j for j in range(self.num_orbits,
                                              2*self.num_orbits)], 4)] 
        gate_sequence = [] 
        for param_id, qubit_anchor in zip(param_id_list,qubit_anchor_list):
            qubit_supp_l = [i for i in range(qubit_anchor[0],
                                             qubit_anchor[1]+1)]
            qubit_supp_r = [i for i in range(qubit_anchor[2],
                                             qubit_anchor[3]+1)]
            oper_l = ['Z' for i in qubit_supp_l]
            oper_r = ['Z' for i in qubit_supp_r]
            pauli_list = ['X','Y']
            for oper_id in [0,1]:
                for group_id in [0,1]:
                    for pos_id in [0,1]:
                        oper_list_temp = [oper_l,oper_r]
                        oper_list_temp[0][0] = pauli_list[1-oper_id]
                        oper_list_temp[0][-1] = pauli_list[1-oper_id]
                        oper_list_temp[1][0] = pauli_list[1-oper_id]
                        oper_list_temp[1][-1] = pauli_list[1-oper_id]
                        oper_list_temp[group_id][-pos_id] = pauli_list[
                                                            oper_id]
                        gate_sequence.append(Pauli_rotation(
                            qubit_supp_l + qubit_supp_r,
                            oper_list_temp[0] + oper_list_temp[1],
                            param_id,
                            ((-1)**(oper_id+group_id+1))/4
                        ).gateSequence)
        ret_gate_sequence = []
        for row in gate_sequence:
            for gate in row:
                ret_gate_sequence.append(gate)
        return ret_gate_sequence

    
    def circ_d_mix(self) -> List[SingleGate]:
        param_id_list = self.param_id_list_dict['d_mix']
        qubit_anchor_list_l = [i for i in combinations(
                               [j for j in range(0,self.num_orbits)],2)]
        qubit_anchor_list_r = [i for i in combinations(
                               [j for j in range(self.num_orbits,
                                                 2*self.num_orbits)],2)]
        qubit_anchor_list = []
        for item_l in qubit_anchor_list_l:
            for item_r in qubit_anchor_list_r:
                qubit_anchor_list.append(item_l+item_r)
        gate_sequence = []
        for param_id, qubit_anchor in zip(param_id_list,qubit_anchor_list):
            qubit_supp_l = [i for i in range(qubit_anchor[0],
                                             qubit_anchor[1]+1)]
            qubit_supp_r = [i for i in range(qubit_anchor[2],
                                             qubit_anchor[3]+1)] 
            oper_l = ['Z' for i in qubit_supp_l]
            oper_r = ['Z' for i in qubit_supp_r]
            pauli_list = ['X','Y']
            for oper_id in [0,1]:
                for group_id in [0,1]:
                    for pos_id in [0,1]:
                        oper_list_temp = [oper_l,oper_r]
                        oper_list_temp[0][0] = pauli_list[1-oper_id]
                        oper_list_temp[0][-1] = pauli_list[1-oper_id]
                        oper_list_temp[1][0] = pauli_list[1-oper_id]
                        oper_list_temp[1][-1] = pauli_list[1-oper_id]
                        oper_list_temp[group_id][-pos_id] = pauli_list[
                                                            oper_id]
                        gate_sequence.append(Pauli_rotation(
                            qubit_supp_l + qubit_supp_r,
                            oper_list_temp[0] + oper_list_temp[1],
                            param_id,
                            ((-1)**(oper_id+pos_id))/4
                        ).gateSequence)
        ret_gate_sequence = []
        for row in gate_sequence:
            for gate in row:
                ret_gate_sequence.append(gate)
        return ret_gate_sequence

        

    def circ_s_up(self) -> List[SingleGate]:
        param_id_list = self.param_id_list_dict['s_up']
        qubit_anchor_list = [i for i in combinations(
                             [j for j in range(0,self.num_orbits)],2)]
        gate_sequence = []
        for param_id, qubit_anchor in zip(param_id_list,qubit_anchor_list):
            qubit_supp = [i for i in range(qubit_anchor[0],
                                           qubit_anchor[1]+1)]
            oper = ['Z' for i in qubit_supp]
            for pos in [0,1]:
                oper_list_temp = oper
                oper_list_temp[0] = 'X'
                oper_list_temp[-1] = 'X'
                oper_list_temp[-pos] = 'Y'
                gate_sequence.append(Pauli_rotation(
                    qubit_supp,
                    oper_list_temp,
                    param_id,
                    ((-1)**(pos+1))
                ).gateSequence)
        ret_gate_sequence = []
        for row in gate_sequence:
            for gate in row:
                ret_gate_sequence.append(gate)
        return ret_gate_sequence
    




    def circ_s_down(self) -> List[SingleGate]:
        param_id_list = self.param_id_list_dict['s_down']
        qubit_anchor_list = [i for i in combinations(
                             [j for j in range(self.num_orbits,
                                               2*self.num_orbits)],2)]
        gate_sequence = []
        for param_id, qubit_anchor in zip(param_id_list,qubit_anchor_list):
            qubit_supp = [i for i in range(qubit_anchor[0],
                                           qubit_anchor[1]+1)]
            oper = ['Z' for i in qubit_supp]
            for pos in [0,1]:
                oper_list_temp = oper
                oper_list_temp[0] = 'X'
                oper_list_temp[-1] = 'X'
                oper_list_temp[-pos] = 'Y'
                gate_sequence.append(Pauli_rotation(
                    qubit_supp,
                    oper_list_temp,
                    param_id,
                    ((-1)**(pos+1))
                ).gateSequence)
        ret_gate_sequence = []
        for row in gate_sequence:
            for gate in row:
                ret_gate_sequence.append(gate)
        return ret_gate_sequence
    

    def _calc_gate_sequences(self) -> List[SingleGate]:
        gate_sequences = []
        gate_sequences = gate_sequences + self.circ_d_up()
        gate_sequences = gate_sequences + self.circ_d_down()
        gate_sequences = gate_sequences + self.circ_d_mix()
        gate_sequences = gate_sequences + self.circ_s_up()
        gate_sequences = gate_sequences + self.circ_s_down()
        return gate_sequences
    
