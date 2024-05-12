from typing import Union, Sequence, Any, Tuple, List
import warnings
from numpy import array

# from utensils import Is_strictly_sorted


# int | None is only supported on python 3.10



class SingleGate():

    def __init__(self, gateType: str, 
                 supp: List[int], paramId: int | None, 
                 param_factor: int | None) -> None:
        if Is_strictly_sorted(supp) is not True:
            raise ValueError("Invalid definition of operator support.")
        self.gateType = gateType
        self.supp = supp
        self.paramId = paramId
        self.param_factor = param_factor

    



class CircuitBlock():

    def __init__(self) -> None:
        self.gateSequence: list[SingleGate] = []
        pass

    def Add_gate(self, gate: SingleGate) -> None:
        self.gateSequence.append(gate)
    
    def Assign_sequence(self, gateSequence: list[SingleGate]) -> None:
        if self.gateSequence is not None:
            warnings.warn("Altering an existing circuit block!")
        self.gateSequence = gateSequence
    



def Pauli_rotation(supp: List[int], 
                   pauliSeq: List[str], paramId: int | None, 
                   param_factor: int | None) -> CircuitBlock:
    if Is_strictly_sorted(supp) is not True:
        raise ValueError("Invalid definition of operator support.")
    if len(supp) != len(pauliSeq):
        raise ValueError("Operator not compatible with its support.")
    circ_block = CircuitBlock()
    for loc, gate in zip(supp, pauliSeq):
        if gate != 'Z':
            if gate == 'X':
                circ_block.Add_gate(SingleGate(gateType='H', 
                                               supp=[loc],
                                               paramId=None,
                                               param_factor=None))
            elif gate == 'Y':
                circ_block.Add_gate(SingleGate(gateType='Rxh',
                                               supp=[loc],
                                               paramId=None,
                                               param_factor=1))
            else:
                raise ValueError("Ill-defined Pauli operator.")
    for i in range(len(supp)-1):
        circ_block.Add_gate(SingleGate(gateType='cx',
                                       supp=[supp[i],supp[i+1]],
                                       paramId=None,
                                       param_factor=None))
    circ_block.Add_gate(SingleGate(gateType='Rz', supp=[supp[-1]],
                                   paramId=paramId,
                                   param_factor=param_factor))
    for i in range(len(supp)-1):
        circ_block.Add_gate(SingleGate(gateType='cx',
                                       supp=[supp[-2-i],supp[-1-i]],
                                       paramId=None,
                                       param_factor=None))
    for loc, gate in zip(supp, pauliSeq):
        if gate != 'Z':
            if gate == 'X':
                circ_block.Add_gate(SingleGate(gateType='H',
                                               supp=[loc],
                                               paramId=None,
                                               param_factor=None))
            elif gate == 'Y':
                circ_block.Add_gate(SingleGate(gateType='Rxh',
                                               supp=[loc],
                                               paramId=None,
                                               param_factor=-1))
            else:
                raise ValueError("Ill-defined Pauli operator.")
    return circ_block
                    
                


def Is_strictly_sorted(list: List) -> bool:
    if len(list) == 1:
        return True
    else: 
        return all(list[i]<list[i+1] for i in range(len(list)-1))



