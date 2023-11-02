import cirq
import numpy as np
import sympy
import math
from typing import Callable, Union, Tuple, List
from qiskit import QuantumCircuit, execute, Aer
import types
import time
import matplotlib.pyplot as plt
import random
import qiskit
from qiskit import IBMQ
IBMQ.save_account('your IBM key here')
provider = IBMQ.load_account()

def run_on_IBM(circuit, qubits, backend='ibm_quito'):
    
    results = run_cirq_circuit_on_qiskit(circuit, qubits, backend)
    
    # Returns counts
    print("Total counts:", results)
    return results
    
def run_cirq_circuit_on_qiskit(circuit: 'cirq.Circuit', qubits, backend='ibmq_quito'):

    backend = provider.get_backend(backend)
    
    qasm_output = cirq.QasmOutput((circuit.all_operations()), qubits)
    qasm_circuit = QuantumCircuit().from_qasm_str(str(qasm_output))
    
    #IBM Code
    optimized_circuit = qiskit.transpile(qasm_circuit, backend) #5
    optimized_circuit.draw()
    
    # run in real hardware
    job = backend.run(optimized_circuit)
    retrieved_job = backend.retrieve_job(job.job_id())
    result = retrieved_job.result() #6
    
    return result.get_counts()

#converts a function with n inputs into a collection of phase gadgets
def to_phases(f, n):
    phase_totals = np.zeros(2**(n+1))
    for x in range(2**n):
        if f(x) == 1:
            for y in range(2**(n+1)):
                dot =(~x)&y
                exponent = sum(dot>>i for i in range(n))%2
                phase_totals[y] += (-1)**(exponent)
    return phase_totals

#builds a phase gadget with phase alpha on the wires corresponding to the bits of y. Here qubits are a list of all qubits, the last of which should be the "target".
def phase_gadget(y, alpha, qubits):
    m = len(qubits)
    #Make the list, wire_list, of which wires the phase gadget will be on.
    wire_list = []
    for i in range(m):
        if (y>>i)%2 == 1:
            wire_list.append(i)
    #Make a list, op_list, of the operations. This is what this function returns.
    op_list = []
    if alpha == 0:
        return op_list
    else:
        op_list.append((cirq.Z**(alpha/np.pi)).on(qubits[wire_list[0]]))
        for i in range(len(wire_list)-1):
            op_list = [cirq.CNOT(qubits[wire_list[i+1]], qubits[wire_list[i]])] + op_list + [cirq.CNOT(qubits[wire_list[i+1]], qubits[wire_list[i]])]
        return op_list

#builds the oracle out of f, n, and a list of n+1 qubits. Returns a list of cirq operations on the first n entries in qubits
def build_oracle_from_phase_gadgets(f, n, qubits):
    phase_totals = to_phases(f, n)
    oracle_op_list = []
    oracle_op_list.append(cirq.H(qubits[n]))
    for y in range(1,2**(n+1)):
        phase_factor = np.pi/2**n * (-1) * (-1)**sum(y>>i for i in range(n+1))
        oracle_op_list = oracle_op_list + phase_gadget(y, phase_factor * phase_totals[y], qubits)
    oracle_op_list.append(cirq.H(qubits[n]))
    return oracle_op_list

# Makes a Toffoli gate using one of the standard decompositions that use either controlled-R_x gates or T gates. 
def toffoli(ctrl1, ctrl2, target, Rxgates = True):
    if Rxgates:
        yield (cirq.CX**(0.5))(ctrl2, target)
        yield cirq.CX(ctrl1, ctrl2)
        yield (cirq.CX**(-0.5))(ctrl2, target)
        yield cirq.CX(ctrl1, ctrl2)
        yield (cirq.CX**(0.5))(ctrl1, target)
    else:
        yield cirq.H(target)
        yield cirq.CX(ctrl2, target)
        yield (cirq.Z**(-0.25))(target)
        yield cirq.CX(ctrl1, target)
        yield cirq.T(target)
        yield cirq.CX(ctrl2, target)
        yield (cirq.Z**(-0.25))(target)
        yield cirq.CX(ctrl1, target)
        yield cirq.T(ctrl2)
        yield cirq.T(target)
        yield cirq.H(target)
        yield cirq.CX(ctrl1, ctrl2)
        yield cirq.T(ctrl1)
        yield (cirq.Z**(-0.25))(ctrl2)
        yield cirq.CX(ctrl1, ctrl2)

# toffolify makes a C^kX gate given at least k-2 borrowed ancillae using Toffoli gates. 
def toffolify(controls, ancillae, target):
    assert len(controls) - 3 < len(ancillae), "Not enough ancillae."
    
    if len(controls) == 2:
        yield toffoli(controls[0], controls[1], target)
        return
    
    # Each loop will yield 2(k-2) Toffoli gates. 
    for _ in range(2):
        yield toffoli(controls[0], ancillae[0], target)
        for i in range(1, len(controls)-2):
            yield toffoli(controls[i], ancillae[i], ancillae[i-1])
        yield toffoli(controls[len(controls)-2], controls[len(controls)-1], ancillae[len(controls)-3])
        for i in range(len(controls)-3, 0, -1):
            yield toffoli(controls[i], ancillae[i], ancillae[i-1])
        
        
def CrX_maker(controls, target, ancilla = None, borrowed = True):
    
    # The function makes a circuit representing C^rX for r > 1 in terms of smaller general Toffoli gates, i.e. in terms
    # of C^kX for k < r. It relies on the method toffolify() which decomposes these smaller general Toffoli's into
    # genuine Toffoli's, i.e. CCX's. 
    
    # The ancilla may be "borrowed", i.e., its initial state does not have to be 0, and the circuit does not alter it.
    # Set 'borrowed' to False if ancilla is zeroed. If no ancilla is given, the method will create a new one and 
    # assume that its state afterward does not matter, i.e., that it is "burnable". 
    r = len(controls)
    
    if r == 1:
        yield cirq.CX(controls[0], target)
        return
    
    if r == 2:
        yield toffoli(controls[0], controls[1], target)
        return
    
    burnable = False
    if ancilla == None:
        ancilla = cirq.NamedQubit()
        borrowed = False
        burnable = True
    
    half1 = controls[0:r-r//2] # first ceil(r/2) qubits, to be used as controls/ancillae in smaller generalized Toffoli
    half2 = controls[r-r//2:r] # last floor(r/2) qubits, to be used as controls/ancillae in smaller generalized Toffoli
    half2.append(ancilla) # half2 has 0 or 1 more qubit than half1 depending on whether r is odd or even, respectively
    
    # This first decomposition into smaller C^kX's depends on whether ancilla is borrowed or burnable. 
    # This may be further modified, e.g., by using more ancillae to decrease the depth of the decomposition. 
    yield toffolify(half1, half2, ancilla)
    yield toffolify(half2, half1, target)
    if not burnable:
        yield toffolify(half1, half2, ancilla)
        if borrowed:
            yield toffolify(half2, half1, target)
