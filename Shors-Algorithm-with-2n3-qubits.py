#!/usr/bin/env python
# coding: utf-8

# In[20]:


import cirq
import numpy as np
import sympy
from fractions import Fraction
from math import gcd, ceil, log2, floor
from typing import Callable, Optional, Sequence, Union, List
import time
from cirq import ops


# In[2]:


def toffoli(ctrl1, ctrl2, target) -> List[cirq.Operation]:
    # Makes a Toffoli gate using one of the standard decompositions that uses controlled-R_x gates (up to global phase). 
    
    return [(cirq.CX**(0.5))(ctrl2, target),cirq.CX(ctrl1, ctrl2),(cirq.CX**(-0.5))(ctrl2, target),
        cirq.CX(ctrl1, ctrl2), (cirq.CX**(0.5))(ctrl1, target)]


# In[3]:


def phaseAdder(a: int, n: int, qubits: List[cirq.Qid], ctrl1 = None, ctrl2 = None) -> List[cirq.Operation]:
    # We add a classical int a in the Fourier basis, assuming a is less than 2**n. 
    # a is considered as QFT of n+1 bit integer state because register size must be n+1 to hold overflow. 
    # Convention is qubit[i] is ith most significant qubit (with 0-based indexing), so qubits[0] handles overflow. 
    # The controls ctrl1, ctrl2 are optional and are used in this version of Shor's algorithm. 
    
    # Returns list of n+1 operations. 
    gates = []
    
    assert len(qubits) == n+1, "Need n+1 qubits to handle overflow of a+b."
    
    # Determine the phase for the Z**(2*phase) gate on qubits[i]. After application, will hold bits 0 through n-i of
    # a + b in the phase. I.e. we assume the QFT used to get to Fourier basis does not flip the qubits. 
    for i in range(n+1):
        phase = (a/(2**(n+1-i)))%1
        # Note: cirq.RZ(theta) creates global phase in general, while cirq.Z**(theta) does not. 
        # cirq.Z**(theta) returns gate with unitary representation [1 0, 0 e^(pi*i*theta)].
        if (ctrl1 == None) and (ctrl2 == None):
            gates.append((cirq.Z**(2*phase))(qubits[i]))
        else:
            if ctrl2 == None:
                # Use two CX's to create singly-controlled Rz(2*phase)
                gates.append((cirq.Z**(phase))(qubits[i]))
                gates.append(cirq.CX(ctrl1, qubits[i]))
                gates.append((cirq.Z**(-phase))(qubits[i]))
                gates.append(cirq.CX(ctrl1, qubits[i]))
            else:
                # Use 2 Toffoli's to create doubly-controlled Rz(2*phase)
                gates.append((cirq.Z**(phase))(qubits[i]))
                gates += toffoli(ctrl1, ctrl2, qubits[i])
                gates.append((cirq.Z**(-phase))(qubits[i]))
                gates += toffoli(ctrl1, ctrl2, qubits[i])
    
    return gates

def phaseSubtractor(a, n, qubits, ctrl1 = None, ctrl2 = None) -> List[cirq.Operation]:
    temp = phaseAdder(a, n, qubits, ctrl1, ctrl2)
    temp.reverse()
    return list(map(cirq.inverse, temp))


# In[4]:


def modPhaseAdder(a, N, n, qubits: List[cirq.Qid], ancilla: cirq.Qid, ctrl1 = None, ctrl2 = None, p = None) -> List[cirq.Operation]:
    # We add classical int a to quantum int b mod N, assuming a, b, and N are less than 2**n. This is done 
    # in the Fourier basis, just as in phaseAdder(). 
    # 'qubits' should have n+1 qubits to handle overflow of a+b. 
    # Convention is qubit[i] is ith most significant qubit (with 0-based indexing), so qubits[0] handles overflow. 
    # 'ancilla' is a zeroed ancilla qubit. 
    
    assert (a < N < 2**n), "Need a < N < 2**n."
    
    gates = []
    gates += phaseAdder(a, n, qubits, ctrl1, ctrl2)
    gates += phaseSubtractor(N, n, qubits)
    gates.append(cirq.qft(*qubits, without_reverse = True)**-1)
    gates.append(cirq.CX(qubits[0], ancilla))
    gates.append(cirq.qft(*qubits, without_reverse = True))
    gates += phaseAdder(N, n, qubits, ancilla)
    gates += phaseSubtractor(a, n, qubits, ctrl1, ctrl2)
    gates.append(cirq.qft(*qubits, without_reverse = True)**-1)
    gates.append(cirq.X(qubits[0]))
    gates.append(cirq.CX(qubits[0], ancilla))
    gates.append(cirq.X(qubits[0]))
    gates.append(cirq.qft(*qubits, without_reverse = True))
    gates += phaseAdder(a, n, qubits, ctrl1, ctrl2)
    
    return gates 


# In[5]:


def modMultiplier(a, N, n, qubits1: List[cirq.Qid], qubits2: List[cirq.Qid], ancilla, ctrl) -> List[cirq.Operation]:
    # Defines a controlled multiplier mod N gate which maps integers x, b mod N to b + a*x mod N. 
    # The integers a, b, x, and N are all assumed to be less than 2**n. x is stored in qubits1 of length n, 
    # and b in qubits2 of length n+1. 'ancilla' is the ancilla for modPhaseAdder.
    
    assert a < N < 2**n, 'Need a < N < 2**n.'
    
    gates = []
    # Puts b into Fourier basis.
    gates.append(cirq.qft(*qubits2, without_reverse = True))
    for i in range(n):
        # Adds (2^i)*a*(x_i) to b mod N where x = x_{n-1}...x_0. 
        gates += modPhaseAdder((a*(2**(n-i-1)))%N, N, n, qubits2, ancilla, ctrl, qubits1[i], i)
    gates.append(cirq.qft(*qubits2, without_reverse = True)**-1)
    
    return gates


# In[6]:


def controlledUa(a, N, n, qubits1, qubits2, ancilla, ctrl) -> List[cirq.Operation]:
    # Multiplies the int stored in qubits1 by 'a' mod N if ctrl is on by using ancilla register qubits2 of length n+1. 
    # Here, qubits1 is length n, and qubits2[0] is always in the 0 state after each gate in this method. 
    # This requires a to be a unit mod N as otherwise this operation is not invertible. 
    
    ainv = pow(a, -1, N)
    
    gates = []
    gates += modMultiplier(a, N, n, qubits1, qubits2, ancilla, ctrl)
    # Apply a controlled swap between qubits1 and qubits2[1:]
    for i in range(n):
        gates.append(cirq.CX(qubits2[i+1], qubits1[i]))
        gates.append(toffoli(ctrl, qubits1[i], qubits2[i+1]))
        gates.append(cirq.CX(qubits2[i+1], qubits1[i]))
    temp = modMultiplier(ainv, N, n, qubits1, qubits2, ancilla, ctrl)
    temp.reverse()
    gates += list(map(cirq.inverse, temp))
    
    return gates


# In[54]:


# Test cell: Semiclassical inverse QFT + phase estimation, ad hoc implementation of Shor's. 
a, N, n = 7, 15, 4

ctrl = cirq.NamedQubit('ctrl')
reg1 = cirq.LineQubit.range(n)
reg2 = cirq.LineQubit.range(n,2*n+1)
ancilla = cirq.NamedQubit('ancilla')
measurements = '' # Will be a bit string. Used to determine rotation and presence of X gate. 
state = (1<<(n+2)) # Initial state of the combined ctrl + registers + ancilla. Want 1 in reg1, 0 elsewhere. 

for i in range(2*n):
    circuit = cirq.Circuit()
    # Need to zero the ctrl if last measured 1. 
    if (not len(measurements) == 0) and measurements[0] == '1':
        circuit.append(cirq.X(ctrl))
    circuit.append(cirq.H(ctrl))
    circuit.append(controlledUa(pow(a,2**(2*n-i-1), N), N, n, reg1, reg2, ancilla, ctrl))
    # Determine inverse QFT component based off measurements. 
    # Want to apply [1 0, 0 e**(-2*pi*i*(0.m_{i-1}...m_0))]
    if i == 0:
        phase = 0
    else:
        phase = (int(measurements, 2)/(2**i))%1
    circuit.append((cirq.Z**(-2*phase))(ctrl))
    circuit.append(cirq.H(ctrl))
    circuit.append(cirq.measure(ctrl, key = f'bit{i}'))
    # simulate() produces measurements as well as final state vector after measurement. 
    # We take the measurement to determine next circuit, and we take the final state vector to
    # feed it into next circuit. 
    sim = cirq.Simulator()
    result = sim.simulate(program = circuit, qubit_order = [ctrl, *reg1, *reg2, ancilla], initial_state = state)
    measurements = str(result.measurements[f"bit{i}"][0]) + measurements
    state = result.state_vector()
    print(result.measurements[f"bit{i}"])
print(measurements)
    


# In[43]:


# Junk cell to test out classical controls

a, N, n = 7, 15, 4
ctrl = cirq.NamedQubit('ctrl')
reg1 = cirq.LineQubit.range(n)
reg2 = cirq.LineQubit.range(n,2*n+1)
ancilla = cirq.NamedQubit('ancilla')
string = ''
print(cirq.__version__)
circuit = cirq.Circuit()
circuit.append(cirq.measure(ctrl, key = 'a'))
circuit.append(cirq.ClassicallyControlledOperation(cirq.X(ctrl), ['a']))
circuit.append(cirq.X(ctrl).with_classical_controls(''))
print(circuit)
print(circuit.all_qubits())
# sim = cirq.Simulator()
# print(sim.simulate(circuit).state_vector())
print(-1**3)
print((7**)


# In[112]:


# Test cell: Classically controlled version of Shor using pre-release Cirq v0.14. 
a, N, n = 7, 15, 4

ctrl = cirq.NamedQubit('ctrl')
reg1 = cirq.LineQubit.range(n)
reg2 = cirq.LineQubit.range(n,2*n+1)
ancilla = cirq.NamedQubit('ancilla')

circuit = cirq.Circuit()
# Initialize state of reg1 to |1>.
circuit.append(cirq.X(reg1[n-1]))
# Start with gates before first measurement.
circuit.append(cirq.H(ctrl))
circuit.append(controlledUa(pow(a, 2**(2*n-1), N), N, n, reg1, reg2, ancilla, ctrl))
circuit.append(cirq.H(ctrl))
circuit.append(cirq.measure(ctrl, key = 'bit0'))

# Apply the rest of the gates. 
for i in range(1,2*n):
    circuit.append(cirq.X(ctrl).with_classical_controls(f'bit{i-1}'))
    circuit.append(cirq.H(ctrl))
    circuit.append(controlledUa(pow(a, 2**(2*n-i-1), N), N, n, reg1, reg2, ancilla, ctrl))
    for j in range(i): 
        # Apply phase gate conditioned upon past measurements. 
        # Want to apply [1 0, 0 e**(-2*pi*i*(0.m_{i-1}...m_0))]
        circuit.append((cirq.Z**(-(2**(i-j))))(ctrl).with_classical_controls(f'bit{j}'))
    circuit.append(cirq.H(ctrl))
    circuit.append(cirq.measure(ctrl, key = f'bit{i}'))

sim = cirq.Simulator()
res = sim.simulate(circuit)
phase = 0
for i in range(2*n):
    phase += int(res.measurements[f'bit{i}'][0])/(2**(2*n-i))
print(phase)


# In[113]:


for i in range(2*n):
    print(int(res.measurements[f'bit{i}']))
    
# tq = cirq.NamedQubit('test')
# tc = cirq.Circuit(cirq.H(tq))
# tc.append(cirq.measure(tq, key = 't'))
# tc.append(cirq.H(tq).with_classical_controls('t'))
# simu = cirq.Simulator()
# resu = simu.simulate(tc)
# print(resu.measurements['t'])
# print(resu.state_vector())
# print(tc)


# In[51]:


# Junk cell to test speed of controlledUa on 4n+ qubits
qubits = cirq.LineQubit.range(4) # up to N=255
qubits2 = cirq.LineQubit.range(4,9) # starts as |0> and is needed to implement modular multiplication
ctrl = cirq.NamedQubit('ctrl') # Will always be on in this cell
ancilla = cirq.NamedQubit('ancilla') # Needed for uncomputation in modular addition
x = 1 # Input ket |x>
a = 7 # a for Ua multiplication
N = 15 # 
n = ceil(log2(N))
n = 4
xbin = format(x, f'0{n}b')
for i in range(n): # Set input ket |x>
    if xbin[i] == '1':
        cir.append(cirq.X(qubits[i]))
# time1 = time.time()
b = 0
f = 12
temp = controlledUa(a, N, n, qubits, qubits2, ancilla, ctrl)
totes = 0
for _ in range(f):
    cir = cirq.Circuit()
    
    cir.append(cirq.X(ctrl)) # Set ctrl on so controlledUa does something
    
    for i in range(n): # Set target ket |b>
        if format(b, f'0{n}b')[i] == '1':
            cir.append(cirq.X(qubits2[i+1]))
            
    cir.append(temp)
    print(f"Number of gates in arithmetic gate: {len(temp)}.")
#     time2 = time.time()
    cir.append(cirq.measure(*qubits, key = 'result'))
    cir.append(cirq.measure(*qubits2, key = 'reg2'))
    cir.append(cirq.measure(ctrl, key = 'ctrl'))
    cir.append(cirq.measure(ancilla, key = 'ancilla'))
    sim = cirq.Simulator()
    time3 = time.time()
    res = sim.simulate(cir)
    time4 = time.time()
    totes += time4-time3
print(totes)
# print(f"Time to build circuit: {time2-time1}")
# print(f"Time to run circuit: {time4-time3}")
# print(f"Expect {format(x, f'0{n}b')} in the first register.")
# print(f"Expect {(b + f*a*x)%N} in the second register: {format(int((b + f*a*x)% N), f'0{n+1}b')}")
# print(f"First register: {res.measurements['result'][0]}")
# print(f"Second register: {res.measurements['reg2'][0]}")
# print(f"Ancilla sanity check should be 0: {res.measurements['ancilla'][0]}")
# print(f"Ctrl sanity check should be 1: {res.measurements['ctrl'][0]}")


# In[60]:


def power_check(N):
    L = int(ceil(log2(N)))
    for b in range(2, L + 1):
        if int(floor(2**(log2(N)/b)))**b == N:
            return True, int(floor(2**(log2(N)/b))), b
        if int(ceil(2**(log2(N)/b)))**b == N:
            return True, int(ceil(2**(log2(N)/b))), b
    return False, 0, 0


# In[64]:


def Shor(N: int) -> int:
    # For an integer N, returns a non-trivial factor of N if N is composite, returns 0 if N is prime
    # 2 if N is even
    
    if N < 0:
        N = -N
    # Handle dumb cases. 
    if N == 1:
        return 1
    if not N%2:
        return 2
    if sympy.isprime(N):
        return 0
    nontrivial_power, factor, b = power_check(N)
    if nontrivial_power:
        return factor
    
    # Begin the search. 
    while True:
        a = np.random.randint(2,N-1)
        if gcd(a, N) != 1:
            return gcd(a, N)

        n = ceil(log2(N))
        # Make 2n+3 qubits. 
        reg1 = cirq.LineQubit.range(n)
        reg2 = cirq.LineQubit.range(n,2*n+1)
        ancilla = cirq.NamedQubit('ancilla')
        ctrl = cirq.NamedQubit('ctrl')

        # Implement semi-classical QFT^-1 and phase estimation. 
        measurements = '' # Will be a bit string. Used to determine rotation and presence of X gate. 
        state = (1<<(n+2)) # Initial state of the combined ctrl + registers + ancilla. Want 1 in reg1, 0 elsewhere. 

        for i in range(2*n):
            circuit = cirq.Circuit()
            # Need to zero the ctrl if last measured 1. 
            if (not len(measurements) == 0) and measurements[0] == '1':
                circuit.append(cirq.X(ctrl))
            circuit.append(cirq.H(ctrl))
            circuit.append(controlledUa(pow(a, 2**(2*n-i-1), N), N, n, reg1, reg2, ancilla, ctrl))
            # Determine inverse QFT component based off measurements. 
            # Want to apply [1 0, 0 e**(-2*pi*i*(0.m_i...m_0))]
            if len(measurements) == 0:
                phase = 0
            else:
                phase = (int(measurements, 2)/(2**i))%1
            circuit.append((cirq.Z**(-2*phase))(ctrl))
            circuit.append(cirq.H(ctrl))
            circuit.append(cirq.measure(ctrl, key = f'bit{i}'))
            # simulate() produces measurements as well as final state vector after measurement. 
            # We take the measurement to determine next circuit, and we take the final state vector to
            # feed it into next circuit. 
            sim = cirq.Simulator()
            result = sim.simulate(program = circuit, qubit_order = [ctrl, *reg1, *reg2, ancilla], initial_state = state)
            measurements = str(result.measurements[f"bit{i}"][0]) + measurements
            state = result.state_vector()

        # measurements bitstring represents 0.m_{2n-1}...m_0 ~ s/r. 
        # Convert it to a float and then fraction with denominator at most N. 
        frac = Fraction(int(measurements, 2)/(2**(2*n))).limit_denominator(N)
        r = frac.denominator # A divisor of order of a (mod N)
        if r % 2 == 0:
            div1, div2 = gcd(a**(r//2) + 1, N), gcd(a**(r//2) - 1, N)
            if N > div1 > 1:
                return div1
            if N > div2 > 1:
                return div2


# In[68]:


# Shor(15) # Seems to work. Unsure of Fourier transform indexing


# In[114]:


# Version of Shor's using classically conditioned gates in Cirq v0.14.0.
def Shor2(N: int) -> int:
    # For an integer N, returns a non-trivial factor of N if N is composite, returns 0 if N is prime
    # 2 if N is even
    
    if N < 0:
        N = -N
    # Handle dumb cases. 
    if N == 1:
        return 1
    if not N%2:
        return 2
    if sympy.isprime(N):
        return 0
    nontrivial_power, factor, b = power_check(N)
    if nontrivial_power:
        return factor
    
    # Begin the search. 
    while True:
        a = np.random.randint(2,N-1)
        if gcd(a, N) != 1:
            return gcd(a, N)

        n = ceil(log2(N))
        # Make 2n+3 qubits. 
        reg1 = cirq.LineQubit.range(n)
        reg2 = cirq.LineQubit.range(n,2*n+1)
        ancilla = cirq.NamedQubit('ancilla')
        ctrl = cirq.NamedQubit('ctrl')
        
        circuit = cirq.Circuit()
        # Initialize state of reg1 to |1>.
        circuit.append(cirq.X(reg1[n-1]))
        # Start with gates before first measurement.
        circuit.append(cirq.H(ctrl))
        circuit.append(controlledUa(pow(a, 2**(2*n-1), N), N, n, reg1, reg2, ancilla, ctrl))
        circuit.append(cirq.H(ctrl))
        circuit.append(cirq.measure(ctrl, key = 'bit0'))

        # Apply the rest of the gates. 
        for i in range(1,2*n):
            circuit.append(cirq.X(ctrl).with_classical_controls(f'bit{i-1}'))
            circuit.append(cirq.H(ctrl))
            circuit.append(controlledUa(pow(a, 2**(2*n-i-1), N), N, n, reg1, reg2, ancilla, ctrl))
            for j in range(i): 
                # Apply phase gate conditioned upon past measurements. 
                # Want to apply [1 0, 0 e**(-2*pi*i*(0.m_{i-1}...m_0))]
                circuit.append((cirq.Z**(-(2**(i-j))))(ctrl).with_classical_controls(f'bit{j}'))
            circuit.append(cirq.H(ctrl))
            circuit.append(cirq.measure(ctrl, key = f'bit{i}'))

        sim = cirq.Simulator()
        res = sim.simulate(circuit)
        phase = 0
        for i in range(2*n):
            phase += int(res.measurements[f'bit{i}'][0])/(2**(2*n-i))
        
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator # A divisor of order of a (mod N)
        if r % 2 == 0:
            div1, div2 = gcd(a**(r//2) + 1, N), gcd(a**(r//2) - 1, N)
            if N > div1 > 1:
                return div1
            if N > div2 > 1:
                return div2
        


# In[119]:


# Shor2(15) # Seems to work faster. Also ports to QASM just fine. 

