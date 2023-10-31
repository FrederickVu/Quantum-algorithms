from shared_oracle_code import *
from dj import *

def oracle_BV(f, n, qubits):
    # f(x) := a*x + b for some n-bit string a and bit b.
    
    # The BV oracle can be represented by at most n CNOTs corresponding to the bits of a that are 1, and an X if b = 1. 
    for i in range(n):
        if (f(1<<i) == 1):
            yield cirq.CX(qubits[i], qubits[n])
    if f(0) == 1:
        yield cirq.X(qubits[n])

        
        
def BV(f: Callable[[int], int], n) -> Tuple[List[int], int]:
    
    # f must be a function defined on integers, outputting the integer 0 or 1. 
    # f(x) := a*x + b for some n-bit string a and bit b.
    
    # Initialize the circuit object for n+1 qubits in register, construct the oracle (as generator),
    # and set ancilla qubit to 1. 
    qubits = cirq.LineQubit.range(n+1)
    oracle = oracle_BV(f,n,qubits)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(qubits[0]))
    
    # Apply oracle conjugated by Hadamards to entire register. 
    circuit.append(cirq.H(q) for q in qubits)
    circuit.append(oracle)
    circuit.append(cirq.H(q) for q in qubits)
    
    # Add measurement gates to the computational part of the register. This allows for the creation of a dictionary object 
    # of the form Dict[key: string, value: np.ndarray] with key name 'result' upon simulation/running. 
    circuit.append(cirq.measure(*qubits[1:], key = 'result'))
    
    sim = cirq.Simulator()
    res = sim.simulate(circuit)

    # We create and run a separate circuit to determine the value of the bit b in the description of f. 
    # Explicitly, as f(x) := a*x + b, we can apply f to 0^n, 
    # or rather we can input the state ket(0^n) into the oracle U_f, to determine b. 
    circuitB = cirq.Circuit(oracle_BV(f, n, qubits))
    circuitB.append(cirq.measure(qubits[0], key = 'resultB'))
    resB = sim.simulate(circuitB)
    
    # We change from np.ndarray to list to get rid of 'dtype' in output. 
    return (list(res.measurements['result']), list(resB.measurements['resultB']))

"""Test code for construction of BV oracle and BV run time"""

def test_BV(n):
  functions = [lambda x: random.randint(0,2**n-1)&x + random.randint(0,1) for _ in range(10)]
  times = []
  for i in range(1,n):
    qubits=cirq.LineQubit.range(i+1)
    total = 0
    for function in functions:
      a= time.time()
      BV(function, i)
      b = time.time()
      total += b-a
    times.append(total/len(functions))
  return times

BV_run_times = [0]+test_BV(15)

plt.title('BV run time vs n')
plt.plot(BV_run_times)
plt.xlabel = 'n'
plt.ylabel = 'run time in ms'
plt.labelsize = 5
plt.figsize=(10, 6)

def test_oracle_BV(n):
  functions = [lambda x: random.randint(0,2**n-1)&x + random.randint(0,1) for _ in range(10)]
  times = []
  for i in range(1,n):
    total = 0
    for function in functions:
      qubits = cirq.LineQubit.range(i+1)
      a= time.time()
      
      oracle = oracle_BV(function,i,qubits)
      circuit = cirq.Circuit()
      circuit.append(oracle)
    
      b = time.time()
      total += b-a
    times.append(total/len(functions))
  return times

BV_oracle_construction_times = [0]+test_oracle_BV(100)
print(BV_oracle_construction_times)
plt.title('BV oracle construction time vs n')
plt.plot(BV_oracle_construction_times)
plt.xlabel = 'n'
plt.ylabel = 'run time in ms'
plt.labelsize = 5
plt.figsize=(10, 6)
