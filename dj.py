from shared_oracle_code import *


def oracle_DJ(f: Callable[[int], int], n, qubits, phase_gadget_method = True):
    if phase_gadget_method == False:
        # Contructs the oracle for DJ algorithm using two ancillae qubits, b and d. The idea is that each time f(x)=1, 
        # we must transpose the two basis states |x,b> and |x,b+1>. This can be done with a single C^nX targeting b,
        # with X's to change from control to anti-controls when needed. The ancilla d is to help us decompose the C^nX.
        
        assert len(qubits) > n, "Needs more ancillae."

        if len(qubits) == n+1:
          qubits.append(cirq.LineQubit(n+1))

        # Check whether f is constant, in which case Uf is represented by a scalar, and so no gate is needed.
        constant = True
        test = f(0)
        for i in range(2**n):
            if f(i) != test:
                constant = False
                break
        if constant:
            return

        # Add a C^nX into the circuit each time f(x) = 1.
        for x in range(2**n):
            if f(x) == 1:
                # The two for-loops are to implement anti-controls. 
                for j in range(n):
                    if format(x, f'0{n}b')[n - j - 1] == '0':
                        yield cirq.X(qubits[j])
                yield CrX_maker(qubits[:n], qubits[n], qubits[n+1], False)
                for j in range(n):
                    if format(x, f'0{n}b')[n - j - 1] == '0':
                        yield cirq.X(qubits[j])
                        
    else:
        # Constructs the oracle for the DJ algorithm using the phase gadget method. This requires no ancillae.
        oracle_list = build_oracle_from_phase_gadgets(f, n, qubits)
        for i in oracle_list:
            yield i

def DJ(f: Callable[[int], int], n: int) -> int:
    
    # f must be a function defined on integers, outputting the integer 0 or 1. 
    
    # Construct the oracle, initialize the circuit object for n+1 qubits in register, and set ancilla qubit to 1. 
    qubits = cirq.LineQubit.range(n+1) #  CrX_maker will create additional ancilla qubit
    circuit = cirq.Circuit()
    circuit.append(cirq.X(qubits[n])) # ancilla on different index 0 or n depending on oracle choice
    
    # Apply oracle conjugated by Hadamards to entire register. 
    circuit.append(cirq.H(q) for q in qubits)
    circuit.append(oracle_DJ(f, n, qubits, phase_gadget_method=True))
    circuit.append(cirq.H(q) for q in qubits)
    
    # Add measurement gates to the computational part of the register. This allows for the creation of a dictionary object 
    # of the form Dict[key: string, value: np.ndarray] with key name 'result' upon simulation/running. 
    circuit.append(cirq.measure(*qubits[:n], key = 'result'))
    
    # We simulate the circuit. 
    sim = cirq.Simulator()
    res = sim.simulate(circuit)
    quantum_sim = run_on_IBM(circuit, qubits)
    print('simulated_results:\n', res, '\nreal results:\n', quantum_sim)

    # res has an attribute 'measurements' which is the dictionary object mentioned above. 
    # res.measurements['result'] is a np.ndarray of integers, so to check if output is the 0^n state, we may add entries. 
    if sum(res.measurements['result']) == 0:
        return 0
    else:
        return 1

def test_DJ(n, boolean):
  functions = []
  #test on five randomly created functions
  for i in range(1):
        output_one_list = random.sample(range(2**n), 2**(n-1))
        def test_func(x):
            #if x in output_one_list:
            #    return 1
            #else:
            #    return 0
            return 0
        functions.append(test_func)
  for function in functions:
      DJ(function, n)

#test example
run_times_gadgets = test_DJ(3, True)