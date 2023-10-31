from shared_oracle_code import *
from dj import *
#returns a list specifying the phases for the 2**n phase gadgets making up the oracle circuit for the function f on n bits. More precisely, it gives(-1)^thue_morse(y) * 2**n/pi times the phase for each phase gadget type y.
def to_phases_Simon(f, n):
    shared_phases = np.zeros(2**n)
    nonshared_phases = np.zeros((n, 2**n))
    for x in range(2**n):
        output = f(x)
        for k in range(n):
            if (output>>k)%2 == 1:
                for y in range(2**n):
                    dot =(~x)&y
                    exponent = sum(dot>>i for i in range(n))%2
                    shared_phases[y] += (-1)**(exponent)
                for y in range(2**n,2**(n+1)):
                    dot =(~x)&y
                    exponent = sum(dot>>i for i in range(n))%2
                    nonshared_phases[k][y-2**n] += (-1)**(exponent)
    phase_totals = np.zeros((n+1, 2**n))
    phase_totals[0][:] = shared_phases
    phase_totals[1:][:] = nonshared_phases
    return phase_totals

def oracle_Simon(f, n, qubits):
    phase_totals = to_phases_Simon(f, n)
    oracle_op_list = []
    oracle_op_list.append(cirq.H(q) for q in qubits[n:])
    #Shared phases
    for y in range(1,2**n):
        phase_factor = np.pi/2**n * (-1) * (-1)**sum(y>>i for i in range(n+1))
        oracle_op_list = oracle_op_list + phase_gadget(y, phase_factor * phase_totals[0][y], qubits)
    #Nonshared phases
    for k in range(n):
        for y in range(2**n):
            phase_factor = np.pi/2**n * (-1) * (-1)**sum((y+2**(n+k))>>i for i in range(2*n+1))
            oracle_op_list = oracle_op_list + phase_gadget(y+2**(n+k), phase_factor * phase_totals[k+1][y], qubits)
    oracle_op_list.append(cirq.H(q) for q in qubits[n:])
    return oracle_op_list

def Simon(f: Callable[[int], int], n) -> int:
    # f(x) = f(y) iff x+y = 0 or s, where x, y, and s are integers. Goal: find s. 
    
    # This function tries to locate s in the intersection of the kernels of a large number of homomorphisms 
    # that are guaranteed to contain s by Simon's algorithm. The intersection it finds is very likely to be at most
    # 1-dimensional, and if it isn't, then the function is called again. 
    
    # Initialize the circuit object for 2n qubits in register, construct the oracle. 
    qubits = cirq.LineQubit.range(2*n)
    circuit = cirq.Circuit()
    
    # Apply oracle conjugated by Hadamards to entire register. 
    circuit.append(cirq.H(q) for q in qubits[:n])
    circuit.append(oracle_Simon(f, n, qubits))
    circuit.append(cirq.H(q) for q in qubits[:n])

    # Add measurement gates to the computational part of the register. This allows for the creation of a dictionary object 
    # of the form Dict[key: string, value: np.ndarray] with key name 'result' upon simulation/running.
    circuit.append(cirq.measure(*qubits[:n], key = 'result'))
    sim = cirq.Simulator()
    
    # We collect a number N of vectors from the circuit. In general, the probability that N uniformly randomly
    # chosen vectors of GF(2)^n span is given by (1-2^(-N))*(1-2*(-N+1))*...*(1-2^(-N+n-1)). This value is very close
    # to 1 whenever N-n is not small. This is the probability of success of finding s when s=0, and in the case
    # s is not 0, the probability is even higher. For N = 30, the probability of failure is less than .1% for n up to 20.
    store = []
    for _ in range(30):
        res = sim.simulate(circuit)
        store.append(list(res.measurements['result']))

    # We use sympy.nullspace() to find the a basis for the kernel of the linear transformation corresponding to the
    # matrix formed by the 12*n measured outputs. The argument of the nullspace() method allows us to work in GF(2), 
    # or rather, it treats even integers as 0 when performing row reduction to find the nullspace. 
    store = sympy.Matrix(store)
    kernel = store.nullspace(iszerofunc = lambda x: (x % 2 == 0))
    
    # If the kernel is more than 1 dimensional, then we try again.
    # If the kernel is 0 dimensional, so that the basis produced above is empty, then it must be that s = 0. 
    # Otherwise, we have found a candidate for s. 
    if len(kernel) > 1:
        Simon(f,n)
    elif len(kernel) == 0:
        return 0
    else:
        bits = kernel[0].applyfunc(lambda x : x % 2) # We change the integer values in kernel[0] to 0's and 1's. 
        bits = [int(bit) for bit in bits] # We must change from SymPy 'core' objects to ints. 
        candidate_s = sum(j<<i for i, j in enumerate(bits)) # We must change from the list[int] object 'bits' to an int.
        if f(candidate_s) == f(0):
            return candidate_s
        else:
            Simon(f, n)

# Test for simons

def test_simon(n):
    random_outputs = random.sample(range(2**n), 2**n)
    s = random.randint(0,2**n-1)

    def test_func(x):
        if x^s < x:
            return test_func(x^s)
        else:
            return random_outputs[x]
    times = []
    Simonfuncs = [lambda x: test_func(x) for _ in range(5)]
    for i in range(1,n):
        qubits=cirq.LineQubit.range(2*i)
        total = 0
        for function in Simonfuncs:
            circuit = cirq.Circuit()
            a= time.time()
            circuit.append(oracle_Simon(function, i, qubits))
            b = time.time()
            total += b-a
        times.append(total/len(Simonfuncs))
    return times
results2 = test_simon(10)

print(results2)
plt.title('Simon oracle construction vs. n')
plt.plot(range(1,10), results2)
plt.xlabel = 'n'
plt.ylabel = 'run time in ms'
plt.labelsize = 5
plt.figsize=(10, 6)