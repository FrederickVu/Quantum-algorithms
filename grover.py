from dj import *
from shared_oracle_code import *
def toffoli(ctrl1, ctrl2, target, Rxgates = True, **kwargs):
    # Makes a Toffoli gate using one of the standard decompositions that use either controlled-R_x gates or T gates. 
    
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

def toffolify(controls: list[cirq.Qid], ancillae: list[cirq.Qid], target):
    # toffolify makes a C^kX gate given at least k-2 borrowed ancillae using Toffoli gates. 
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
        
        
def CrX_maker(controls: list[cirq.Qid], target, ancilla = None, borrowed = True):
    
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

def oracle_DJ(f: Callable[[int], int], n, qubits):
    # Contructs the oracle for DJ algorithm using two ancillae qubits, b and d. The idea is that each time f(x)=1, 
    # we must transpose the two basis states |x,b> and |x,b+1>. This can be done with a single C^nX targeting b,
    # with X's to change from control to anti-controls when needed. The ancilla d is to help us decompose the C^nX. 
    
    assert len(qubits) > n, "Needs more ancillae."
    
    # We add an ancilla qubit to help in decomposition. 
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
                if format(x, f'0{n}b')[j] == '0':
                    yield cirq.X(qubits[j])
            yield CrX_maker(qubits[:n], qubits[n], qubits[n+1], False)
            for j in range(n):
                if format(x, f'0{n}b')[j] == '0':
                    yield cirq.X(qubits[j])

def expand(generator) -> List:
        toReturn = []
        for x in generator:
            if not isinstance(x, types.GeneratorType):
                toReturn.append(x)
            else:
                toReturn += expand(x)
        return toReturn

def Grover(f: Callable[[int], int] , n, a = 1) -> Union[int, List[int]]:
    
    # f is a bit-valued function which evaluates to 1 on exactly 'a' distinct inputs.  
    # If 'a' is not specified, it is assumed to be 1. 
    # This method tries to return the full list of the 'a' inputs. 
    
    if a == 0:
        print("You fool.")
        return []
    
    # We use n+2 qubits, where the first n are the usual computation qubits, then two ancillae: b and d. 
    # b is the usual ancilla for the quantum oracle. d is to aid in decomposition of the C^nX's and C^(n-1)X's that
    # appear in the rotation operator. 
    qubits = cirq.LineQubit.range(n+2)
    
    # We construct the oracle for f once and store it as a list of gates. 
    oracle = expand(oracle_DJ(f, n, qubits))
    
    # List of ints to return. 
    winners = []
    
    while a - len(winners) > 0:
        # Find number of times to apply Grover's diffusion/rotation operator. 
        reps = round((math.pi/(2*np.arcsin(math.sqrt((a-len(winners))/2**n)))-1)/2)\
        
        # Define indicator function for already found targets
        def g(num):
            return 1 if (num in winners) else 0
        
        # Create a new circuit to find remaining search targets. 
        circuit = cirq.Circuit()
        circuit.append(cirq.X(qubits[n])) # Set ancilla b to |1>
        circuit.append(cirq.H(q) for q in qubits[:n+1])
        
        # Apply rotation operator, accounting for already discovered search items
        for _ in range(reps): 
            circuit.append(oracle)
            
            # Cancel out the permutations corresponding to found targets in the original oracle above. 
            circuit.append(oracle_DJ(g, n, qubits))
            
            circuit.append(cirq.H(q) for q in qubits[:n])
            circuit.append(cirq.X(q) for q in qubits[:n])
            
            # Make a C^(n-1)Z using a C^(n-1)X and Hadamards targeting qubit with index n-1.
            circuit.append(cirq.H(qubits[n-1])) 
            circuit.append(CrX_maker(qubits[:n-1], qubits[n-1], qubits[n+1], False))
            circuit.append(cirq.H(qubits[n-1]))
            
            circuit.append(cirq.X(q) for q in qubits[:n])
            circuit.append(cirq.H(q) for q in qubits[:n])
        
        circuit.append(cirq.measure(*qubits[:n], key = 'result'))
        
        sim = cirq.Simulator()
        run = sim.run(circuit, repetitions = 1000*(a-len(winners)))
        results = run.histogram(key = 'result')

        # most_common(i) returns sublist of tuples (results, count) of the i results with highest counts. 
        candidates = [results.most_common(a-len(winners))[i][0] for i in range(a-len(winners))]
        
        winners += [x for x in candidates if (f(x) == 1 and x not in winners)]
        if len(winners) == a:
            if a == 1:
                return winners[0]
            return winners

def f_maker(sample: list[int]):
    def f(num):
      return 1 if num in sample else 0
    return f

def test_Grover(n):
    times = []
    for i in range(2,n):
        times.append([])
        total = 0
        a_values = [1]
        for a in a_values:
            functions = [f_maker(random.sample(range(0,2**i), a)) for _ in range(4)]
            for f in functions:
                x = time.time()
                Grover(f,i,a)
                y = time.time()
                total += y-x
            times[i-2].append(total/len(functions))
    return times
print(test_Grover(4))