# Quantum-algorithms
All works joint with J. Gardiner and J. Kazdan. 

These are some simple implementations of four elementary quantum algorithms -- Deutsch-Jozsa, Bernstein-Vazirani, Grover's, and Simon's -- 
written in Python using Google's Cirq library for the purposes of CS 238 in the Winter quarter of 2021 at UCLA. 

Additionally, we give an implementation of Shor's algorithm in Cirq using 2n+3 qubits where n = ceil(log_2(N)), as first described by Beauregard. 
There are two versions, both of which implement semiclassical controls of the quantum circuit. In version 0.14.0 of Cirq, semiclassical
controls are built-in, while earlier versions require a technical workaround, which we implemented. 

Reports detailing development and run time efficiency are included. 
