# Quantum-algorithms
All works joint with J. Gardiner and J. Kazdan for the course CS 238 in the Winter quarter of 2022 at UCLA
written in Python using Google's Cirq library. 

Two included reports detail the methods and testing of the code, though we provide a summary here. 

We have four elementary quantum algorithms -- Deutsch-Jozsa, Bernstein-Vazirani, Grover's, and Simon's -- 
each of which require an 'oracle'. The methods for running and testing these algorithms are in the DJ.py, BV.py, 
Grover.py, and Simon.py files, each of which import from shared_oracle_code.py. The latter file contains, 
as one would expect, methods to help construct the oracles for each algorithm. However, the respective oracles
are quite different in nature, and so there is very little modularity of which to take advantage. The file
also contains code to translate from Cirq to IBM's Qiskit and to run on IBM's quantum computers. 

Additionally, we give an implementation of Shor's algorithm in Cirq using 2n+3 qubits where n = ceil(log_2(N)), 
as first described by Beauregard. There are two versions Shor() and Shor2(), both of which implement semiclassical 
controls of the quantum circuit. In version 0.14.0 of Cirq, semiclassical controls are built-in, while earlier 
versionsrequire a technical workaround, which we implemented. 
