import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import functools as ft
import random
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister,transpile,assemble
from qiskit.quantum_info import random_unitary, partial_trace, Statevector
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit.library import UnitaryGate
from qiskit_algorithms import AmplificationProblem,Grover
from qiskit.primitives import Sampler
from qiskit.visualization import array_to_latex
from qiskit_aer import Aer
#from fable import fable
from fable.fable import fable
sim = Aer.get_backend('aer_simulator',device='GPU')
from functions_QQ import *
N = 4
d = 6
h = 1.0
jx = 0.5
jy = 1.25
jz = 2.0
H_heisenberg = transverse_field_heisenberg(N, h, jx, jy, jz)
energy_vector = []
for beta in np.arange(0.0, 1.6, 0.05):
    exp_value = 0.0
    for i in range(100):
        Q = expm(-beta*H_heisenberg/2.0)
        entropy, circ, state = quantum_haar_state(N, d)
        tpq  = np.matmul(Q, state)
        exp_value += (np.inner(tpq.conj().T, np.matmul(H_heisenberg, tpq))/np.inner(tpq.conj().T, tpq)).real
    exp_value /= 100
    energy_vector.append(exp_value)
Ener=[]

for beta in np.arange(0.0, 1.6, 0.05):
    exp_value = 0.0
    for i in range(100):
        Q = expm(-beta*H_heisenberg/2.0)
        estate = block_encoding_amplification(N,d,H_heisenberg,beta)['statevector']
        tpq  = np.matmul(Q, state.data[0:2**N])
        exp_value += (np.inner(tpq.conj().T, np.matmul(H_heisenberg, tpq))/np.inner(tpq.conj().T, tpq)).real
    exp_value /= 100
    Ener.append(exp_value)
fig, ax = plt.subplots()
ax.scatter(np.arange(0.0, 1.6, 0.05), energy_vector, color='blue')
ax.plot(np.arange(0.0, 1.6, 0.05), energy_vector, label='Classical method', color='blue')
#ax.scatter(np.linspace(0.0, 1.6, 32), np.array(Ener), color='red')
#ax.plot(np.linspace(0.0, 1.6, 32), np.array(Ener), label='Quantum method', color='red')
ax.scatter(np.arange(0.0, 1.6, 0.05), Ener, color='red')
ax.plot(np.arange(0.0, 1.6, 0.05), Ener, label='Quantum method', color='red')
# Add labels and legend
ax.set_xlabel('Inverse Temperature')
ax.set_ylabel('Energy')
ax.legend()
plt.grid()
plt.show()
