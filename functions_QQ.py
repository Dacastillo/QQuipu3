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
haar_random_gate = UnitaryGate(random_unitary(2))
jx=0.5;jy=1.25;jz=2.0;hx=1.0
s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])
def circ_haar_1d(qc,n1d):
    #CNOT tipo A
    for j in range (n1d): qc.append(haar_random_gate, [j])
    for j in range (n1d-1): qc.cx(j,j+1)
    #CNOT tipo B
    for j in range (n1d): qc.append(haar_random_gate, [j])
    for j in range (n1d-1): qc.cx(j+1,j)
def circ_haar_2d(qc,x2d,y2d):
    #CNOT tipo A
    for j in range (x2d*y2d): qc.append(haar_random_gate, [j])
    for j in range (x2d-1):
        for k in range (y2d-1): qc.cx(k*x2d+j,k*x2d+(j+1))
    #CNOT tipo B
    for j in range (x2d*y2d): qc.append(haar_random_gate, [j])
    for j in range (x2d-1):
        for k in range (y2d-1): qc.cx(k*x2d+(j+1),k*x2d+j)
    #CNOT tipo C
    for j in range (x2d*y2d): qc.append(haar_random_gate, [j])
    for j in range (x2d-1):
        for k in range (y2d-1): qc.cx(k*x2d+j,(k+1)*x2d+j)
    #CNOT tipo D
    for j in range (x2d*y2d): qc.append(haar_random_gate, [j])
    for j in range (x2d-1):
        for k in range (y2d-1): qc.cx((k+1)*x2d+j,k*x2d+j)
def haar_1d(nrep,n1d):
  hchain=[]
  for j in range (nrep+1):
      qc1 = QuantumCircuit(n1d)
      for k in range (j): circ_haar_1d(qc1,n1d)
      qc1.save_statevector()
      state = sim.run(qc1,shots=1).result().get_statevector()
      probs = Statevector(state).probabilities()
      hstate=0;
      for j in range(2**n1d): hstate+=-(probs[j]*np.log(probs[j]))
      hchain.append(hstate)
  hchain[0]=0
  return hchain
def haar_2d(nrep,x2d,y2d):
  hchain=[]
  for j in range (nrep+1):
      qc1 = QuantumCircuit(x2d*y2d)
      for k in range (j): circ_haar_2d(qc1,x2d,y2d)
      qc1.save_statevector()
      state = sim.run(qc1,shots=1).result().get_statevector()
      probs = Statevector(state).probabilities()
      hstate=0;
      for j in range(2**(x2d*y2d)): hstate+=-(probs[j]*np.log(probs[j]))
      hchain.append(hstate)
  hchain[0]=0
  return hchain
def hbuilder1d(n1d):
    hfinal=np.zeros((2**n1d,2**n1d))
    hfinal=hfinal.astype(complex)
    lst= [np.identity(2) for j in range(n1d)]
    for k in range (n1d):
       lst[k]=hx*s1;
       hfinal += ft.reduce(np.kron, lst)
       lst[k]=np.identity(2)
    for k in range (n1d-1):
       lst[k]=jx*s1;lst[k+1]=jx*s1
       hfinal += ft.reduce(np.kron, lst)
       lst[k]=np.identity(2);lst[k+1]=np.identity(2)
    for k in range (n1d-1):
       lst[k]=jy*s2;lst[k+1]=jy*s2
       hfinal += ft.reduce(np.kron, lst)
       lst[k]=np.identity(2);lst[k+1]=np.identity(2)
    for k in range (n1d-1):
       lst[k]=jz*s3;lst[k+1]=jz*s3
       hfinal += ft.reduce(np.kron, lst)
       lst[k]=np.identity(2);lst[k+1]=np.identity(2)
    return hfinal
def hbuilder2d(x2d,y2d):
    hfinal=np.zeros((2**(x2d*y2d),2**(x2d*y2d)))
    hfinal=hfinal.astype(complex)
    lst= [np.identity(2) for j in range(x2d*y2d)]
    for k in range (x2d*y2d):
       lst[k]=hx*s1;
       hfinal += ft.reduce(np.kron, lst)
       lst[k]=np.identity(2)
    for j in range (x2d):
        for k in range (y2d-1):
            lst[j*x2d+k]=jx*s1;lst[j*x2d+k+1]=jx*s1
            hfinal += ft.reduce(np.kron, lst)
            lst[j*x2d+k]=np.identity(2);lst[j*x2d+k+1]=np.identity(2)
    for j in range (x2d):
        for k in range (y2d-1):
            lst[j*x2d+k]=jy*s2;lst[j*x2d+k+1]=jy*s2
            hfinal += ft.reduce(np.kron, lst)
            lst[j*x2d+k]=np.identity(2);lst[j*x2d+k+1]=np.identity(2)
    for j in range (x2d):
        for k in range (y2d-1):
            lst[j*x2d+k]=jz*s3;lst[j*x2d+k+1]=jz*s3
            hfinal += ft.reduce(np.kron, lst)
            lst[j*x2d+k]=np.identity(2);lst[j*x2d+k+1]=np.identity(2)
    return hfinal
def dag(self):
        return self.conj().T
def quantum_haar_state(N, d):
    haar_random_state = QuantumCircuit(N)
    def gate_random(circ, qubit):
        A = [1, 2, 3]
        x = random.choice(A)
        if x == 1:
            circ.rx(np.pi/2, qubit)
        elif x == 2:
            circ.ry(np.pi/2, qubit)
        else:
            circ.t(qubit) 
        return x
    def firs_step_of_block(circ, N):
        random_gate_chosen = []
        for i in range(N):
            random_gate_chosen.append(gate_random(circ, i))
        return random_gate_chosen
    def second_step_of_block_A(circ, N):
        for i in np.arange(0, N-1, 2):
            circ.cz(i,i+1)
    def second_step_of_block_B(circ, N):
        for i in np.arange(0, N-2, 2):
            circ.cz(i+1,i+2)
    def firs_step_of_block_after_first_block(circ, N, list):
        chosen_gates = []
        for _ in list:
            available_numbers = [1, 2, 3]
            available_numbers.remove(_)
            random_number = random.choice(available_numbers)
            chosen_gates.append(random_number)
        for qubit in range(N):    
            x = chosen_gates[qubit]
            if  x == 1:
                circ.rx(np.pi/2, qubit)
            elif x == 2:
                circ.ry(np.pi/2, qubit)
            elif x == 3:
                circ.t(qubit)
        return chosen_gates
    if d >= 1:
        rcg = firs_step_of_block(haar_random_state, N)
        haar_random_state.barrier()
        second_step_of_block_A(haar_random_state, N)
        haar_random_state.barrier()
    if d >= 2:
        rcg = firs_step_of_block_after_first_block(haar_random_state, N, rcg)
        haar_random_state.barrier()
        second_step_of_block_B(haar_random_state, N)
    if d > 2:    
        for i in range(int(d/2)-1):
            haar_random_state.barrier()
            rcg = firs_step_of_block_after_first_block(haar_random_state, N, rcg)
            haar_random_state.barrier()
            second_step_of_block_A(haar_random_state, N)
            haar_random_state.barrier()
            rcg = firs_step_of_block_after_first_block(haar_random_state, N, rcg)
            haar_random_state.barrier()
            second_step_of_block_B(haar_random_state, N)
    if d >= 3:        
        if d%2 != 0:
            haar_random_state.barrier()
            rcg = firs_step_of_block_after_first_block(haar_random_state, N, rcg)
            haar_random_state.barrier()
            second_step_of_block_A(haar_random_state, N)
    simulator = Aer.get_backend('statevector_simulator')
    transpiled_circuit = transpile(haar_random_state, simulator)
    job = assemble(transpiled_circuit)
    result = simulator.run(job).result()
    final_state_vector = result.get_statevector()
    #print("Final State Vector:")
    #print(Statevector(final_state_vector))
    prob_state_vector = Statevector(final_state_vector).probabilities()
    #print(prob_state_vector)
    probabilities = np.abs(final_state_vector) ** 2
    Haar_entropy = 0
    for prob in prob_state_vector:
        if prob == 0:
            Haar_entropy += 0
        else :
            Haar_entropy += prob*np.log(prob)
    return -Haar_entropy, haar_random_state, final_state_vector
def transverse_field_heisenberg(N, h, jx, jy, jz):
    id = np.array([[1, 0], [0, 1]])
    σx = np.array([[0, 1], [1, 0]])
    σz = np.array([[1, 0], [0, -1]])
    σy = np.array([[0, -1j], [1j, 0]])
    first_term_ops = [id.copy() for _ in range(N)]
    first_term_ops[0] = σx.copy()
    first_term_ops[1] = σx.copy()
    first_term_ops_n = [id.copy() for _ in range(N)]
    first_term_ops_n[0] = σx.copy()
    first_term_ops_n[1] = σx.copy()
    second_term_ops = [id.copy() for _ in range(N)]
    second_term_ops[0] = σy.copy()
    second_term_ops[1] = σy.copy()
    second_term_ops_n = [id.copy() for _ in range(N)]
    second_term_ops_n[0] = σy.copy()
    second_term_ops_n[1] = σy.copy()
    third_term_ops = [id.copy() for _ in range(N)]
    third_term_ops[0] = σz.copy()
    third_term_ops[1] = σz.copy()
    third_term_ops_n = [id.copy() for _ in range(N)]
    third_term_ops_n[0] = σz.copy()
    third_term_ops_n[1] = σz.copy()
    fourth_term_ops = [id.copy() for _ in range(N)]
    fourth_term_ops[0] = σx.copy()
    fourth_term_ops_n = [id.copy() for _ in range(N)]
    fourth_term_ops_n[0] = σx.copy()
    H_dim = 2 ** N
    H = np.zeros((H_dim, H_dim))
    indexs = range(N)
    for i in range(1, N):
        first_term_tensor = np.kron(first_term_ops[N-2], first_term_ops[N-1])
        for j in range(3, N+1):first_term_tensor = np.kron(first_term_ops[N-j], first_term_tensor)
        H = H - jx*first_term_tensor
        indexs = np.roll(indexs, 1)
        first_term_ops = [first_term_ops_n[k] for k in indexs]
    indexs = range(N)
    for i in range(1, N):
        second_term_tensor = np.kron(second_term_ops[N-2], second_term_ops[N-1])
        for j in range(3, N+1):second_term_tensor = np.kron(second_term_ops[N-j], second_term_tensor)
        H = H - jy*second_term_tensor
        indexs = np.roll(indexs, 1)
        second_term_ops = [second_term_ops_n[k] for k in indexs]
    indexs = range(N)
    for i in range(1, N):
        third_term_tensor = np.kron(third_term_ops[N-2], third_term_ops[N-1])
        for j in range(3, N+1):third_term_tensor = np.kron(third_term_ops[N-j], third_term_tensor)
        H = H - jz*third_term_tensor
        indexs = np.roll(indexs, 1)
        third_term_ops = [third_term_ops_n[k] for k in indexs]
    indexs = range(N)
    for i in range(N):
        fourth_term_tensor = np.kron(fourth_term_ops[N-2], fourth_term_ops[N-1])
        for j in range(3, N+1):
            fourth_term_tensor = np.kron(fourth_term_ops[N-j], fourth_term_tensor)
        H = H - h * fourth_term_tensor
        indexs = np.roll(indexs, 1)
        fourth_term_ops = [fourth_term_ops_n[k] for k in indexs]
    return H

def block_encoding_amplification(N,d,H,beta):
    entropy, circuit, state_vector= quantum_haar_state(N, d)
    circuit.add_register(QuantumRegister(N+1))
    #circuit.draw('mpl')
    Q = expm(-beta*H/2.0)
    circ_f, alpha = fable(Q, 0)
    circuit.add_register(ClassicalRegister(N+1))
    circuit.barrier()
    circuit.compose(circ_f, qubits=range(2*N + 1), inplace=True)
    circuit.barrier()
    circuit.measure(range(N, 2*N+1), range(N+1))
    circuit.save_statevector(label = 'test', pershot = True)
    backend = Aer.get_backend("statevector_simulator")
    #result = execute(circuit, backend = backend, shots = 100).result()
    new_circuit = transpile(circuit, backend=backend)
    result = backend.run(new_circuit, shots=100).result()
    #array_to_latex(result.data(0)['test'][10])
    for i in range(99):
        psi_ancillar = partial_trace(result.data(0)['test'][i], range(N))
        psi_ancillar = np.diagonal(psi_ancillar)
    circ_f, alpha = fable(Q, 0)
    good_ancilla = np.asarray(Statevector.from_label('0'*(N+1)))
    np.shape(np.outer(good_ancilla, good_ancilla.T))
    np.shape(good_ancilla.conj().T)
    Rgood = np.kron((np.identity(2**(N+1))-2*np.outer(good_ancilla, good_ancilla.conj())), np.identity(2**(N)))
    good_state = np.asarray(Statevector.from_label('0'*(2*N+1)))
    middle = np.outer(good_state, good_state.conj().T)
    R_psi_middle = 2*middle - np.identity(2**(2*N+1))
    G = QuantumCircuit(2*N + 1)
    Rgood = Operator(Rgood)
    G.append(Rgood, range(2*N+1))
    G.append(Rgood, range(2*N+1))
    G.compose(circ_f, range(2*N+1), inplace=True)
    R_psi_middle = Operator(R_psi_middle)
    G.append(R_psi_middle, range(2*N+1))
    G.append(circ_f.inverse(), range(2*N+1))
    entropy, circuit, state_vector= quantum_haar_state(N, d)
    circuit.add_register(QuantumRegister(N+1))
    for j in range(10): circuit.compose(G, range(2*N+1))
    circuit.barrier()
    circuit.add_register(ClassicalRegister(N+1))
    circuit.measure(range(N,2*N+1), range(N+1))
    circuit.save_statevector(label = 'test', pershot = True)
    new_circuit = transpile(circuit, backend=backend)
    result2 = backend.run(new_circuit, label='test2').result()
    return result2.data(0)
