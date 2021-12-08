#!/usr/bin/env python
# coding: utf-8

# # Quantum Monte Carlo


from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.circuit import AncillaRegister, Parameter
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import Aer
from qiskit.utils import QuantumInstance

backend = Aer.get_backend('aer_simulator')


# Useful additional packages 
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt



def create_cirquit(amplitudes, backend=backend):
    alpha = Parameter('α')
    theta = Parameter('θ')
    npoints = int(np.log2(len(amplitudes)))
    p = QuantumRegister(npoints, name='p')
    ancilla = QuantumRegister(1, name='ancilla')
    qc = QuantumCircuit(p, ancilla)
    qc.initialize(amplitudes, p)


    crots = QuantumCircuit(p, ancilla, name='CRYs')
    crots.ry(alpha, ancilla[0])
    for i in range(npoints):
        crots.cry(2**i  *theta, npoints-1-i, npoints)

    qc.append(crots, p[:] + ancilla[:])

    backend = Aer.get_backend('aer_simulator')
    qc_transpiled = transpile(qc, backend, optimization_level=3)
    return qc_transpiled


    
def amplitude_estimation(qc, alpha, theta, epsilon=.01, conf_lvl=.05, backend=backend):
    # # Quantum Amplitude Estimation using IAE
    par_alpha = qc.parameters[0]
    par_theta = qc.parameters[1]
    qcn = qc.assign_parameters({par_alpha: alpha, 
                          par_theta: theta})


    # construct amplitude estimation

    problem = EstimationProblem(state_preparation=qcn,
                                           objective_qubits=[qcn.num_qubits-1])

    IAE = IterativeAmplitudeEstimation(epsilon_target=epsilon,  alpha=conf_lvl, quantum_instance=backend)

    result_cvar = IAE.estimate(problem)
    return result_cvar.estimation
    



def fft_coeffs(y, terms, return_complex=True):
    complex_coeffs = np.fft.rfft(y, len(y))#/len(y)
    np.put(complex_coeffs, range(terms+1, len(complex_coeffs)), 0.0) 
    complex_coeffs = complex_coeffs[:terms+1]
    
    if return_complex:
        return complex_coeffs
    else:
        complex_coeffs *= 2
        return complex_coeffs.real[0], complex_coeffs.real[1:-1], -complex_coeffs.imag[1:-1]


cubic_base = lambda x: x**np.arange(4)
cubic_base_derivative = lambda x: np.arange(4)*x**np.arange(-1, 3)

def periodic_extension(func, func_derivative, xl, xu, xuu, extension_only=False):
    xl = float(xl)
    xu = float(xu)
    xuu = float(xuu)
    system_matrix = np.vstack([cubic_base(xuu), 
                               cubic_base_derivative(xuu),
                               cubic_base(xu),
                               cubic_base_derivative(xu)])
    
    constraints = np.array([func(xl), func_derivative(xl), func(xu), func_derivative(xu)])
    cubic_coeffs = np.linalg.solve(system_matrix, constraints)
    cubic_value = lambda x: cubic_coeffs.dot(cubic_base(x))
    cubic_extension = np.vectorize(cubic_value)
    if extension_only:
        return cubic_extension
    else:
        return (lambda x: np.piecewise(x, [x <= xu, x>xu], [func, cubic_extension]))


if __name__=='__main__':
    pass

    

