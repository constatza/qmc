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

    
def amplitude_estimation(qc, alpha, theta, epsilon=.01, conf_lvl=.01, backend=backend):
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
    complex_coeffs = np.fft.rfft(y, len(y))/len(y)
    np.put(complex_coeffs, range(terms+1, len(complex_coeffs)), 0.0) 
    complex_coeffs = complex_coeffs[:terms+1]
    
    if return_complex:
        return complex_coeffs
    else:
        complex_coeffs *= 2
        return complex_coeffs.real[0]/2, complex_coeffs.real[1:-1], -complex_coeffs.imag[1:-1]


cubic_base = lambda x: x**np.arange(4)
cubic_base_derivative = lambda x: np.arange(4)*x**np.array([0, 0, 1, 2])

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


def integral(qc, n, beta, omega, delta, xlower, epsilon=0.01, conf_lvl=0.01):
    alpha = n*omega*xlower - beta
    theta = n*omega * delta
    backend = Aer.get_backend('aer_simulator')
    phase_good = amplitude_estimation(qc, alpha, theta, epsilon=epsilon, conf_lvl=conf_lvl, backend=backend) 
    return 1 - 2* phase_good


def sum_estimation(pdf, fourier_coeffs, x_piecewise, epsilon=0.001, conf_lvl=0.01):
    num_points = pdf.shape[0]
    xlower, xupper, xupper_extension = x_piecewise
    period = xupper_extension- xlower
    omega = 2*np.pi /period
    delta = (xupper-xlower)/(num_points - 1)
 
    pdf_normalized = pdf/pdf.sum()
    pdf_amplitudes = np.sqrt(pdf_normalized)
    qc = create_cirquit(pdf_amplitudes)

    num_fourier = fourier_coeffs.shape[0]
    for n in range(1, num_fourier):
        cos_sum = integral(qc, n, 0, omega, delta, xlower, epsilon=epsilon, conf_lvl=conf_lvl)
        sin_sum = integral(qc, n, np.pi/2, omega, delta, xlower, epsilon=epsilon, conf_lvl=conf_lvl)
        fourier_coeffs.real[n] *= 2*cos_sum
        fourier_coeffs.imag[n] *= 2*sin_sum
    s = fourier_coeffs.real.sum() - fourier_coeffs.imag.sum()
    print(f'Im: {fourier_coeffs.imag.sum()}')
    return s

    
def plot_fourier_from_fft(fourier_coeffs, x):
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.fft.irfft(fourier_coeffs, len(x))#*len(x)
    ax.plot(x, y)
    return ax


def fourier_from_sines(coeffs, omega, x):
    coeffs *= 2
    a0 = coeffs.real[0]/2 
    a = coeffs.real[1:] 
    b = coeffs.imag[1:]
    y = a0
    for n in range(a.shape[0]):
        phase = omega*x*(n+1)
        y += a[n]* np.cos(phase) + b[n] * np.sin(phase)
    return y

def dft(y, x, nterms):
    yr = recast_for_dft(y)
    period = x[-1] - x[0]
    coeffs = [yr.mean()]
    for n in range(1, nterms):
        cn = yr*np.exp(-1j*2*n*np.pi*x/period)
        coeffs.append(cn.mean())
    return np.array(coeffs)

def idft(complex_coeffs, x):
    period = x[-1] - x[0]
    fx = complex_coeffs[0]*np.ones_like(x)
    for i in range(1, complex_coeffs.size):
        fx += 2*complex_coeffs[i]*np.exp(1j*2*i*np.pi*x/period) 
    return fx.real

def recast_for_dft(array):
    array[0] += array[-1]
    array[0] /= 2
    array[-1] = 0
    return array

if __name__=='__main__':
    pass

    

