from fourier import *
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


num_fourier = 5
num_qubits = 5
xlower = 0
xupper = 6

func = lambda x: x * 2
func_derivative = lambda x: 2

distribution = stats.uniform(xlower, xupper-xlower)
distribution = stats.norm()

xupper_extension = 0.5*(3*xupper - xlower) 
num_points = 2**num_qubits
x_piecewise = [xlower, xupper, xupper_extension]
pdf_domain_points = np.linspace(x_piecewise[0], x_piecewise[1], 2**num_qubits)
pdf = distribution.pdf(pdf_domain_points)
xpoints = np.linspace(xlower, xupper, num_points) 
plt.stem(xpoints, pdf)


x = np.linspace(xlower, xupper_extension, 2000, endpoint=True)
 

# Extended Fourier
Func = periodic_extension(func, func_derivative, xlower, xupper, xupper_extension)
y = Func(x)

coeffs = dft(y, x, num_fourier)

epsilons = np.linspace(0.1, 0.001, 5)
real = 1
total_error = []
for epsilon in epsilons:

    expected_value_quantum = sum_estimation(pdf, coeffs, x_piecewise, epsilon=epsilon)
    total_error.append((expected_value_quantum-real)**2)


plt.plot(epsilons, total_error)