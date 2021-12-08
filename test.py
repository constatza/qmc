# Useful additional packages 
from fourier import *
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


num_fourier = 5
num_qubits = 7
xlower = -4
xupper = 4
xupper_extension = 1*(2*xupper - xlower) 
mean = 0.5*(xupper+xlower)
distribution = stats.uniform(xlower, xupper-xlower)
distribution = stats.norm(mean, 1)
# distribution = stats.poisson
num_points = 2**num_qubits
x_piecewise = [xlower, xupper, xupper_extension]
xpoints = np.linspace(xlower, xupper, num_points) 
prob_dist = distribution.pdf(xpoints)


plt.stem(xpoints, prob_dist)

# Function 
order = 2
func = lambda x: (x )**order
func_derivative = lambda x: order*(x )**(order-1)
x = np.linspace(xlower, xupper_extension, 20000, endpoint=True)
 

# Extended Fourier
Func = periodic_extension(func, func_derivative, xlower, xupper, xupper_extension)
period = xupper_extension - xlower
y = Func(x)


# Q2
coeffs = dft(y, x, num_fourier)
y_complex = idft(coeffs, x)


expected_value_quantum = sum_estimation(prob_dist, coeffs, x_piecewise, 
                                        epsilon=0.001, conf_lvl=0.01)

num_samples = 1000000
samples = distribution.rvs(size=num_samples)
expected_value_classical = func(samples).sum()/num_samples

print(expected_value_quantum, expected_value_classical) 


y_real = fourier_from_sines(coeffs, 2*np.pi/period, x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, y_complex)

plt.show()
