from fourier import *
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


num_fourier = 5
num_qubits = 4
xlower = -1
xupper = 1
xupper_extension = 1*(2*xupper - xlower) 
mean = 0.5*(xupper+xlower)

# distribution = stats.uniform(xlower, xupper-xlower).pdf
#distribution = stats.norm(mean, 1).pdf
distribution = lambda x : np.exp(-x)

num_points = 2**num_qubits

xpoints = np.linspace(xlower, xupper, num_points) 
prob_dist = distribution(xpoints)


period = xupper_extension- xlower
omega = 2*np.pi /period
delta = (xupper-xlower)/(num_points - 1)
pdf_normalized = prob_dist/prob_dist.sum()
pdf_amplitudes = np.sqrt(pdf_normalized)

qc = create_cirquit(pdf_amplitudes)

n=1
res = integral(qc, n, 0, omega, delta, xlower, epsilon=0.001, conf_lvl=0.01)

classical = np.dot(pdf_normalized, np.cos(n*omega*xpoints))

print(res, classical)
