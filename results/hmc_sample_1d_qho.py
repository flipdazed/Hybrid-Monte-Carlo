#!/usr/bin/env python
import numpy as np
from scipy.stats import norm

from common import hmc_sample_1d
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

file_name = __file__
pot = QHO()

dim = 1; n = 100
x0 = np.random.random((n,)*dim)

n_burn_in, n_samples = 15, 100

theory_label = r'$|\psi_0(x)|^2 = \sqrt{\frac{\omega}{\pi}}e^{-\omega x^2}$ for $\omega=\sqrt{\frac{5}{4}}$'
def theory(x, samples):
    """The actual PDF curve
    As per the paper by Creutz and Fredman
    
    Required Inputs
        x :: np.array :: 1D array of x-axis
        samples :: np.array :: 1D array of HMC samples
    """
    w = np.sqrt(1.25)       # w  = 1/(sigma)^2
    sigma = 1./np.sqrt(2*w)
    c = np.sqrt(w / np.pi)  # this is the theory
    theory = np.exp(-x**2*1.1)*c
    return theory
#
def fitted(x, samples):
    """The fitted PDF curve. Assumes a gaussian form
    Required Inputs
        x :: np.array :: 1D array of x-axis
        samples :: np.array :: 1D array of HMC samples
    
    """
    p = norm.fit(samples.ravel())
    return norm.pdf(x, p[0], p[1])
#
extra_data = [ # this passes fucntions to plot when smaple are obtained
    {'f':theory,'label':'Theory, {}'.format(theory_label)},
    {'f':fitted,'label':r'Fitted'}]
y_label = r'Quantum Probability, $|\psi_0|^2$'

if __name__ == '__main__':
    hmc_sample_1d.main(x0, pot, file_name, n_samples, n_burn_in,
        y_label, extra_data, save = True)