#!/usr/bin/env python
import numpy as np
from scipy.stats import norm

from common import hmc_sample_1d
from hmc.potentials import Simple_Harmonic_Oscillator as SHO

file_name = __file__
pot = SHO()

dim = 1; n = 1
x0 = np.random.random((n,)*dim+(1,))
x0 = np.asarray([[1.]])
n_burn_in, n_samples = 15, 10000

theory_label = r'$f(x) = \sqrt{{\frac{{\omega}}{{\pi}}}}e^{{-\omega x^2}}$ for $\omega=\frac{{1}}{{2}}$'
def theory(x, samples):
    """The actual PDF curve
    
    Required Inputs
        x :: np.array :: 1D array of x-axis
        samples :: np.array :: 1D array of HMC samples
    """
    w = 1.25
    sigma_sq = 1./np.sqrt(2.*w)
    return (None,), norm.pdf(x, 0, 0.79)
def fitted(x, samples):
    """The fitted PDF curve
    
    Required Inputs
        x :: np.array :: 1D array of x-axis
        samples :: np.array :: 1D array of HMC samples
    """
    p = norm.fit(samples)
    return p,norm.pdf(x, p[0], p[1])
#
extra_data = [ # this passes fucntions to plot when smaple are obtained
    {'f':theory,'label':'Theory, {}'.format(theory_label)},
    {'f':fitted,'label':r'Fitted $\mu = {:4.2f}$; $\sigma = {:4.2f}: \omega = \frac{{1}}{{2\sigma^2}}$'}]
    
y_label = r'Sampled Potential, $e^{-V(x)}$'

if __name__ == '__main__':
    hmc_sample_1d.main(x0, pot, file_name, n_samples, n_burn_in,
        y_label, extra_data, save = False)