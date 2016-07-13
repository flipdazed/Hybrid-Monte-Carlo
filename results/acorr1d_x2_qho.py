
import numpy as np

from common import acorr1d_x2
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

file_name = __file__
pot = QHO(m0=1., mu=1.)

n, dim = 20, 1
x0 = np.random.random((n,)*dim)
spacing = 1.

n_samples, n_burn_in = 110, 25
step_size =   .1
n_steps   =  1
c_len     = 10

mixing_angles = 1/np.arange(8,1,-2, dtype='float64')*np.pi
angle_labels = [
    r'$\theta = \pi/8$',
    r'$\theta = \pi/6$',
    r'$\theta = \pi/4$',
    r'$\theta = \pi/2$']

if '__main__' == __name__:
    acorr1d_x2.main(x0, pot, file_name,
        n_samples = n_samples, n_burn_in = n_burn_in, spacing = spacing,
        c_len = c_len, mixing_angles=mixing_angles, angle_labels = angle_labels,
        step_size = step_size, n_steps = n_steps,
        save = True)
