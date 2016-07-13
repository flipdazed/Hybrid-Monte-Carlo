import numpy as np

from common import acorr1d
from hmc.potentials import Klein_Gordon as KG
from correlations.corr import twoPoint

file_name = __file__
pot = KG()

n, dim = 20, 1
x0 = np.random.random((n,)*dim)
spacing = 1.

n_samples, n_burn_in = 1100, 25
step_size =   .1
n_steps   =  1
c_len     = 110

mixing_angles = 1/np.arange(10,1,-2, dtype='float64')*np.pi
angle_labels = [
    r'$\theta = \pi/10$',
    r'$\theta = \pi/8$',
    r'$\theta = \pi/6$',
    r'$\theta = \pi/4$',
    r'$\theta = \pi/2$']

separations = range(c_len)
opFn = lambda samples: twoPoint(samples, separation=0)
op_name = r'$\hat{O} = \langle x(0)x(0) \rangle$'

if '__main__' == __name__:
    acorr1d.main(x0, pot, file_name,
        n_samples = n_samples, n_burn_in = n_burn_in, spacing = spacing,
        mixing_angles=mixing_angles, angle_labels = angle_labels,
        separations = separations, opFn = opFn, op_name = op_name,
        step_size = step_size, n_steps = n_steps,
        save = True)