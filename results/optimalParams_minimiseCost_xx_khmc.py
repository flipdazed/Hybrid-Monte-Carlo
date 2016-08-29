from __future__ import division
import numpy as np
from scipy.optimize import brute

from correlations import errors
from correlations.acorr import acorr
from models import Basic_KHMC as Model
from results.data import store
from theory.operators import x_sq as opFn
from tqdm import tqdm
#
rng = np.random.RandomState()
if __name__ == '__main__':
    from hmc.potentials import Klein_Gordon as KG
    import numpy as np
    
    m         = 0.1
    
    file_name = __file__
    pot = KG(m=m)
    
    n, dim  = 20, 1
    x0 = np.random.random((n,)*dim)
    spacing = 1.
    
    # number of samples/burnin per point
    n_samples, n_burn_in = 100000, 1000
    
    separations = np.arange(150)
    
    def minimise_func(arg):
        """runs the below for an angle, a
        Allows multiprocessing across a range of values for a
        """
        mixing_angle, step_size = arg
        print 'angle:{:10.4f}; step size:{:10.4f}'.format(mixing_angle, step_size)
        model = Model(x0, pot=pot, rng=rng, step_size = step_size)
        model.run(n_samples=n_samples, n_burn_in=n_burn_in, mixing_angle = mixing_angle, verbose=True)
        
        op_samples = opFn(model.samples)
        
        # get parameters generated
        # pacc = c.model.p_acc
        # pacc_err = np.std(c.model.sampler.accept.accept_rates)
        
        ans = errors.uWerr(op_samples)         # get errors
        op_av, op_diff, _, itau, itau_diff, _, _ = ans   # extract data
        
        return itau#, itau_diff, pacc, pacc_err
    
    bnds = [(0.05, 0.3),(0.15, 0.3)]
    params = brute(minimise_func, ranges=bnds, Ns=16, full_output=True)
    
    store.store(params, file_name, '_params')