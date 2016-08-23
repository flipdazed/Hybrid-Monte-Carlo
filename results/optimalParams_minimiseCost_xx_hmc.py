from __future__ import division
import numpy as np
from scipy.optimize import brute

from correlations import errors
from correlations.acorr import acorr
from models import Basic_HMC as Model
from results.data import store
from theory.operators import x_sq as opFn

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
    mixing_angle = np.pi/2.
    def minimise_func(args):
        """runs the below for an angle, a
        Allows multiprocessing across a range of values for a
        """
        step_size, n_steps = args
        print 'step size:{}; n steps {}'.format(step_size, n_steps)
        model = Model(x0, pot=pot, rng=rng, step_size = step_size)
        model.run(n_samples=n_samples, n_burn_in=n_burn_in, mixing_angle = mixing_angle, verbose=True)
        
        op_samples = opFn(model.samples)
        
        # get parameters generated
        # pacc = c.model.p_acc
        # pacc_err = np.std(c.model.sampler.accept.accept_rates)
        
        ans = errors.uWerr(op_samples)         # get errors
        op_av, op_diff, _, itau, itau_diff, _, _ = ans   # extract data
        
        return itau#, itau_diff, pacc, pacc_err
    
    bnds = np.array([(0.01, 0.6), (20, 1000)])
    params = brute(minimise_func, ranges=bnds, Ns=16, full_output=True)
    
    store.store(params, file_name, '_params')
