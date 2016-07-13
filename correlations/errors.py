import numpy as np

from hmc import checks 

__doc__ = """This code is based on ``Monte Carlo errors with less errors''
 - by Ulli Wolff, hep-lat/0306017

The aim is to make the routine more readable and simplified with python
"""

def uWerr(samples, sample_obs, s_tau, quantity = 1, name = None, n_rep = None, **kwargs):
    """autocorrelation-analysis of MC time-series following the Gamma-method
    This (simplified) implementation assumes samples have been acted upon by an operator
    and just completes basic calculations
    
    **Currently only supports ONE observable of length sample.shape[0]**
    
    Required Inputs
        samples  :: np.ndarray :: n samples
        sample_ops :: np.ndarray :: observables corresponding to n_samples
        s_tau    :: float :: guess for the ratio S of tau/tauint [D=1.5]
                             0 = assumed absence of autocorrs
    
    Optional Inputs
        name     :: str/None :: observable titles for plots. None: no plots.
        n_rep    :: np.ndarray :: list of int: len of each data for replica of len n_rep
        quantity :: int :: shouldn't be used
    
    Notation notes:
        x_av0 is an average of x over the 0th dim - \bar{x}^r in the paper
        x_aav is the average over all dims        - \bbar{x} in the paper
        v_err is the error in v                   - \sigma_F in the paper
        v_eerr is the error of the error          - given by Eq. 40 : no symbol in paper
    """
    
    # legacy: n_rep is set up as the number of entries in the data if reps = None
    n = samples.size
    # primary_index = 1 # legacy - don't need
    primary = True      # legacy - don't need as we do all func evals before
    
    # legacy: find the means of the primaries
    # a_av0 is equal to a_aav beacuse we have no replicas here
    a_aav = np.average(samples)
    
    # legacy: means of derived / primary - depending on quantity
    # here quantity is 1 and primary = True
    f_aav = a_aav[idx1]
    
    # legacy: form the gradient of f and project fluctuations:
    delpro = samples - a_aav
    
    ### possibly add another function? ###
    ### Computation of Gamma, automatic windowing
    # note: W=1,2,3,.. in Matlab-vectorindex <-> W=0,1,2,.. in the paper!
    
    # legacy - set up some defaults incase the error analysis fails
    v       = f_aav
    v_diff  = 0.
    v_ddiff = 0.
    q_val   = []
    itau        = 0.5
    itau_diff   = 0.
    
    # values for W=0:
    normalisation = np.average(delpro**2.)
    
    check.tryAssertNotEqual(normalisation, 0,
        msg = 'Normalisation cannot be zero: No fluctuations.' \
        + '\nNormalisation: {}'.format(normalisation))
      
    gamma_f_aav = [normalisation] # first element is the normalisation
    
    # compute Gamma(t), find the optimal W
    if s_tau == 0: # 0 = assumed absence of autocorrs
        w_optimal   = 0.
        t_max       = 0.
        flag        = False
    else:
        t_max       = samples.size/2 # note integer division deliberate
        flag        = True
        g_int       = 0.
    
    for t in in range(t_max):
        gamma_f_i = 0.
        # legacy:  a sum over the replica - can ignore as r=1 and just use:
        gamma_f_i += delpro[:n-t]
        