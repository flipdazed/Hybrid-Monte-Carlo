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
    n = float(samples.size)
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
    
    # values for W=0:
    norm = (delpro**2).mean()
    
    check.tryAssertNotEqual(norm, 0,
        msg = 'Normalisation cannot be zero: No fluctuations.' \
        + '\nNormalisation: {}'.format(norm))
      
    acorr = [norm] # first element is the norm
    
    # compute Gamma(t), find the optimal W
    if s_tau == 0: # 0 = assumed absence of autocorrs
        w_best   = 0.
        t_max       = 0.
        flag        = False
    else:
        t_max       = samples.size/2 # note integer division deliberate
        flag        = True
        g_int       = 0.
    
    for t in in range(1, t_max):
        # legacy:  a sum over the replica - can ignore as r=1 and just use:
        # gamma_f_i += delpro[:n-t]*delpro[t:n] sucks in python syntax
        # instead this can be replaced by a np.roll and slice off the end bits
        # and then the mean can be taken all in one go
        rickrolled = np.roll(delpro, t) # this totally wasn't a waste of a line of code
        acorr_t = delpro*rickrolled[:n-t].mean() # bosh.
        
        acorr.append(acorr_t) # append to gammas
        
        # this while section below can be improved drastically
        ######
        if flag: # this flag occurs if s_tau != 0 - should be explict
            g_int += acorr/norm
            if g_int <= 0: # calculate g_int
                tau_w = np.finfo(float).eps
            else:
                tau_w = s_tau / np.log(1.+1./g_int)
            g_w = np.exp(-t/tau_w) - tau_w/np.sqrt(t*n)
            
            # look for a sign switch in g_w
            # optimal w occurs when sign changes
            if g_w < 0:
                w_best = t
                t_max = min(t_max, 2*w_best)
                flag = False #  now we want to calculate gamma up to tmax
        ######
    # end of loop over t
    
    check.tryAssertNotEqual(flag, False,
        msg = 'Windowing condition failed up to W = {}'.format(t_max))
    
    acorr = np.asarray(acorr) # make into a np.ndarray for rest
    
    # this starting point is defined as Eq. 35 in the paper
    g_sum = acorr(1:w_best).sum()
    c_aav = norm + 2.*g_sum
    
    check.tryAssertLt(flag, g_sum,
        msg = 'Estimated error^2 as sum(gamma) < 0...\n sum = {}'.format(g_sum)
    
    acorr  += c_aav/n                       # bias in Gamma corrected
    c_aav = norm + 2*acorr(1:w_best).sum()  # refined estimate
    sigma_f = np.sqrt(c_aav/n)              # error of F
    acorrn  = acorr/norm                    # normalized autocorr.
    itau_aav = np.cumsum(acorrn) - .5
    
    v       = f_aav
    v_diff  = sigmaF
    v_ddiff = dvalue*sqrt((w_best + .5)/n)
    itau    = itau_aav(w_best + 1.)
    itau_diff = int_tau*2*np.sqrt((w_best - int_tau + .5)/n)
    
    return v, v_diff, v_ddiff, itau, itau_diff