import numpy as np

from hmc import checks 

__doc__ = """This code is based on ``Monte Carlo errors with less errors''
 - by Ulli Wolff, hep-lat/0306017

The aim is to make the routine more readable and simplified with python
"""

def uWerr(f_ret, s_tau, quantity = 1, name = None, n_rep = None, **kwargs):
    """autocorrelation-analysis of MC time-series following the Gamma-method
    This (simplified) implementation assumes f_ret have been acted upon by an operator
    and just completes basic calculations
    
    **Currently only supports ONE observable of length sample.shape[0]**
    
    Required Inputs
        f_ret   :: np.ndarray :: the return of a function action upon all f_ret
        s_tau   :: float :: guess for the ratio S of tau/tauint [D=1.5]
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
    n = float(f_ret.size)
    
    # a_av0 is equal to a_aav beacuse we have no replicas here
    # note that f_aav == a_aav because R=1
    f_aav = a_aav = np.average(f_ret)   # get the mean of the function outputs
    diffs = f_ret - a_aav               # get fluctuations around mean
    
    ### Computation of Gamma, automatic windowing
    norm = (diffs**2).mean() # values for w = 0
    
    check.tryAssertNotEqual(norm, 0,
        msg = 'Normalisation cannot be zero: No fluctuations.' \
        + '\nNormalisation: {}'.format(norm))
      
    acorr = [norm] # first element is the norm
    
    # compute Gamma(t), find the optimal W
    flag = (s_tau == 0)
    
    if flag: # 0 = assumed absence of autocorrs
        w_best      = 0.
        t_max       = 0.
    else:
        t_max       = f_ret.size/2 # note integer division deliberate
        g_int       = 0.
    
    for t in in range(1, t_max):
        # legacy:  a sum over the replica - can ignore as r=1 and just use:
        # gamma_f_i += delpro[:n-t]*delpro[t:n] sucks in python syntax
        # instead this can be replaced by a np.roll and slice off the end bits
        # and then the mean can be taken all in one go
        rickrolled = np.roll(diffs, t) # this totally wasn't a waste of a line of code
        acorr_t = diffs*rickrolled[:n-t].mean() # bosh.
        
        acorr.append(acorr_t) # append to gammas
        
        # this while section below can be improved drastically
        ######
        if flag: # this flag occurs if s_tau != 0 - should be explict
            g_int += acorr_t/norm
            # np.nan handles the divisino by 0 fine (tested)
            tau_w = s_tau / np.log(1. + 1./g_int.clip(0))
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

if __name__ == '__main__':
    plot(x, np.exp(-1./np.log(1. + 1./x.clip(0))) - x, label='0 giving 1/0 = np.inf', linewidth=2., alpha = .4)
    plot(x, np.exp(-1./np.log(1. + 1./x.clip(np.finfo(float).eps))) - x, label='eps', linewidth=2., alpha = .4)
    pass