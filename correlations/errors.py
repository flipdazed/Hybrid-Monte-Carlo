import numpy as np

from hmc import checks
from acorr import acorr as getAcorr

import matplotlib.pyplot as plt
from matplotlib import colors
import random
from plotter import Pretty_Plotter, PLOT_LOC

__doc__ = """This code is based on ``Monte Carlo errors with less errors''
- by Ulli Wolff, hep-lat/0306017

The aim is to make the routine more readable and simplified with python
"""
def getW(itau, itau_diff, n):
    """Recreates integer w from the output of uWerr
    
    Required Inputs
        itau        :: float :: integrated a/c time
        itau_diff   :: float :: error in the integrated a/c time
        n           :: int   :: number of original MCMC samples
    """
    if np.isnan(itau): return np.nan
    w  = np.around((itau_diff/itau/2.)**2*n - .5 + itau, 0)
    return np.rint(w).astype(int)
    
def acorrnErr(acn, w, n):
    """Calculates the errors in the autocorrelations
    construct errors acc. to hep-lat/0409106 eq. (E.11)
    
    Required Input
        acn    :: np.ndarray :: normalised autocorrelations
        w      :: int :: cuttoff for the error summation - suggest w_best
        n      :: int :: number of MCMC samples
    """
    err = []
    if not isinstance(w, int): 
        if np.isnan(w): return np.array([np.nan]*acn.size)
        w = int(w)
    l = acn.size
    pd = np.zeros(2*l+w, dtype='float64')   # make sure enough room
    pd[:acn.size] = np.nan_to_num(acn)
    for t in range(0,l): # this is a horrible loop 
        tmp = 0                 # but doesn't run that slow
        for k in range(max(0,t-w), t+w):
            tmp += (pd[k+t] + pd[abs(k-t)] - 2*pd[t]*pd[k])**2
        err.append(np.sqrt(tmp/n))
    return np.asarray(err, dtype='float64')
#
def itauErrors(itau, n, window = None):
    """Calculates the standard deviation in the integrated autocorrelations
    If window is None then the function can calculate errors varying with W
    
    Required Inputs
        acorrn :: np.ndarray :: normalised autocorrelation function
        n      :: int        :: number of MCMC samples
    
    Optional
        window :: int  :: optional window to integrate up to
    """
    if window is None: window = np.arange(itau.size)
    
    # check that the error can be calculated
    if hasattr(window, '__iter__'):
        for i, (window_i, itau_i) in enumerate(zip(window, itau)):
            print 'Warning: {}th Window is too small.\n > W = {}; \n > itau = {}'.format(
                i, window_i, itau_i) + \
                '\n result: {}'.format(window_i - itau_i + .5)
            return np.empty(itau.shape)
    else:
        checks.tryAssertLtEqual(0, window - itau + .5,
            'Window is too small.\n > W = {}; \n > itau = {}'.format(window, itau))
    
    return itau*2*np.sqrt((window - itau + .5)/n)
#
def intAcorr(acorrn, n, window = None):
    """Calculates the integrated autocorellations by integrating
    up to a window length, w, across the autocorrelation function
    
    Required Inputs
        acorrn :: np.ndarray :: normalised autocorrelation function
        n      :: int        :: number of MCMC samples
    
    Optional
        window :: int  :: optional window to integrate up to
    
    Returns
        itau      :: float :: integrated autocorrelation function
        itau_diff :: float :: errors in itau
        itau_aav  :: np.ndarray :: itau at each window length
    
    Notes
        The correction of - 0.5,
            $$\bbar{C}_F(W) = \Lambda_F(0) + 2\sum_1^W \Lambda_F(W)$$
        estimating for $\bbar{\nu}_F \approx \Lambda_F(0)$ then,
            $$\bbar{\tau_{int},F}(W) = \frac{\bbar{C}_F(W)}{2\bbar{\nu}_F}$$
            $$\bbar{\tau_{int},F}(W) = 0.5 + \frac{2}{\bbar{\nu}_F}\sum_1^W \Lambda_F(W)$$
    """
    if window is None: window = acorrn.size # assume alrady windowed
    itau_aav  = np.cumsum(np.nan_to_num(acorrn)) - .5      # Eq. (41)
    itau = itau_aav[window]
    itau_diff = itauErrors(itau, n, window=window)
    return itau, itau_diff, itau_aav
#
def gW(t, g_int, s_tau, n):
    """Calculates g_W as in Eq. (52)
    
    Required Inputs
        t :: int :: MCMC sample time
        g_int :: float :: somewhere around Eq.50?
        s_tau :: float :: S(t) in the equation
        n     :: int   :: f_ret.size
    """
    # np.nan handles the divisino by 0 fine (tested)
    t = float(t)
    tau_w = s_tau / np.log(1. + 1./g_int.clip(0))
    g_w = np.exp(-t/tau_w) - tau_w/np.sqrt(t*float(n))
    return g_w
#
def covarianceN(acorr, window, var = None):
    """Covariance*N as defined in Eq. (12, 26, 35)
    Practical estimate of variance is the t=0 autocorrelation
    as per Eq. 35 which is taken as the normalisation
    
    Required Inputs
        acorr :: np.ndarray :: unnormalised autocorellations
        window :: int  :: optional window to integrate up to
    
    Optional Inputs
        var   :: float :: variance estimate of the underlying function
    
    Comments:
    Possible error in the indexing of the matlab arrays that I carried forwards
    see uWerr for more comments
    """
    if var is None: var = acorr[0]
    c = var + 2.*acorr[1:window+1].sum()
    checks.tryAssertLt(0, c, 'Estimated error^2 < 0...\n sum = {}'.format(c))
    return c
#
def autoWindow(acorrn, s_tau, n, t_max = None):
    """Automatic windowing procedure as 
    described in Section 3.3 in the paper
    
    Required Inputs
        acorrn :: np.ndarray :: normalised autocorellations
        s_tau :: float      :: guess for the ratio S of tau/itau
    
    Optional Inputs
        t_max :: int :: maximum window length
    
    w is returned when gW changes sign from positive to negative
    the t counter starts at 1 and evaluates using next() comprehension
    
    If not supplied, it is assumed that the length of the acorrn has
    already been adjusted w.r.t. the n//2 parameter for t_max
    """
    if t_max is None: t_max = acorrn.size
    g_int = np.cumsum(np.nan_to_num(acorrn[1:t_max]))
    
    # see http://stackoverflow.com/a/8534381/4013571
    try:
        w = next(t for t,v in enumerate(g_int,1) if gW(t, v, s_tau, n) < 0)
    except:
        print 'Error: Windowing condition failed up to W = {}'.format(g_int.size)
        return None
        # UWerr actually returns min(t_max, 2*t) anyway
    return w
#
def windowing(f_ret, f_aav, s_tau, n, fast):
    """
    Required Inputs
        f_ret   :: np.ndarray :: the return of a function action upon all f_ret
        f_aav   :: float :: the average value of f_ret
        s_tau   :: float>0 :: guess for the ratio S of tau/tauint [D=1.5]
        n       :: int        :: number of MCMC samples
        fast    :: bool :: Determines fast or slow method
    """
    
    # get autocorrelations: Don't normalise until bias corrected
    norm = ((f_ret-f_aav)**2).mean()
    fn = lambda t: getAcorr(op_samples=f_ret, mean=f_aav, separation=t, norm=norm)
    t_max = int(n//2) + 1
    if fast:
        acorr = []
        acorr.append(norm)
        g_int = 0.
        for w in range(1, t_max + 1):
            v = fn(t=w)
            acorr.append(v)
            g_int += v
            if gW(w, g_int, s_tau, n) < 0: return norm, np.asarray(acorr), w
        
        ## should exit before here!
        print 'Error: Windowing condition failed up to W = {}'.format(g_int.size)
        return None, None, None
    else:
        acorr  = np.asarray([fn(t=t) for t in range(0, t_max)]) # t_max implicit n//2
        checks.tryAssertNotEqual(norm, 0,
            'Normalisation cannot be zero: No fluctuations.' \
            + '\nNormalisation: {}'.format(norm))
        
        # The automatic windowing proceedure
        w = autoWindow(acorrn=acorr, s_tau=s_tau, n=n)
        return norm, acorr, w
    checks.tryAssertNotEqual(False, False, "Shouldn't get here! wtf...?!")
    pass
#
def uWerr(f_ret, acorr=None, s_tau=1.5, fast_threshold=5000):
    """autocorrelation-analysis of MC time-series following the Gamma-method
    This (simplified) implementation assumes f_ret have been acted upon by an operator
    and just completes basic calculations
    
    **Currently only supports ONE observable where # samples == f_ret.shape[0]**
    
    Required Inputs
        f_ret   :: np.ndarray :: the return of a function action upon all f_ret
        acorr   :: np.ndarray :: provide autocorrelations if already calculated
    
    Optional Inputs
        s_tau   :: float>0 :: guess for the ratio S of tau/tauint [D=1.5]
        fast_threshold :: int :: determines at what size array we use the faster method
    Notation notes:
        x_av0 is an average of x over the 0th dim - \bar{x}^r in the paper
        x_aav is the average over all dims        - \bbar{x} in the paper
        v_err is the error in v                   - \sigma_F in the paper
        v_eerr is the error of the error          - given by Eq. 40 : no symbol in paper
    
    s_tau or Stau in the original code doesn't work for the 0 case so I removed it.
    see https://github.com/flipdazed/Hybrid-Monte-Carlo/issues/34#issuecomment-232472657
    for info.
    
    Comments:
    I think U.Wolf might have made a mistake with the indexing of Matlab. When calculating
    his version of covarianceN() he uses an array of (2:Wopt + 1) however in matlab
    this includes the final indexed item, arr[Wopt+1]. This doesn't seem right as in
    all other calculations he explicitly mentions that matlab has a mapping of indices 
    from w -> w+1 but I think he has just forgotten it is actually a mapping on both 
    sides of the indexing (w->w+1 : w -> w) with respect to the paper!
    This is also noted in comparing to the matlab code for itau_aav, acorr where
    to get the correct shape we must use 2w+1 as the final index.
    """
    checks.tryAssertNotEqual(s_tau, 0,
        's_tau cannot be zero, see:\n' \
        + 'https://github.com/flipdazed/Hybrid-Monte-Carlo' \
        + '/issues/34#issuecomment-232472657')
    
    if not isinstance(f_ret, np.ndarray): f_ret = np.ndarray(f_ret)
    checks.tryAssertEqual(len(f_ret.shape[1:]), len(set(f_ret.shape[1:])),
        'Only expects cuboid lattices: dims >2 are not equal.' \
        + '\nShape: {}'.format(f_ret.shape))
    
    f_aav = f_ret.mean()                # get the mean of the function outputs
    n = float(f_ret.shape[0])           # number of MCMC samples
    
    if acorr is not None:
        norm = acorr[0]
        w = autoWindow(acorr/norm, s_tau, n, t_max = None)
    else:
        fast = (n >= fast_threshold)
        norm, acorr, w = windowing(f_ret, f_aav, s_tau, n, fast=fast)
    if w == None: 
        nans = np.empty(acorr.shape)
        nans[:] = np.nan
        return f_aav, np.nan, np.nan, np.nan, np.nan, nans, acorr/acorr[0]
    
    l = max(acorr.size, 2*w+1)
    
    # correct acorr for variance in the function
    c_aav  = covarianceN(acorr=np.nan_to_num(acorr), window=w, var=norm) # var = c_aav/n     Eq. 35
    acorr += c_aav/n                                        # correct for bias  Eq. 32,49
    norm   = acorr[0]
    c_aav  = covarianceN(acorr=np.nan_to_num(acorr), window=w, var=norm)
    acorrn = acorr/norm                                     # normalise corrected a/c fn.
    
    itau, itau_diff, itau_aav = intAcorr(acorrn=acorrn, n=n, window=w)
    f_diff  = np.sqrt(c_aav/n)             # error of f from       Eq. (26,44)
    f_ddiff = f_diff*np.sqrt((w + .5)/n)   # error on error of f   Eq. (42)
    
    # return relevant values - perhaps this is all better as a class?
    return f_aav, f_diff, f_ddiff, itau, itau_diff, itau_aav[:l], acorrn[:l]

if __name__ == '__main__':
    pass