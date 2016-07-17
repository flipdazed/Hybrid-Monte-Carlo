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

def acorrnErr(acn, w, n):
    """Calculates the errors in the autocorrelations
    construct errors acc. to hep-lat/0409106 eq. (E.11)
    
    Required Input
        acn    :: np.ndarray :: normalised autocorrelations
        w      :: int :: cuttoff for the error summation - suggest w_best
        n      :: int :: number of MCMC samples
    """
    err = []
    if not isinstance(w, int): w = int(w)
    pd = np.zeros(w*5)
    pd[:acn.size] = acn
    for t in range(0,int(w)*2): # this is a horrible loop 
        tmp = 0                 # but doesn't run that slow
        for k in range(max(1,t-w), t+w):
            tmp += (pd[k+t] + pd[abs(k-t)] - 2*pd[t]*pd[k])**2
        err.append(np.sqrt(tmp/n))
    return np.asarray(err)
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
    itau_aav  = np.cumsum(acorrn) - .5      # Eq. (41)
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
    tau_w = s_tau / np.log(1. + 1./g_int.clip(0))
    g_w = np.exp(-t/tau_w) - tau_w/np.sqrt(t*n)
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
    g_int = np.cumsum(acorrn[1:t_max])
    
    # see http://stackoverflow.com/a/8534381/4013571
    try:
        w = next(t for t,v in enumerate(g_int,1) if gW(t, v, s_tau, n) < 0)
    except:
        checks.tryAssertNotEqual(False, False,
        'Windowing condition failed up to W = {}'.format(g_int.size))
        # UWerr actually returns min(t_max, 2*t) anyway
    return w
#
def uWerr(f_ret, s_tau=1.5):
    """autocorrelation-analysis of MC time-series following the Gamma-method
    This (simplified) implementation assumes f_ret have been acted upon by an operator
    and just completes basic calculations
    
    **Currently only supports ONE observable where # samples == f_ret.shape[0]**
    
    Required Inputs
        f_ret   :: np.ndarray :: the return of a function action upon all f_ret
    
    Optional Inputs
        s_tau   :: float>0 :: guess for the ratio S of tau/tauint [D=1.5]
    
    Notation notes:
        x_av0 is an average of x over the 0th dim - \bar{x}^r in the paper
        x_aav is the average over all dims        - \bbar{x} in the paper
        v_err is the error in v                   - \sigma_F in the paper
        v_eerr is the error of the error          - given by Eq. 40 : no symbol in paper
    
    s_tau or Stau in the original code doesn't work for the 0 case so I removed it.
    see https://github.com/flipdazed/Hybrid-Monte-Carlo/issues/34#issuecomment-232472657
    for info.
    """
    checks.tryAssertNotEqual(s_tau, 0,
        's_tau cannot be zero, see:\n' \
        + 'https://github.com/flipdazed/Hybrid-Monte-Carlo' \
        + '/issues/34#issuecomment-232472657')
    
    if not isinstance(f_ret, np.ndarray): f_ret = np.ndarray(f_ret)
    checks.tryAssertEqual(len(f_ret.shape[1:]), len(set(f_ret.shape[1:])),
        'Only expects cuboid lattices: dims >2 are not equal.' \
        + '\nShape: {}'.format(f_ret.shape))
    
    f_aav = np.average(f_ret)           # get the mean of the function outputs
    n = float(f_ret.shape[0])           # number of MCMC samples
    
    # get autocorrelations: Don't normalise until bias corrected
    fn = lambda t: getAcorr(op_samples=f_ret, mean=f_aav, separation=t, norm=None)
    acorr  = np.asarray([fn(t=t) for t in range(0, int(n//2))]) # t_max implicit n//2
    
    norm = acorr[0] # values for w = 0
    checks.tryAssertNotEqual(norm, 0,
        'Normalisation cannot be zero: No fluctuations.' \
        + '\nNormalisation: {}'.format(norm))
    
    # The automatic windowing proceedure
    w = autoWindow(acorrn=acorr/norm, s_tau=s_tau, n=n)
    
    # correct acorr for variance in the function
    c_aav  = covarianceN(acorr=acorr, window=w, var=norm) # var = c_aav/n     Eq. 35
    acorr += c_aav/n                                        # correct for bias  Eq. 32,49
    norm   = acorr[0]
    c_aav  = covarianceN(acorr=acorr, window=w, var=norm)
    acorrn = acorr/norm                                     # normalise corrected a/c fn.
    
    itau, itau_diff, itau_aav = intAcorr(acorrn=acorrn, n=n, window=w)
    f_diff  = np.sqrt(c_aav/n)             # error of f from       Eq. (26,44)
    f_ddiff = f_diff*np.sqrt((w + .5)/n)   # error on error of f   Eq. (42)
    
    # return relevant values - perhaps this is all better as a class?
    return f_aav, f_diff, f_ddiff, itau, itau_diff, itau_aav[:2*w+1], acorr[:2*w+1]

if __name__ == '__main__':
    pass