# -*- coding: utf-8 -*- 
from __future__ import division
import numpy as np
import itertools
from scipy.special import binom
from tqdm import tqdm
from copy import copy

from hmc import checks
from hmc.common import Init
from .common import Base

def correlatedData(tau = 5, n = 10000):
    r"""A sequence of n normally distributed r.v. unit variance
    and vanishing mean. From this construct:
    $$
      \nu_1 = \eta_1,\quad \nu_{i+1} = \sqrt{1 - a^2} \eta_{i+1} + a
      \nu_i\,,\\
      a = \frac {2 \tau - 1}{2\tau + 1}, \quad \tau \geq \frac 1 2\,,
    $$
    where $\tau$ is the autocorrelation time
    
    Adapted from:
    https://github.com/dhesse/py-uwerr/blob/master/puwr.py
    """
    nu = np.random.rand(n)
    e = nu.copy()
    a  = (2. * tau - 1.)/(2. * tau + 1.)
    nu[1:]    *= np.sqrt(1. - a**2)
    for i in range(1, n):
        nu[i] += a * nu[i-1]
    return nu*.2 + 1.
    
def correlated_data(tau = 5, n = 10000):
    r"""Generate correlated data as explained in the appendix of
    [1]_. One draws a sequence of :math:`n` normally distributed
    random numbers :math:`\eta_i, i = 1,\ldots,n` with unit variance
    and vanishing mean. From this one constructs
    .. math::
      \nu_1 = \eta_1,\quad \nu_{i+1} = \sqrt{1 - a^2} \eta_{i+1} + a
      \nu_i\,,\\
      a = \frac {2 \tau - 1}{2\tau + 1}, \quad \tau \geq \frac 1 2\,,
    where :math:`\tau` is the autocorrelation time::
      >>> from puwr import correlated_data
      >>> correlated_data(2, 10)
      [[array([ 1.02833043,  1.08615234,  1.16421776,  1.15975754,
                1.23046603,  1.13941114,  1.1485227 ,  1.13464388,
                1.12461557,  1.15413354])]]
    :param tau: Target autocorrelation time.
    :param n: Number of data points to generate.
    """
    eta = np.random.rand(n)
    a = (2. * tau - 1)/(2. * tau + 1)
    asq = a**2
    nu = np.zeros(n)
    nu[0] = eta[0]
    for i in range(1, n):
        nu[i] = np.sqrt(1 - asq)*eta[i] + a * nu[i-1]
    return [[nu*0.2 + 1]]

def acorrMapped(op_samples, sep_map, sep, mean, norm = 1.0, tol=1e-7, counts=False):
    """as acorr() except this function maps the correlation to a non homogenous
    separation mapping from t=0 that is provided by `sep_map`
    
    To be explicit, sep_map is a non decreasing array that maps distance from
    t=0 such that we expect the first point to be non-zero (zero sep only possible)
    for probabilistic separations
    
    sep_map will be used to determine which samples can be included
    for averaging at a given separation length. It is assumed that all
    lattice points for a given sample, op_samples[i], have the same separation 
    with respect to op_samples[j] i.e. the lattice moves as a whole in HMC
    sample space
    
    This is equivalent to looking at every unique pair of indices in sep_map
    and assessing:
        
        if sep_map[i]-sep_map[j] == separation:
            # include in autocorrelation calculation
    
    should no values exist at a given separation the routine returns np.nan
    
    Required Inputs
        op_samples  :: np.ndarray :: the operator samples
        sep_map     :: np.ndarray :: the separation mapping to op_samples
        sep         :: int :: the separation between HMC steps
        mean        :: float :: the mean of the operator
    
    Optional Inputs
        norm        :: float :: the autocorrelation with separation=0
        tol         :: float :: tolerance around zero (numpy errors)
        counts      :: bool  :: return counts in a tuple
    
    the standard deviation formula is:
        sum( (x-m)^2 )
    which is equiv to:
        sum( x^2 - 2mx + m^2)
    we cannot know the mean of the measured data until the end of the
    while loop so by summing the squared values and calculating the mean
    in the loop we can do the full sdev calculation at the end!
    """
    n = op_samples.shape[0]
    # note that in HMC we don't have any repeated elements so separations 0 
    # can only be the array on itself
    if sep == 0: 
        result = acorr(op_samples, mean, separation=0, norm = norm)
        if counts: result = (result, n)
        return result
    
    front = 1   # front "pythony-pointer-thing"
    back  = 0   # back "pythony-pointer-thing"
    bssp  = 0   # back sweep start point
    bsfp  = 0   # back sweep finish point
    ans   = 0.0 # the answer
    count = 0   # counter for averaging
    new_front = True # the first front value is new
    while front < n:            # keep going until exhausted sep_mapay
        new_front = abs(sep_map[front]-sep_map[front-1]) > tol  # check if front value is a new one
        back = bsfp if new_front else bssp         # this is the magical step
        
        if abs(sep_map[front] - sep_map[back] - sep) < tol: # if equal subject to tol: pair found
            if new_front:
                bssp  = bsfp    # move sweep start point
                back  = bsfp    # and back to last front point
                bsfp  = front   # send start end point to front's position
            else:
                back  = bssp    # reset back to the sweep start point
            while back < bsfp:  # calculate the correlation function for matched pairs
                count+= 1
                ans  += ((op_samples[front] - mean)*(op_samples[back] - mean)).mean()
                back += 1
        else:   # check if there is a new back
            if abs(sep_map[bssp+1] - sep_map[bssp]) > tol: bsfp = front
        
        front +=1
    result = ans/(count*norm) if count > 0.0 else np.nan # cannot calculate if no pairs
    if counts: result = (result, count)
    return result

def acorrMapped_noDups(op_samples, sep_map, sep, mean, norm = 1.0, tol=1e-7, counts=False):
    """NOT TESTED AND DOESN'T WORK AS INTENDED!
    
    The idea was to write a faster algorithm that uses
    
    Required Inputs
        op_samples  :: np.ndarray :: the operator samples
        sep_map     :: np.ndarray :: the separation mapping to op_samples
        sep         :: int :: the separation between HMC steps
        mean        :: float :: the mean of the operator
    
    Optional Inputs
        norm        :: float :: the autocorrelation with separation=0
        tol         :: float :: tolerance around zero (numpy errors)
        counts      :: bool  :: return counts in a tuple
    
    the standard deviation formula is:
        sum( (x-m)^2 )
    which is equiv to:
        sum( x^2 - 2mx + m^2)
    we cannot know the mean of the measured data until the end of the
    while loop so by summing the squared values and calculating the mean
    in the loop we can do the full sdev calculation at the end!
    """
    raise NotImplemented("Waiting on a good method to index np.ndarray")
    n = op_samples.shape[0]
    # note that in HMC we don't have any repeated elements so separations 0 
    # can only be the array on itself
    if sep == 0: 
        result = acorr(op_samples, mean, separation=0, norm = norm)
        if counts: result = (result, n)
        return result
    
    # see attempt in acorrMapped_noDups.c for inner workings
    # acorrMapped_noDup (double* op_samples,double* sep_map,
    #     double* ans, int* count, int n, double tol, double sep)
    
    result = ans/(count*norm) if count > 0 else np.nan # cannot calculate if no pairs
    if counts: result = (result, count)
    return result

def acorr(op_samples, mean, separation, norm = 1.0):
    """autocorrelation of a measured operator with optional normalisation
    the autocorrelation is measured over the 0th axis
    
    Required Inputs
        op_samples  :: np.ndarray :: the operator samples
        mean        :: float :: the mean of the operator
        separation  :: int :: the separation between HMC steps
        norm        :: float :: the autocorrelation with separation=0
    
    Note: the sample index will always be the first index. A sample is one op_sample
    in the following e.g. op_sample[i] is the i'th sample
    
    There can be two cases which are both valid inputs:
        1. Each sample contains un-averaged operator measurements at each lattice site
        2. Each sample contains averaged op measurements
    
    1. each sample will be an n-dim array***
    2. each sample will be a scalar from averaging across all sites
    
    This func handles both by 'ravelling' as:
        acorr = acorrs.ravel().mean()
    as the mean of means is the same as the overall mean
    """
    if separation == 0: return ((op_samples-mean)**2).ravel().mean() / norm
    return ((op_samples[:op_samples.size-separation]-mean)*(op_samples[separation:]-mean)).ravel().mean() / norm

class Autocorrelations_1d(Init, Base):
    """Runs a model and calculates the 1 dimensional autocorrelation function
    
    Required Inputs
        model           :: class    :: a class that runs some sort of model - see models.py
        attr_samples    :: str      :: attr location of the x sampeles: model.attr_samples
        attr_run        :: str      :: attr points to fn to run the model: model.attr_run()
    
    Expectations
        model has an attribute that contains the MCMC samples following a run
        model has a function that runs the MCMC sampling
        the samples are stored as [sample index, sampled lattice configuration]
    """
    def __init__(self, model, **kwargs):
        super(Autocorrelations_1d, self).__init__()
        self.initArgs(locals())
        self.defaults = {
            'attr_run':'run',
            'attr_samples':'samples',
            'attr_trajs':'traj'
            }
        self.initDefaults(kwargs)
        self._setUp()
        pass
    
    def getAcorr(self, separations, op_func, norm = False, prll_map=None, proc_max=1000, max_sep=None):
        """Returns the autocorrelations for a specific sampled operator 
        
        Once the model is run then the samples can be passed through the autocorrelation
        function in once move utilising the underlying optimisations of numpy in full
        
        Required Inputs
            separations  :: iterable, int :: the separations between HMC steps
            op_func      :: func :: the operator function
        
        Optional Inputs
            norm     :: bool :: specifiy whether to normalise the autocorrelations
            prll_map :: fn :: the prll_map from results.common.utils for multicore usage
            proc_max :: int :: max number of processes to spawn
            max_sep :: float :: define the max separation
        
        Notes: op_func must be optimised to only take ALL HMC trajectories as an input
        """
        
        if not isinstance(separations, list): separations = list(separations)
        checks.tryAssertEqual(True, all(isinstance(s, int) for s in separations),
            "Separations must be list of integers:\n{}".format(separations))
        
        if not hasattr(self, 'op_samples'): 
            if not hasattr(self, 'samples'): self._getSamples() # get samples if not already
            if not isinstance(self.samples, np.ndarray): self.samples = np.asarray(self.samples)
            if not isinstance(self.trajs, np.ndarray): self.trajs = np.asarray(self.trajs)
            self.op_samples = op_func(self.samples)
        
        # get mean for these samples if doesn't already exist - don't waste time doing multiple
        if not hasattr(self, 'op_mean'): self.op_mean = self.op_samples.ravel().mean()
        
        # This section flips between random and fixed trajectories
        
        # get normalised autocorrelations for each separation
        if norm:
            if not hasattr(self, 'op_norm'):
                self.op_norm = acFn(separation=0)
        else:
            self.op_norm = 1.0
        
        if self.model.rand_steps:
            cumut       = np.cumsum(self.trajs)
            if max_sep is None:
                max_sep     = self.model.step_size*len(separations)/2.
            n_seps      = max_sep/self.model.step_size + 1
            separations = np.linspace(0, max_sep, n_seps)
            tolerance   = self.model.step_size/2.-self.model.step_size*0.1
            
            # http://stackoverflow.com/q/5900985/4013571
            # ensures that do not hit limit of processes
            sep_skip    = max(1, int(n_seps // proc_max))
            separations = separations[::sep_skip]
            
            acFn = lambda separation: acorrMapped(self.op_samples, cumut, separation, self.op_mean, 
            norm=self.op_norm, tol=tolerance, counts=True)
        else:
            acFn = lambda separation: acorr(self.op_samples, mean=self.op_mean, 
                separation=separation, norm=self.op_norm)
        
        if prll_map is not None:
            result = prll_map(acFn, separations, verbose=True)
            if self.model.rand_steps: acs, counts = zip(*result)
            else: acs, counts = result, None
        else:
            result = [acFn(separation=s) for s in separations]
            if self.model.rand_steps: acs, counts = zip(*result)
            else: acs, counts = result, None
        
        self.acorr_ficticous_time = separations
        self.acorr_counts = np.asarray(counts, dtype='float64')
        self.acorr = np.asarray(acs)
        return self.acorr
        
    