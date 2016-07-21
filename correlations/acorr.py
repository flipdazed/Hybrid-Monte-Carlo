# -*- coding: utf-8 -*- 
import numpy as np

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

def acorr(op_samples, mean, separation, norm = 1):
    """autocorrelation of a measured operator with optional normalisation
    
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
    
    if type(op_samples) != np.ndarray: op_samples = np.asarray(op_samples)
    # shift the array by the separation
    # need the axis=0 as this is the sample index, 1 is the lattice index
    # -ve so that a[0]->a[sep] as default is sending a[0] -> a[-1]
    shifted = np.roll(op_samples, -separation, axis = 0)
    
    # Need to be wary that roll will roll all elements arouns the array boundary
    # so cannot take element from the end. The last sample that can have an acorr
    # of len m within N samples is x[N-1-m]x[N-1] so we want up to index N-m
    n = op_samples.shape[0]
    acorrs = ((shifted - mean)*(op_samples - mean))[:n-separation] # indexing the 1st index for samples
    
    # normalise if a normalisation is given
    acorrs = acorrs / norm
    
    # av over all even of random trjectories
    acorr = acorrs.ravel().mean()
    return acorr

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
    
    def getAcorr(self, separations, op_func, norm = False):
        """Returns the autocorrelations for a specific sampled operator 
        
        Once the model is run then the samples can be passed through the autocorrelation
        function in once move utilising the underlying optimisations of numpy in full
        
        Required Inputs
            separations  :: iterable, int :: the separations between HMC steps
            op_func      :: func :: the operator function
        
        Optional Inputs
            norm    :: bool :: specifiy whether to normalise the autocorrelations
        
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
        
        # get normalised autocorrelations for each separation
        separations.sort()
        
        if norm:
            if separations[0] == 0:
                self.acorr = [1.] # first entry is the normalisation
                separations = separations[1:]
            
            if not hasattr(self, 'op_norm'):
                self.op_norm = acorr(self.op_samples, self.op_mean, separation=0)
            
            self.acorr += [acorr(self.op_samples, self.op_mean, 
                            s, self.op_norm) for s in separations]
        else:
            self.acorr = [acorr(self.op_samples, self.op_mean, s) for s in separations]
        
        self.acorr = np.asarray(self.acorr)
        
        return self.acorr
        
    