# -*- coding: utf-8 -*- 
import numpy as np

from hmc import checks
from hmc.common import Init
from .common import Base

def acorr(op_samples, mean, separation, weights = None, norm = None):
    """autocorrelation of a measured operator with optional normalisation
    
    Required Inputs
        op_samples     :: np.ndarray :: the operator samples from a number of lattices
        mean        :: float :: the mean of the operator
        separation  :: int :: the separation between HMC steps
        weights     :: np.ndarray :: the weights for each respective sample
        norm        :: float :: the autocorrelation with separation=0
    
    Note: the sample index will always be the first index. A sample is one op_sample
    in the following e.g. op_sample[i] is the i'th sample
    
    There can be two cases which are both valid inputs:
        1. Each sample contains un-averaged operator measurements at each lattice site
        2. Each sample contains averaged op measurements
    
    1. each sample will be an n-dim array
    2. each sample will be a scalar from averaging across all sites
    
    This func handles both by 'ravelling' as:
        acorr = acorrs.ravel().mean()
    as the mean of means is the same as the overall mean
    """
    
    if type(op_samples) != np.ndarray: op_samples = np.asarray(op_samples)
    # shift the array by the separation
    # need the axis=0 as this is the sample index, 1 is the lattice index
    # corellations use axis 1
    shifted = np.roll(op_samples, separation, axis = 0)
    
    # Need to be wary that roll will roll all elements arouns the array boundary
    # so cannot take element from the end. The last sample that can have an autocorrelation
    # of length m within N samples is x[N-1-m]x[N-1] so we want up to index N-m
    n = op_samples.shape[0] # get the number of samples from the 0th dimension
    acorrs = ((shifted - mean)*(op_samples - mean))[:n-separation] # indexing the 1st index for samples
    
    # normalise if a normalisation is given
    if norm is not None: acorrs /= norm
    
    # ravel() just flattens averything into one dimension
    # rather than averaging over each lattice sample and averaging over
    # all samples I jsut average the lot saving a calculation
    if weights is not None: weights = weights[:n-separation]
    acorr = np.average(acorrs, weights=weights, axis = 0).mean()
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
    
    def getAcorr(self, separations, op_func):
        """Returns the autocorrelations for a specific sampled operator 
        
        Once the model is run then the samples can be passed through the autocorrelation
        function in once move utilising the underlying optimisations of numpy in full
        
        Required Inputs
            separations  :: iterable, int :: the separations between HMC steps
            op_func     :: func :: the operator function
        
        
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
        if not hasattr(self, 'op_norm'):
            self.op_norm = acorr(self.op_samples, self.op_mean, separation=0)
        
        # get normalised autocorrelations for each separation
        separations.sort()
        if separations[0] == 0:
            self.acorr = [1.] # first entry is the normalisation
            separations = separations[1:]
        
        w = self.trajs/self.trajs.sum()
        
        checks.tryAssertEqual(self.op_samples.shape[0], w.shape[0],
            "Shapes differ! Op Samples: {}, weights: {}".format(self.op_samples.shape, w.shape))
        
        self.acorr += [acorr(self.op_samples, self.op_mean, 
                            s, w, self.op_norm) for s in separations]
        
        self.acorr = np.asarray(self.acorr)
        
        return self.acorr
        
    