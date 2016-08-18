# -*- coding: utf-8 -*- 
import numpy as np

from hmc.common import Init
from hmc import checks
from .common import Base
import theory.operators

def twoPoint(samples, separation):
    """Delivers the UNAVERAGED two point with a given separation
    
    Equivalent to evaluating for each sample[i] and averaging over them
    
    Required Inputs
        samples    :: n+1-dim lattice :: the 0th dim is `m` HMC lattice-samples
        separation :: int :: the lattice spacing between the two point function
    """
    
    # shift the array by the separation
    # need the axis=1 as this is the lattice index, 0 is the sample index
    shifted = np.roll(samples, separation, axis=1)
    
    # this actually works for any dim arrays as * op broadcasts
    two_point = (shifted*samples)
    
    return two_point

class Correlations_1d(Init, Base):
    """Runs a model and calculates the 1 dimensional correlation function
    
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
        super(Correlations_1d, self).__init__()
        self.initArgs(locals())
        self.defaults = {
            'attr_run':'run',
            'attr_samples':'samples',
            'attr_trajs':'traj'
            }
        self.initDefaults(kwargs)
        self._setUp()
        pass
    
    def getTwoPoint(self, separation):
        """Delivers the two point with a given separation and runs model if necessary
        
        Once the model is run then the samples can be passed through the correlation
        function in once move utilising the underlying optimisations of numpy in full
        
        Required Inputs
            separation :: int :: the lattice spacing between the two point function
        """
        
        checks.tryAssertEqual(int, type(separation),
            "Separation must be integer: type is: {}".format(type(separation)))
        
        # here the sample index will be the first index
        # the remaning indices are the lattice indices
        if not hasattr(self, 'samples'): self._getSamples()
        if not isinstance(self.samples, np.ndarray): self.samples = np.asarray(self.samples)
        
        self.op_samples = twoPoint(self.samples, separation)
        
        # ravel() just flattens averything into one dimension
        # rather than averaging over each lattice sample and averaging over
        # all samples I jsut average the lot saving a calculation
        return self.op_samples.ravel().mean()
    
