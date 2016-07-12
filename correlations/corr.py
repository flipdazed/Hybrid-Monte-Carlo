# -*- coding: utf-8 -*- 
import numpy as np
from hmc import checks

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

def twoPointTheoryQHO(spacing, mu, length, separation=0.):
    """Theoretical prediction for the 1D 2-point correlation function
    
    Required Inputs
      spacing  :: float :: lattice spacing
      mu       :: float :: mass-coupling
      length   :: int   :: length of 1D lattice
      separation :: int :: number of sites between observables <x(0)x(separation)>
    """
    amu = mu*spacing
    if np.abs(amu) <= 0.1: 
        r = 1. - amu + .5*amu**2
        print '> approximating R = 1. - aµ + aµ**2/2 as |aµ| <= 0.1 '
    else:
        r = 1. + .5*amu**2 - amu*np.sqrt(1. + .25*amu**2)
    if separation == 0:
        ratio = (1. + r**length)/(1. - r**length)
    else:
        ratio = (r**separation + r**(length-separation))/(1. - r**length)
    
    av_xx = ratio / (2.*mu*np.sqrt(1. + .25*amu**2))
    return av_xx

class Correlations_1d(object):
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
    def __init__(self, model, attr_run, attr_samples, *args, **kwargs):
        self.model = model
        self.attr_run = attr_run
        self.attr_samples = attr_samples
        
        checks.tryAssertEqual(True, hasattr(self.model, self.attr_run),
            "The model has no attribute: self.model.{} ".format(self.attr_run))
        
        # set all kwargs and args
        for kwarg,val in kwargs.iteritems(): setattr(self, kwarg, val)
        for arg in args: setattr(self, arg, arg)
        
        # run the model
        self.runWrapper = getattr(self.model, self.attr_run)
        pass
    
    def runModel(self, *args, **kwargs):
        """Runs the model with any given *args and **kwargs"""
        self.result = self.runWrapper(*args, **kwargs)
        return self.result
    
    def getTwoPoint(self, separation):
        """Delivers the two point with a given separation and runs model if necessary
        
        Once the model is run then the samples can be passed through the correlation
        function in once move utilising the underlying optimisations of numpy in full
        
        Required Inputs
            separation :: int :: the lattice spacing between the two point function
        """
        
        checks.tryAssertEqual(int, type(separation),
            "Separation must be integer: type is: {}".format(type(separation)))
        
        if not hasattr(self, 'samples'): self._getSamples()
        
        # here the sample index will be the first index
        # the remaning indices are the lattice indices
        samples = np.asarray(samples)
        
        self.two_point = twoPoint(self.samples, separation)
        
        # ravel() just flattens averything into one dimension
        # rather than averaging over each lattice sample and averaging over
        # all samples I jsut average the lot saving a calculation
        return self.two_point.ravel().mean()
        
    def _getSamples(self):
        """grabs the position samples from the model"""
        
        checks.tryAssertEqual(True, hasattr(self, 'result'),
            "The model has not been run yet!\n\tself.result not found")
        checks.tryAssertEqual(True, hasattr(self.model, self.attr_samples),
            "The model has no lattice attribute: self.model.{} ".format(self.attr_samples))
        
        samples = getattr(self.model, self.attr_samples) # pull out the lattice values
        
        # check that the lattice is indeed 1D as the size should be equal to
        # the largest dimension. Won't be true if more than 1 entry != (1,)
        checks.tryAssertEqual(samples[0].size, max(samples[0].shape),
            "The lattice is not 1-dimensional: self.model.{}.shape ={}".format(
            self.attr_samples, samples[0].shape))
        self.samples = samples[1:] # the last burn-in sample is the 1st entry
        pass
