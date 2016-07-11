# -*- coding: utf-8 -*- 
import numpy as np
from hmc import checks

class Autocorrelations_1d(object):
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
    
    def getAcorr(self, separation):
        """Delivers the two point with a given separation
        
        Once the model is run then the samples can be passed through the correlation
        function in once move utilising the underlying optimisations of numpy in full
        
        Equivalent to evaluating for each sample[i] and averaging over them
        
        Required Inputs
            separation :: int :: the lattice spacing between the two point function
        """
        
        checks.tryAssertEqual(int, type(separation),
            "Separation must be integer: type is: {}".format(type(separation)))
        
        if not hasattr(self, 'samples'): self._getSamples()
        
        # here the sample index will be the first index
        # the remaning indices are the lattice indices
        self.samples = np.asarray(self.samples)
        
        # shift the array by the separation
        # need the axis=0 as this is the sample index, 1 is the lattice index
        # corellations use axis 1
        shifted = np.roll(self.samples, separation, axis=0)
        
        # Need to be wary that roll will roll all elements arouns the array boundary
        # so cannot take element from the end. The last sample that can have an autocorrelation
        # of length m within N samples is x[N-1-m]x[N-1] so we want up to index N-m
        n = self.samples.shape[0] # get the number of samples
        self.acorr = (shifted*self.samples)[:n-separation] # indexing the 1st index for samples
        
        # ravel() just flattens averything into one dimension
        # rather than averaging over each lattice sample and averaging over
        # all samples I jsut average the lot saving a calculation
        return self.acorr.ravel().mean()
        
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
