
class Base(object):
    """Provides Base funcs for (auto)correlations"""
    def __init__(self):
        pass
    
    def runModel(self, *args, **kwargs):
        """Runs the model with any given *args and **kwargs"""
        self.result = self.runWrapper(*args, **kwargs)
        return self.result
    
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
    
    def _setUp(self):
        checks.tryAssertEqual(True, hasattr(self.model, self.attr_run),
            "The model has no attribute: self.model.{} ".format(self.attr_run))
        
        # run the model
        self.runWrapper = getattr(self.model, self.attr_run)
        pass