class Init(object):
    """Provides initialisation helpers"""
    def __init__(self):
        pass
    
    def initDefaults(self, kwargs):
        """Assigns kwargs to self and relies on self.defaults if none are found
        
        Required Inputs
            kwargs :: dict :: this is a dict of kwargs to add to self.
        
        Expectations
            has an attribute of self.defaults to check against
        """
        for k,v in self.defaults.iteritems():
            if k in kwargs: # use assigned values
                setattr(self, k, kwargs[k])
            else: # use default values
                setattr(self, k, self.defaults[k])
        pass
    
    def initArgs(self, args):
        """Adds arguments to self
        
        Required Inputs
            args :: dict :: this is a dict of args to add to self.
        """
        ignore = ['self', 'kwargs', 'args']
        for k,v in args.iteritems():
            if k not in ignore: 
                setattr(self, k, v)
        pass