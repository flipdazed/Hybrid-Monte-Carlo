import numpy as np
from common import Init

class Accept_Reject(Init):
    """Contains accept-reject routines
    
    Required Inputs
        rng :: np.random.RandomState :: random number generator
    
    Optional Inputs
        store_acceptance :: bool :: optionally store the acceptance rates
        accept_all :: Bool :: function always returns True
    """
    def __init__(self, rng, **kwargs):
        super(Accept_Reject, self).__init__()
        self.initArgs(locals())
        self.defaults = {
            'accept_all':False,
            'store_acceptance':False
            }
        
        # these are the plurals (+'s') of each paramter
        # we want to record
        self.store = ['accept_rates', 'accept_rejects',
            'delta_hs', 'h_olds', 'h_news']
        for k in self.store: self.defaults['get_'+k] = False
        
        self.initDefaults(kwargs)
        
        # legacy code to support prev versions
        if self.store_acceptance:
            for k in self.store: setattr(self, 'get_' + k, True)
        
        # set up the lists as empty
        for k in self.store: 
            if getattr(self, 'get_' + k): setattr(self, k, [])
        pass
    
    def metropolisHastings(self, h_old, h_new):
        """A M-H accept/reject test as per
        Duane, Kennedy, Pendleton (1987)
        and also used by Neal (2003)
        
        The following, 
            min(1., np.exp(-delta_h)) - self.rng.uniform() >= 0.
            (np.exp(-delta_h) - self.rng.uniform()) >= 0
        
        are equivalent to the original step:
            self.rng.uniform() < min(1., np.exp(-delta_h))
        
        The min() function need not be evaluated as both
        the resultant 1. a huge +ve number will both result
        in acceptance.
        >= is also introduced for OCD reasons.
        
        Required Inputs
            h_old :: float :: old hamiltonian
            h_new :: float :: new hamiltonian
        
        Return :: bool
            True    :: acceptance
            False   :: rejection
        """
        delta_h = h_new - h_old
        
        if self.accept_all:
            accept_reject = True
        else:
            # (self.rng.uniform() < min(1., np.exp(-delta_h))) # Neal / DKP original
            accept_reject = (np.exp(-delta_h) - self.rng.uniform()) >= 0 # faster
        
        accept_rate = min(1., np.exp(-delta_h))
        
        # stores useful values for analysis during runtime
        l = locals()
        for k in self.store:
            # if the parameter has a 'get_' set as True... we get it!
            if getattr(self, 'get_' + k): 
                # append the non plural from locals
                lk = k[:-1]
                getattr(self, k).append(l[lk])
        
        return accept_reject
