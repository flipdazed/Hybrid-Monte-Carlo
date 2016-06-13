

class Accept_Reject(object):
    """Contains accept-reject routines
    
    Required Inputs
        rng :: np.random.RandomState :: random number generator
    """
    def __init__(self, rng):
        self.rng = rng
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
        # (self.rng.uniform() < min(1., np.exp(-delta_h))) # Neal / DKP original
        return (np.exp(-delta_h) - self.rng.uniform()) >= 0 # faster

