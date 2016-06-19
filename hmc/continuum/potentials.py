import numpy as np

from .. import checks

class Simple_Harmonic_Oscillator(object):
    """Simple Harmonic Oscillator
    
    The potential is given by: F(x) = k*x
    
    Optional Inputs
        k :: float :: spring constant
    """
    def __init__(self, k=[[1.]]):
        self.k = np.asarray(k)
        
        self.kE = lambda p: self.kineticEnergy(p)
        self.uE = lambda x: self.potentialEnergy(x)
        self.dkE = lambda p: self.gradKineticEnergy(p)
        self.duE = lambda x: self.gradPotentialEnergy(x)
        
        self.all = [self.kE, self.uE, self.dkE, self.duE]
        
        self.mean = np.asarray([[0.]]).sum(axis=0)
        self.cov = np.asarray([[1.]]).sum(axis=0)
        pass
    
    def kineticEnergy(self, p):
        return .5 * np.square(p)
    
    def potentialEnergy(self, x):
        return .5 * np.square(x)
    
    def gradKineticEnergy(self, p):
        return p
    
    def gradPotentialEnergy(self, x, *discard):
        """
        
        Required Inputs
            x :: np.matrix :: column vector
        
        Notes
            discard just stores extra arguments passed for compatibility
            with the lattice versions
        """
        return self.k * x
    
    def hamiltonian(self, p, x):
        h = np.asarray(self.kineticEnergy(p) + self.potentialEnergy(x))
        
        # check 1 dimensional
        checks.tryAssertEqual(h.shape, (1,)*len(h.shape),
             ' hamiltonian() not scalar.\n> shape: {}'.format(h.shape))
        
        return h.reshape(1)
#
class Multivariate_Gaussian(object):
    """Multivariate Gaussian Distribution
    
    The potential is given by the n-dimensional gaussian
    
    Required Inputs
        dim     :: integer >=0              :: number of dimensions
        mean    :: n-dim vector (float)     :: the mean of each dimension
        cov     :: n-dim^2 matrix (float)   :: covariance matrix
    """
    def __init__(self, mean=[[0.], [0.]], cov=[[1.,.8],[.8,1.]]):
        self.mean = np.matrix(mean) # use 1D matrix for vector to use linalg ops.
        self.cov = np.matrix(cov)
        self.dim = self.mean.shape[1]
        
        self.dim_rng = np.arange(self.dim) # range of dimensions for iterating
        # assert (self.cov.T == self.cov).all() # must be symmetric
        assert (self.cov[self.dim_rng, self.dim_rng] == 1.).all() # diagonal of 1s
        
        self.cov_inv = self.cov.I # calculate inverse (save comp. time)
        
        self.kE = lambda p: self.kineticEnergy(p)
        self.uE = lambda x: self.potentialEnergy(x)
        self.dkE = lambda p: self.gradKineticEnergy(p)
        self.duE = lambda x: self.gradPotentialEnergy(x)
        
        self.all = [self.kE, self.uE, self.dkE, self.duE]
        pass
    
    def kineticEnergy(self, p):
        """n-dim KE
        
        Required Inputs
            p :: np.matrix (col vector) :: momentum vector
        """
        checks.tryAssertEqual(len(p.shape), 2,
             ' expected momentum dims = 2.\n> p: {}'.format(p))
        return .5 * np.square(p).sum(axis=0)
    
    def potentialEnergy(self, x):
        """n-dim potential
        
        Required Inputs
            x :: np.matrix (col vector) :: position vector
        """
        checks.tryAssertEqual(x.shape, self.mean.shape,
            ' expected x.shape = self.mean.shape\n> x: {}, mu: {}'.format(
            x.shape, self.mean.shape))
        x -= self.mean
        return .5 * ( np.dot(x.T, self.cov_inv) * x).sum(axis=0)
    
    def gradKineticEnergy(self, p):
        """n-dim Kinetic Energy"""
        
        checks.tryAssertEqual(len(p.shape), 2,
             ' expected momentum dims = 2.\n> x: {}'.format(x))
        
        return p
    
    def gradPotentialEnergy(self, x, *discard):
        """n-dim gradient
        
        Notes
            discard just stores extra arguments passed for compatibility
            with the lattice versions
        """
        
        checks.tryAssertEqual(len(x.shape), 2,
             ' expected position dims = 2.\n> x: {}'.format(x))
        
        return np.dot(self.cov_inv, x)
    def hamiltonian(self, p, x):
        h = self.kineticEnergy(p) + self.potentialEnergy(x)
        
        # check 1 dimensional
        checks.tryAssertEqual(h.shape, (1,)*len(h.shape),
             ' hamiltonian() not scalar.\n> shape: {}'.format(h.shape))
        
        return h.reshape(1)
#
if __name__ == '__main__':
    pass