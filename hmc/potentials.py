from __future__ import division
import numpy as np

from lattice import Periodic_Lattice, laplacian, gradSquared
from scipy import ndimage
from scipy.ndimage import _nd_image
import checks

__all__ = [ 'Klein_Gordon',
            'Quantum_Harmonic_Oscillator',
            'Simple_Harmonic_Oscillator',
            'Multivariate_Gaussian']

laplace_filter = np.asarray([1, -2, 1], dtype=np.float64)
def fastLaplaceNd(arr):
    """A very fast laplace filter for small arrays directly calling the scipy c++ fucntion
    
    Required Inputs
        arr :: nd.array :: the array to calculate the n-dim laplace filter
    """
    output = np.zeros(arr.shape, 'float64')
    if arr.ndim > 0:
        # send output as a pointer sio no need for equals sign
        _nd_image.correlate1d(arr, laplace_filter, 0, output, 1, 0.0, 0)
        if arr.ndim == 1: return output
        for ax in xrange(1, arr.ndim):
            output += _nd_image.correlate1d(arr, laplace_filter, ax, output, 1, 0.0, 0)
    return output

class Shared(object):
    """Shared methods"""
    def __init__(self):
        self.all = [self.kE, self.uE, self.duE]
        self.kE.__name__  = 'Conjugate HMC Kinetic Energy'
        self.uE.__name__  = 'Action (HMC Potential)'
        self.duE.__name__ = 'Gradient of Action (HMC Potential)'
        pass
    
    def _lattice(self):
        self.kE  = lambda p, *args, **kwargs: self.kineticEnergy(p=p)
        self.uE  = lambda x, *args, **kwargs: self.potentialEnergy(positions=x)
        self.duE = lambda x, *args, **kwargs: self.gradPotentialEnergy(positions=x)
        pass
    
    def _nonLattice(self):
        self.kE = lambda p, *args, **kwargs: self.kineticEnergy(p=p)
        self.uE = lambda x, *args, **kwargs: self.potentialEnergy(x=x)
        self.duE = lambda x, *args, **kwargs: self.gradPotentialEnergy(x=x)
        pass
    
    def hamiltonian(self, p, x):
        """Returns the Hamiltonian
        
        Required Inputs
            p :: np.array (nd) :: momentum array
            x :: class :: see lattice.py for info
        """
        if not hasattr(self, 'debug'): self.debug = False
        if self.debug:
            h = self.kE(p) + self.uE(x)[0]
        else:
            h = self.kE(p) + self.uE(x)
        
        # check 1 dimensional
        checks.tryAssertEqual(h.shape, (1,)*len(h.shape),
             ' hamiltonian() not scalar.\n> shape: {}'.format(h.shape))
        return h.reshape(1)
    
#
class Klein_Gordon(Shared):
    """Klein Gordon Potential on a lattice
    
    H = \frac{m}{2}\dot{x}^2 + V(x)
    
    V(x) = \frac{1}{2}mx^2 + \frac{1}{3!}\lambda_3 x^3 + \frac{1}{4!}\lambda_4 x^4
    
    Optional Inputs
        m       :: float :: mass
        phi_3   :: phi_3 coupling constant
        phi_4   :: phi_4 coupling constant
    """
    def __init__(self, m=1., phi_3=0., phi_4=0., debug=False):
        self.name = 'Klein-Gordon'
        self.debug = debug
        self.m = m
        self.phi_3 = phi_3      # phi^3 coupling const.
        self.phi_4 = phi_4      # phi^4 coupling const.
        
        if self.phi_3 == self.phi_4 == 0 and self.debug == False:
            self.potentialEnergy = self.potentialEnergyBare
            self.gradPotentialEnergy = self.gradPotentialEnergyBare
        else: 
            self.potentialEnergy = self.potentialEnergyint
            self.gradPotentialEnergy = self.gradPotentialEnergyInt
        
        super(Klein_Gordon, self)._lattice()
        super(Klein_Gordon, self).__init__()
        pass
    
    def kineticEnergy(self, p):
        """n-dim KE
        
        This is the kinetic energy for the shadow hamiltonian
        
        Required Inputs
            p :: np.array (nd) :: momentum array
        """
        return .5 * np.sum(p**2, axis=None)
    
    def potentialEnergyBare(self, positions):
        """n-dim potential without interactions for speed
        
        See potentialEnergyInt for help docs
        
        Required Inputs
            positions :: class :: see lattice.py for info
        """
        p_sq = fastLaplaceNd(positions) / float(positions.lattice_spacing)
        
        # multiply the potential by the positions spacing as required
        return .5 * (-np.sum(positions * p_sq) + positions.lattice_spacing * self.m**2 * np.sum(positions**2))
    def gradPotentialEnergyBare(self, positions):
        """Gradient of the action with interactions
        
        See gradPotentialEnergyInt for help docs
        
        Required Inputs
            positions :: class :: see lattice.py for info
        """
        return -fastLaplaceNd(positions) / positions.lattice_spacing + positions.lattice_spacing * self.m * positions
    def potentialEnergyInt(self, positions):
        """n-dim potential with interactions
        
        This is the action. In HMC, the action
        is the potential in the shadow hamiltonian.
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            positions :: class :: see lattice.py for info
        """
        x_sq_sum = (positions**2).sum()
        
        # p_sq_sum = np.array(0.)
        # sum (integrate) across euclidean-space (i.e. all positions sites)
        # p_sq = ndimage.filters.laplace(positions, mode='wrap') / float(positions.lattice_spacing)
        p_sq = fastLaplaceNd(positions) / float(positions.lattice_spacing)
        p_sq_sum = (positions * p_sq).sum()
        
        #### free action S_0: 1/2 \phi(m^2 - \klein_gordon)\phi 
        kinetic = - .5 * p_sq_sum
        u_0 = .5 * self.m**2 * x_sq_sum
        ### End free action
        
        # Add interation terms if required
        if self.phi_3: # phi^3 term
            x_3_sum = np.power(positions.ravel(), 3).sum()
            u_3 = self.phi_3 * x_3_sum / np.math.factorial(3)
        else:
            u_3 = 0.
        
        if self.phi_4: # phi^4 term
            x_4_sum = np.power(positions.ravel(), 4).sum()
            u_4 = self.phi_4 * x_4_sum / np.math.factorial(4)
        else:
            u_4 = 0.
        
        # the potential terms in the action
        potential = u_0 + u_3 + u_4
        
        # multiply the potential by the positions spacing as required
        euclidean_action = kinetic + positions.lattice_spacing * potential
        
        if self.debug: # allows for debugging
            ret_val = [euclidean_action, kinetic, potential*positions.positions_spacing]
        else:
            ret_val = euclidean_action
        return ret_val
    def gradPotentialEnergyInt(self, positions):
        """Gradient of the action with interactions
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            positions :: class :: see lattice.py for info
        """
        # gradient of kinetic term positions \klein_gordon^2 positions = 2 \klein_gordon^2 positions
        p_sq = fastLaplaceNd(positions) / float(positions.lattice_spacing)
        # p_sq = laplacian(positions, idpositions, a_power=1)
        
        #### grad of free action S_0: 2/2 * (m^2 - \klein_gordon^2)\phi
        kinetic = - p_sq
        u_0 = self.m * positions # derivative taken
        ### End free action
        
        # Add interation terms if required
        if self.phi_3: # phi^3 term
            u_3 = self.phi_3 * positions**2 / np.math.factorial(2)
        else:
            u_3 = 0.
        
        if self.phi_4: # phi^4 term
            u_4 = self.phi_4 * positions**3 / np.math.factorial(3)
        else:
            u_4 = 0.
        
        # the potential terms in the action
        potential = u_0 + u_3 + u_4
            
        # multiply the potential by the lattice spacing as required
        derivative = kinetic + (positions.lattice_spacing * potential)
        # print 'kinetic, {}\npot: {}\n\n'.format(kinetic, potential)
        return derivative
    
#
class Quantum_Harmonic_Oscillator(Shared):
    """Quantum Harmonic Oscillator on a lattice
    
    H = \frac{m}{2}\dot{x}^2 + V(x)
    
    V(x) = \frac{1}{2}mx^2 + \frac{1}{3!}\lambda_3 x^3 + \frac{1}{4!}\lambda_4 x^4
    
    Optional Inputs
        m0      :: float :: rest mass
        mu      :: float :: x^2 coupling
        phi_3   :: phi_3 coupling constant
        phi_4   :: phi_4 coupling constant
    """
    def __init__(self, m0=1., mu=1., phi_3=0., phi_4=0., debug=False):
        self.name = 'QHO'
        self.debug = debug
        self.m0 = m0
        self.mu = mu
        self.phi_3 = phi_3      # phi^3 coupling const.
        self.phi_4 = phi_4      # phi^4 coupling const.
        
        super(Quantum_Harmonic_Oscillator, self)._lattice()
        super(Quantum_Harmonic_Oscillator, self).__init__()
        pass
    
    def kineticEnergy(self, p):
        """n-dim KE
        
        This is the kinetic energy for the shadow hamiltonian
        
        Required Inputs
            p :: np.array (nd) :: momentum array
        """
        return .5 * np.square(p).ravel().sum(axis=0)
    
    def potentialEnergy(self, positions):
        """n-dim potential
        
        This is the action. In HMC, the action
        is the potential in the shadow hamiltonian.
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            positions :: class :: see lattice.py for info
        """
        lattice = positions # shortcut for brevity
        
        x_sq_sum = (lattice**2).ravel().sum()
        
        v_sq_sum = np.array(0.) # initiate velocity squared
        # sum (integrate) across euclidean-space (i.e. all lattice sites)
        for idx in np.ndindex(lattice.shape):
            
            # sum velocity squared
            v_sq = gradSquared(positions, idx, a_power=1)
            
            # gradient should be an array of the length of degrees of freedom 
            checks.tryAssertEqual(v_sq.shape, (),
                 ' derivative^2 shape should be scalar' \
                 + '\n> v_sq shape: {}'.format(v_sq_sum.shape)
                 )
                 
            # sum to previous
            v_sq_sum +=  v_sq
        
        #### free action S_0: m/2 \phi(v^2 + m)\phi
        kinetic = .5 * self.m0 * v_sq_sum
        u_0 = .5 * self.mu**2 * x_sq_sum
        ### End free action
        
        # Add interation terms if required
        if self.phi_3: # phi^3 term
            x_3_sum = np.power(lattice.ravel(), 3).sum()
            u_3 = self.phi_3 * x_3_sum / np.math.factorial(3)
        else:
            u_3 = 0.
        
        if self.phi_4: # phi^4 term
            x_4_sum = np.power(lattice.ravel(), 4).sum()
            u_4 = self.phi_4 * x_4_sum / np.math.factorial(4)
        else:
            u_4 = 0.
        
        # the potential terms in the action
        potential = u_0 + u_3 + u_4
        
        # multiply the potential by the lattice spacing as required
        euclidean_action = kinetic + positions.lattice_spacing * potential
        
        if self.debug: # alows for debugging
            ret_val = [euclidean_action, kinetic, potential*positions.lattice_spacing]
        else:
            ret_val = euclidean_action
        
        return ret_val
    
    def gradPotentialEnergy(self, positions):
        """Gradient of the action
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            idx   :: integer :: lattice position
            positions :: class :: see lattice.py for info
        """
        
        # don't want the whole lattice in here!
        # the laplacian indexs the other elements
        x = positions
        
        # derivative of velocity squared
        # the derivative of the velocity squared is actually
        # identical to - \klein_gordon^2
        a = positions.lattice_spacing
        # gradient of kinetic term x \klein_gordon^2 x = 2 \klein_gordon^2 x
        v_sq = ndimage.filters.laplace(x, mode='wrap')/float(a)
        
        # gradient should be an array of the length of degrees of freedom 
        # checks.tryAssertEqual(v_sq.shape, (),
        #      ' derivative^2 shape should be scalar' \
        #      + '\n> v_sq shape: {}'.format(v_sq.shape)
        #      )
        
        #### free action S_0: m/2 \phi(v^2 + m)\phi
        kinetic = - self.m0 * v_sq
        u_0 = self.mu**2 * x
        ### End free action
        
        # Add interation terms if required
        if self.phi_3: # phi^3 term
            u_3 = self.phi_3 * x**2 / np.math.factorial(2)
        else:
            u_3 = 0.
        
        if self.phi_4: # phi^4 term
            u_4 = self.phi_4 * x**3 / np.math.factorial(3)
        else:
            u_4 = 0.
        
        # the potential terms in the action
        potential = u_0 + u_3 + u_4
            
        # multiply the potential by the lattice spacing as required
        derivative = kinetic + (positions.lattice_spacing * potential)
            
        return derivative
    
#
class Simple_Harmonic_Oscillator(Shared):
    """Simple Harmonic Oscillator
    
    The potential is given by: F(x) = k*x
    
    Optional Inputs
        k :: float :: spring constant
    """
    def __init__(self, k=1.):
        self.name = 'SHO'
        self.k = np.asarray(k)
        
        super(Simple_Harmonic_Oscillator, self)._nonLattice()
        super(Simple_Harmonic_Oscillator, self).__init__()
        pass
    
    def kineticEnergy(self, p):
        return .5 * (p**2).sum()
    
    def potentialEnergy(self, x):
        return .5 * (x**2).sum()
    
    def gradPotentialEnergy(self, x):
        """
        Required Inputs
            x :: np.matrix :: column vector
        
        Optional Inputs
            idx :: tuple(int) :: an index for the n-dim SHO
        Notes
            discard just stores extra arguments passed for compatibility
            with the lattice versions
        """
        return self.k * x
    
#
class Multivariate_Gaussian(Shared):
    """Multivariate Gaussian Distribution
    
    The potential is given by the n-dimensional gaussian
    
    Required Inputs
        dim     :: integer >=0              :: number of dimensions
        mean    :: n-dim vector (float)     :: the mean of each dimension
        cov     :: n-dim^2 matrix (float)   :: covariance matrix
    """
    def __init__(self, mean=[[0.], [0.]], cov=[[1.,.8],[.8,1.]]):
        self.name = 'MVG'
        self.mean = np.asarray(mean)
        self.cov = np.asmatrix(cov)
        self.dim = self.mean.shape[1]
        
        self.dim_rng = np.arange(self.dim) # range of dimensions for iterating
        # assert (self.cov.T == self.cov).all() # must be symmetric
        assert (self.cov[self.dim_rng, self.dim_rng] == 1.).all() # diagonal of 1s
        
        self.cov_inv = self.cov.I # calculate inverse (save comp. time)
        
        super(Multivariate_Gaussian, self)._nonLattice()
        super(Multivariate_Gaussian, self).__init__()
        pass
    
    def kineticEnergy(self, p):
        """n-dim KE
        
        Required Inputs
            p :: np.matrix (col vector) :: momentum vector
        """
        checks.tryAssertEqual(len(p.shape), 2,
             ' expected momentum dims = 2.\n> p: {}'.format(p))
        return .5 * (p**2).sum(axis=0)
    
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
    
    def gradPotentialEnergy(self, x):
        """n-dim gradient
        
        Notes
            discard just stores extra arguments passed for compatibility
            with the lattice versions
        """
        
        checks.tryAssertEqual(len(x.shape), 2,
             ' expected position dims = 2.\n> x: {}'.format(x))
        
        # this is constant irrelevent of the index
        return np.dot(self.cov_inv, x)
#
if __name__ == '__main__':
    from lattice import Periodic_Lattice
    
    arr = Periodic_Lattice(np.random.random(100))
    
    k = Klein_Gordon()
    assert (k.potentialEnergyBare(arr) == k.potentialEnergyInt(arr)).all()
    assert (k.gradPotentialEnergyBare(arr) == k.gradPotentialEnergyInt(arr)).all()
    # %timeit k.potentialEnergyBare(arr)
    # %timeit k.potentialEnergyInt(arr)
    #
    # %timeit k.gradPotentialEnergyBare(arr)
    # %timeit k.gradPotentialEnergyInt(arr)
    pass