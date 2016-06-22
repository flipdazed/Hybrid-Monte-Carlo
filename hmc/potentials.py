import numpy as np

from . import checks

class Klein_Gordon(object):
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
        
        self.kE  = lambda p: self.kineticEnergy(p)
        self.uE  = lambda x: self.potentialEnergy(x)
        self.dkE = lambda p: self.gradKineticEnergy(p)
        self.duE = lambda i, x: self.gradPotentialEnergy(i, x)
        
        self.all = [self.kE, self.uE, self.dkE, self.duE]
        pass
    
    def kineticEnergy(self, p):
        """n-dim KE
        
        This is the kinetic energy for the shadow hamiltonian
        
        Required Inputs
            p :: np.array (nd) :: momentum array
        """
        return .5 * np.square(p).flatten().sum(axis=0)
    
    def potentialEnergy(self, positions):
        """n-dim potential
        
        This is the action. In HMC, the action
        is the potential in the shadow hamiltonian.
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            positions :: class :: see lattice.py for info
        """
        lattice = positions.get # shortcut for brevity
        
        x_sq_sum = np.power(lattice.flatten(), 2).sum()
        
        p_sq_sum = np.array(0.)
        # sum (integrate) across euclidean-space (i.e. all lattice sites)
        for idx in np.ndindex(lattice.shape):
            
            x = lattice[idx] # iterates single points of the lattice
            
            # kinetic term: - x * (Lattice laplacian of x) * lattice spacing
            p_sq = positions.laplacian(idx, a_power=1) 
            
            # gradient should be an array of the length of degrees of freedom 
            checks.tryAssertEqual(p_sq.shape, (),
                 ' laplacian shape should be scalar' \
                 + '\n> p_sq shape: {}'.format(p_sq.shape))
            
            p_sq_sum += np.dot(x.T, p_sq)
            
            # x.p_sq is a scalar
            checks.tryAssertEqual(p_sq_sum.shape, (),
                 'p_sq * x should be scalar.' \
                 + '\n> p_sq: {} \n> x: {}'.format(p_sq, x.T)
                 + '\n> p_sq_sum {}'.format(p_sq_sum)
                 )
        
        #### free action S_0: 1/2 \phi(m^2 - \klein_gordon)\phi 
        kinetic = - .5 * p_sq_sum
        u_0 = .5 * self.m**2 * x_sq_sum
        ### End free action
        
        # Add interation terms if required
        if self.phi_3: # phi^3 term
            x_3_sum = np.power(lattice.flatten(), 3).sum()
            u_3 = self.phi_3 * x_3_sum / np.math.factorial(3)
        else:
            u_3 = 0.
        
        if self.phi_4: # phi^4 term
            x_4_sum = np.power(lattice.flatten(), 4).sum()
            u_4 = self.phi_4 * x_4_sum / np.math.factorial(4)
        else:
            u_4 = 0.
        
        # the potential terms in the action
        potential = u_0 + u_3 + u_4
        
        # multiply the potential by the lattice spacing as required
        euclidean_action = kinetic + positions.spacing * potential
        
        if self.debug: # allows for debugging
            ret_val = [euclidean_action, kinetic, potential*positions.spacing]
        else:
            ret_val = euclidean_action
        return ret_val
    
    def gradPotentialEnergy(self, positions, idx):
        """Gradient of the action
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            idx   :: integer :: lattice position
            positions :: class :: see lattice.py for info
        """
        
        # don't want the whole lattice in here!
        # the laplacian indexs the other elements
        x = positions.get[idx]
        
        # gradient of kinetic term x \klein_gordon^2 x = 2 \klein_gordon^2 x
        p_sq = positions.laplacian(idx, a_power=1)
        
        # gradient should be an array of the length of degrees of freedom 
        checks.tryAssertEqual(p_sq.shape, (),
             ' laplacian shape should be scalar' \
             + '\n> p_sq shape: {}'.format(p_sq.shape))
        
        #### grad of free action S_0: 2/2 * (m^2 - \klein_gordon^2)\phi
        kinetic = - p_sq
        u_0 = self.m * x # derivative taken
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
        derivative = kinetic + (positions.spacing * potential)
        # print 'kinetic, {}\npot: {}\n\n'.format(kinetic, potential)
        return derivative
    
    def hamiltonian(self, p, positions):
        """Returns the Hamiltonian
        
        Required Inputs
            p :: np.array (nd) :: momentum array
            positions :: class :: see lattice.py for info
        """
        if self.debug:
            h = self.kineticEnergy(p) + self.potentialEnergy(positions)[0]
        else:
            h = self.kineticEnergy(p) + self.potentialEnergy(positions)
        
        # check 1 dimensional
        checks.tryAssertEqual(h.shape, (1,)*len(h.shape),
             ' hamiltonian() not scalar.\n> shape: {}'.format(h.shape))
        return h.reshape(1)
#
class Quantum_Harmonic_Oscillator(object):
    """Quantum Harmonic Oscillator on a lattice
    
    H = \frac{m}{2}\dot{x}^2 + V(x)
    
    V(x) = \frac{1}{2}mx^2 + \frac{1}{3!}\lambda_3 x^3 + \frac{1}{4!}\lambda_4 x^4
    
    Optional Inputs
        m       :: float :: mass
        phi_3   :: phi_3 coupling constant
        phi_4   :: phi_4 coupling constant
    """
    def __init__(self, m=1., phi_3=0., phi_4=0., debug=False):
        self.name = 'QHO'
        self.debug = debug
        self.m = m
        self.phi_3 = phi_3      # phi^3 coupling const.
        self.phi_4 = phi_4      # phi^4 coupling const.
        
        self.kE  = lambda p: self.kineticEnergy(p)
        self.uE  = lambda x: self.potentialEnergy(x)
        self.dkE = lambda p: self.gradKineticEnergy(p)
        self.duE = lambda i, x: self.gradPotentialEnergy(i, x)
        
        self.all = [self.kE, self.uE, self.dkE, self.duE]
        pass
    
    def kineticEnergy(self, p):
        """n-dim KE
        
        This is the kinetic energy for the shadow hamiltonian
        
        Required Inputs
            p :: np.array (nd) :: momentum array
        """
        return .5 * np.square(p).flatten().sum(axis=0)
    
    def potentialEnergy(self, positions):
        """n-dim potential
        
        This is the action. In HMC, the action
        is the potential in the shadow hamiltonian.
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            positions :: class :: see lattice.py for info
        """
        lattice = positions.get # shortcut for brevity
        
        x_sq_sum = np.power(lattice.flatten(), 2).sum()
        
        v_sq_sum = np.array(0.) # initiate velocity squared
        # sum (integrate) across euclidean-space (i.e. all lattice sites)
        for idx in np.ndindex(lattice.shape):
            
            # sum velocity squared
            v_sq = positions.gradSquared(idx, a_power=1)
            
            # gradient should be an array of the length of degrees of freedom 
            checks.tryAssertEqual(v_sq.shape, (),
                 ' derivative^2 shape should be scalar' \
                 + '\n> v_sq shape: {}'.format(v_sq_sum.shape)
                 )
                 
            # sum to previous
            v_sq_sum +=  v_sq
        
        #### free action S_0: m/2 \phi(v^2 + m)\phi
        kinetic = .5 * self.m * v_sq_sum
        u_0 = .5 * self.m * x_sq_sum
        ### End free action
        
        # Add interation terms if required
        if self.phi_3: # phi^3 term
            x_3_sum = np.power(lattice.flatten(), 3).sum()
            u_3 = self.phi_3 * x_3_sum / np.math.factorial(3)
        else:
            u_3 = 0.
        
        if self.phi_4: # phi^4 term
            x_4_sum = np.power(lattice.flatten(), 4).sum()
            u_4 = self.phi_4 * x_4_sum / np.math.factorial(4)
        else:
            u_4 = 0.
        
        # the potential terms in the action
        potential = u_0 + u_3 + u_4
        
        # multiply the potential by the lattice spacing as required
        euclidean_action = kinetic + positions.spacing * potential
        
        if self.debug: # alows for debugging
            ret_val = [euclidean_action, kinetic, potential*positions.spacing]
        else:
            ret_val = euclidean_action
        
        return ret_val
    
    def gradPotentialEnergy(self, positions, idx):
        """Gradient of the action
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            idx   :: integer :: lattice position
            positions :: class :: see lattice.py for info
        """
        
        # don't want the whole lattice in here!
        # the laplacian indexs the other elements
        x = positions.get[idx]
        
        # derivative of velocity squared
        # the derivative of the velocity squared is actually
        # identical to - \klein_gordon^2
        v_sq =  positions.laplacian(idx, a_power=1)
        
        # gradient should be an array of the length of degrees of freedom 
        checks.tryAssertEqual(v_sq.shape, (),
             ' derivative^2 shape should be scalar' \
             + '\n> v_sq shape: {}'.format(v_sq.shape)
             )
        
        #### free action S_0: m/2 \phi(v^2 + m)\phi
        kinetic = - self.m * v_sq
        u_0 = self.m * x
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
        derivative = kinetic + (positions.spacing * potential)
            
        return derivative
    
    def hamiltonian(self, p, positions):
        """Returns the Hamiltonian
        
        Required Inputs
            p :: np.array (nd) :: momentum array
            positions :: class :: see lattice.py for info
        """
        if self.debug:
            h = self.kineticEnergy(p) + self.potentialEnergy(positions)[0]
        else:
            h = self.kineticEnergy(p) + self.potentialEnergy(positions)
        
        # check 1 dimensional
        checks.tryAssertEqual(h.shape, (1,)*len(h.shape),
             ' hamiltonian() not scalar.\n> shape: {}'.format(h.shape))
        return h.reshape(1)
#
class Simple_Harmonic_Oscillator(object):
    """Simple Harmonic Oscillator
    
    The potential is given by: F(x) = k*x
    
    Optional Inputs
        k :: float :: spring constant
    """
    def __init__(self, k=[[1.]]):
        self.name = 'SHO'
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
        self.name = 'MVG'
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