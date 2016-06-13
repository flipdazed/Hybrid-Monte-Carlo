import numpy as np

from pretty_plotting import Pretty_Plotter, viridis, magma, inferno, plasma
from lattice import Periodic_Lattice

import matplotlib.pyplot as plt
import subprocess

class Simple_Harmonic_Oscillator(object):
    """Simple Harmonic Oscillator
    
    The potential is given by: F(x) = k*x
    
    Optional Inputs
        k :: float :: spring constant
    """
    def __init__(self, k=1.):
        self.k = k
        
        self.kE = lambda p: self.kineticEnergy(p)
        self.uE = lambda x: self.potentialEnergy(x)
        self.dkE = lambda p: self.gradKineticEnergy(p)
        self.duE = lambda x: self.gradPotentialEnergy(x)
        
        self.all = [self.kE, self.uE, self.dkE, self.duE]
        
        self.mean = np.asarray([[0.]])
        self.cov = np.asarray([[1.]])
        pass
    def kineticEnergy(self, p):
        return .5 * np.dot(p.T, p)
    def potentialEnergy(self, x):
        return .5 * self.k*np.dot(x.T, x)
    def gradKineticEnergy(self, p):
        return p
    def gradPotentialEnergy(self, x):
        return self.k*x
    def hamiltonian(self, p, x):
        h = self.kineticEnergy(p) + self.potentialEnergy(x)
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
        assert len(p.shape) == 2
        return .5 * np.square(p).sum(axis=0)
    
    def potentialEnergy(self, x):
        """n-dim potential
        
        Required Inputs
            x :: np.matrix (col vector) :: position vector
        """
        assert x.shape == self.mean.shape
        x -= self.mean
        return .5 * ( np.dot(x.T, self.cov_inv) * x).sum(axis=0)
    
    def gradKineticEnergy(self, p):
        """n-dim Kinetic Energy"""
        assert len(p.shape) == 2
        return p
    
    def gradPotentialEnergy(self, x):
        """n-dim gradient"""
        assert len(x.shape) == 2
        return np.dot(self.cov_inv, x)
    def hamiltonian(self, p, x):
        h = self.kineticEnergy(p) + self.potentialEnergy(x)
        assert h.shape == (1,)*len(h.shape) # check 1 dimensional
        return h.reshape(1)
#
class Quantum_Harmonic_Oscillator(object):
    """Quantum Harmonic Oscillator on a lattice
    
    H = \frac{m}{2}\dot{x}^2 + V(x)
    
    V(x) = \frac{1}{2}mx^2 + \frac{1}{3!}\lambda_3 x^3 + \frac{1}{4!}\lambda_4 x^4
    
    Required Inputs
        lattice :: class :: see lattice.py for info
    
    Optional Inputs
        m       :: float :: mass
        phi_3   :: phi_3 coupling constant
        phi_4   :: phi_4 coupling constant
    """
    def __init__(self, lattice, m=1., phi_3=0., phi_4=0.):
        self.m = m
        self.lattice = lattice
        
        self.phi_3 = phi_3      # phi^3 coupling const.
        self.phi_4 = phi_4      # phi^4 coupling const.
        
        self.kE = lambda p: self.kineticEnergy(p)
        self.uE = lambda i: self.potentialEnergy(i)
        self.dkE = lambda p: self.gradKineticEnergy(p)
        self.duE = lambda i: self.gradPotentialEnergy(i)
        
        self.all = [self.kE, self.uE, self.dkE, self.duE]
        pass
    
    def kineticEnergy(self, p):
        """n-dim KE
        
        This is the kinetic energy for the shadow hamiltonian
        
        Required Inputs
            p :: np.matrix (col vector) :: momentum vector
        """
        assert len(p.shape) == 2
        return .5 * np.square(p).sum(axis=0)
    
    def potentialEnergy(self, idx):
        """n-dim potential
        
        This is the true hamiltonian. In HMC, the hamiltonian
        is the potential in the shadow hamiltonian.
        
        Required Inputs
            # x :: np.matrix (col vector) :: position vector
            i   :: integer :: euclidean time
        """
        lattice = self.lattice.get # shortcut for brevity
        
        k_e = .5 * self.m
        k_e *= self.lattice.gradSquared(idx, a_power=2) # (dx/dt)^2
        
        u_0 = .5 * self.m                       # mass term: 1/2 * m
        u_0 *= np.dot(lattice[idx].T, lattice[idx])   # position at t=i: x(t)^2
        
        # phi^3 term
        u_3 = self.phi_3 * np.dot(lattice[idx].T, lattice[idx]) * lattice[idx]
        u_3 /= np.math.factorial(3)
        
        # phi^4 terms
        u_4 = self.phi_4 * np.dot(lattice[idx].T, lattice[idx]) * lattice[idx]
        u_4 /= np.math.factorial(4)
        
        u_e = u_0 + u_3 + u_4
        h = k_e - u_e
        return h
    
    def gradKineticEnergy(self, p):
        """Gradient w.r.t. conjugate momentum"""
        return p
    
    def gradPotentialEnergy(self, idx):
        """Gradient of the true hamiltonian"""
        lattice = self.lattice.get # shortcut
        
        # kinetic term:  p/m = m * v[i] = sqrt(v[i]^2)
        dh_dp = self.m * np.sqrt(self.lattice.gradSquared(idx, a_power=2)) 
        
        # mass term (phi^2)
        dh_dx = self.m * lattice[idx]
        # phi^3
        dh_dx += self.phi_3 * np.dot(lattice[idx].T, lattice[idx])
        dh_dx /= np.math.factorial(2)
        # phi^4
        dh_dx += self.phi_4 * np.dot(lattice[idx].T, lattice[idx]) * lattice[idx]
        dh_dx /= np.math.factorial(3)
        
        return dh_dx + dh_dp
    
    def hamiltonian(self, idx):
        h = self.kineticEnergy(idx) + self.potentialEnergy(idx)
        assert h.shape == (1,)*len(h.shape) # check 1 dimensional
        return h.reshape(1)
#
class Test(Pretty_Plotter):
    def __init__(self):
        self.mean = np.asarray([[0.], [0.]])
        self.cov = np.asarray([[1.0,0.8],[0.8,1.0]])
        self.bg = Multivariate_Gaussian(mean = self.mean, cov = self.cov)
        pass
    
    def testBG(self, save = 'potentials_Gaussian_2d.png'):
        """Plots a test image of the Bivariate Gaussian"""
        self._teXify() # LaTeX
        self.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
        self.params['figure.subplot.top'] = 0.85
        self._updateRC()
        
        n = 200 # n**2 is the number of points
        cov = self.bg.cov
        mean = self.bg.mean
        
        x = np.linspace(-5., 5., n, endpoint=True)
        x,y = np.meshgrid(x,x)
        z = np.exp(-np.asarray([self.bg.uE(np.matrix([[i],[j]])) \
            for i,j in zip(np.ravel(x), np.ravel(y))]))
        z = np.asarray(z).reshape(n, n)
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        c = ax.contourf(x, y, z, 100, cmap=plasma)
        
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        fig.suptitle(r'Test plot of a 2D Multivariate (Bivariate) Gaussian',
             fontsize=self.ttfont*self.s)
        ax.set_title(
        r'Parameters: $\mu=\begin{pmatrix}0 & 0\end{pmatrix}$, $\Sigma = \begin{pmatrix} 1.0 & 0.8\\ 0.8 & 1.0 \end{pmatrix}$',
            fontsize=(self.tfont-4)*self.s)
        
        ax.grid(False)
        
        if save:
            save_dir = './plots/'
            subprocess.call(['mkdir', './plots/'])
            
            fig.savefig(save_dir+save)
        else:
            plt.show()
        pass
    def testQHO(self, dim = 4, sites = 10, spacing = 1.):
        """checks that QHO can be initialised and all functions run"""
        np.set_printoptions(suppress=True)
        
        passed = True
        shape = (sites,)*dim
        raw_lattice = np.arange(sites**dim).reshape(shape)
        self.lattice = Periodic_Lattice(array = raw_lattice, spacing = 1.)
        
        passed *= (self.lattice.get.shape == shape) # check dimensions
        
        test_idx = tuple(np.random.randint(0, 10000, dim))
        wrapped_idx = self.lattice.wrapIdx(test_idx)
        lap = self.lattice.laplacian(test_idx, a_power=2.)
        print 'Lap x({})~Lap x({}) : {}'.format(test_idx, wrapped_idx, lap)
        grad = np.sqrt(self.lattice.gradSquared(test_idx, a_power=2.))
        print 'v({})^2~v({})^2 : {}'.format(test_idx, wrapped_idx, grad)
        return passed
#
if __name__ == '__main__':
    test = Test()
    # test.testBG(save = 'potentials_Gaussian_2d.png')
    print 'testing QHO:', test.testQHO()
    