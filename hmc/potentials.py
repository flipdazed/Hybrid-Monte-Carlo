import numpy as np

from pretty_plotting import Pretty_Plotter, viridis, magma, inferno, plasma

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
    """Quantum Harmonic Oscillator
    
    Required Inputs
        x       :: array :: euclidean time & positions (x[i] = time)
    
    Optional Inputs
        m       :: float :: mass
        w       :: float :: angular frequency
    """
    def __init__(self, x, m=1., w=[[1.]], phi_3=0., phi_4=0.):
        self.m = m
        self.w = np.asmatrix(w)
        self.x = np.asarray(x)
        self.phi_3 = phi_3
        self.phi_4 = phi_4
        
        self.kE = lambda p: self.kineticEnergy(p)
        self.uE = lambda x: self.potentialEnergy(x)
        self.dkE = lambda p: self.gradKineticEnergy(p)
        self.duE = lambda x: self.gradPotentialEnergy(x)
        
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
    
    def potentialEnergy(self, t):
        """n-dim potential
        
        This is the true hamiltonian. In HMC, the hamiltonian
        is the potential in the shadow hamiltonian.
        
        Required Inputs
            # x :: np.matrix (col vector) :: position vector
            i   :: integer :: euclidean time
        """
        k = .5 * self.m
        k *= self._velSq(t)
        
        u_0 = .5 * self.m                       # mass term
        u_0 *= np.dot(self.w.T, self.w)         # angular freq.
        u_0 *= np.dot(self.x[t].T, self.x[t])   # position at time i
        
        # phi^3 term
        u_3 = self.phi_3 * np.dot(self.x[t].T, self.x[t]) * self.x[t]
        u_3 /= np.math.factorial(3)
        
        u_4 = self.phi_4 * np.dot(self.x[t].T, self.x[t]) * self.x[t]
        u_4 /= np.math.factorial(4)
        
        h = k_e - u_e
        return h
    
    def gradKineticEnergy(self, p):
        """Gradient w.r.t. conjugate momentum"""
        return p
    
    def gradPotentialEnergy(self, t):
        """Gradient of the true hamiltonian"""
        dh_dp = np.sqrt(self._velSq(t)) # p/m = v[i] = sqrt(v[i]^2)
        dh_dx = self.m * np.dot(self.w.T, self.w) * self.x[t] # m*w^2*x[i]
        dh_dx += self.phi_3 * np.dot(self.x[t].T, self.x[t])
        return dh_dx + dh_dp
    
    def hamiltonian(self, t):
        h = self.kineticEnergy(t) + self.potentialEnergy(t)
        assert h.shape == (1,)*len(h.shape) # check 1 dimensional
        return h.reshape(1)
    def _velSq(self, t):
        """returns the time derivative of position squared"""
        return (self.x[t+1] - self.x[t]) * (self.x[t] - self.x[t-1])
    
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
#
if __name__ == '__main__':
    test = Test()
    test.testBG(save = 'potentials_Gaussian_2d.png')