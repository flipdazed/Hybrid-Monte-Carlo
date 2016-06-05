import numpy as np

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
        pass
    def kineticEnergy(self, p):
        return .5 * p**2
    def potentialEnergy(self, x):
        return .5 * self.k*x**2
    def gradKineticEnergy(self, p):
        return p
    def gradPotentialEnergy(self, x):
        return self.k*x
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
        assert (self.cov.T == self.cov).all() # must be symmetric
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
        return .5 * np.square(p).sum(axis=1)
    
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
    def testPlot(self):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams['text.usetex'] = True
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        
        n = 1000                                # n**2 is the number of points
        cov = self.cov
        mean = self.mean
        print 'random uniform...'
        x = np.linspace(-10., 10., n, endpoint=True)
        x,y = np.meshgrid(x,x)
        
        print 'gaussian...'
        z = np.exp(-np.asarray([self.potentialEnergy(np.matrix([[i],[j]])) for i,j in zip(np.ravel(x), np.ravel(y))]))
        z = np.asarray(z).reshape(n,n)
        
        print 'plotting...'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        l = ax.contourf(x, y, z)
        # l = ax.plot_surface(x, y, z,rstride=1, cstride=1, cmap=cm.coolwarm,
        #                    linewidth=0, antialiased=False)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        ax.set_xlabel(r'$\mathrm{x_1}$')
        ax.set_ylabel(r'$\mathrm{x_2}$')
        # ax.set_zlabel(r'$\mathrm{P(x)}$')
        plt.title(r'$\mathrm{Multivariate\ Gaussian\ Test:}\ \mu=\begin{pmatrix}0 & 0\end{pmatrix},\ \Sigma = \begin{pmatrix} 1.0 & 0.8\\ 0.8 & 1.0 \end{pmatrix}$')
        # plt.xlim([-1, 1])
        # plt.grid(True)
        
        plt.show()
#
if __name__ == '__main__':
    test = Multivariate_Gaussian()
    test.testPlot()