import numpy as np
import copy

from potentials import Simple_Harmonic_Oscillator, Multivariate_Gaussian
from h_dynamics import Leap_Frog

from pretty_plotting import Pretty_Plotter, viridis, magma, inferno, plasma

import matplotlib.pyplot as plt
import subprocess

class Hybrid_Monte_Carlo(object):
    """The Hybrid (Hamiltonian) Monte Carlo method
    
    Optional Inputs
        
    
    Required Inputs
        x0         :: tuple :: initial starting position vector
        potential  :: class :: class from potentials.py
        dynamics   :: class :: integrator class for dynamics from h_dynamics.py
        rng        :: np.random.RandomState :: random number state
    Expectations
    """
    def __init__(self, x0, dynamics, potential, rng):
        
        self.x0 = x0
        self.dynamics = dynamics
        self.potential = potential
        self.rng = rng
        
        self.momentum = Momentum(self.rng)
        self.accept = Accept_Reject(self.rng)
        
        self.x = self.x0
        self.p = self.momentum.fullRefresh(self.x0) # intial mom. sample
        assert self.x.shape == self.p.shape
        
        pass
    
    def sample(self, n_samples, n_burn_in = 1000):
        """runs the sampler for HMC
        
        Required Inputs
            n_samples   :: integer :: Number of samples (# steps after burn in)
        
        Optional Inputs
            n_burn_in   :: integer :: Number of steps to discard at start
            store_path  :: bool    :: Store path for plotting
        """
        self.burn_in_p = [copy.copy(self.p)]
        self.burn_in = [copy.copy(self.x)]
        
        for step in xrange(n_burn_in): # burn in
            self.p, self.x = self.moveHMC()
            self.burn_in_p.append(copy.copy(self.p))
            self.burn_in.append(copy.copy(self.x))
        
        self.samples_p = [copy.copy(self.p)]
        self.samples = [copy.copy(self.x)]
        
        for step in xrange(n_samples):
            p, x = self.moveHMC()
            self.samples_p.append(copy.copy(self.p))
            self.samples.append(copy.copy(self.x))
        
        return (self.burn_in_p, self.samples_p), (self.burn_in, self.samples)
    
    def moveHMC(self, step_size = None, n_steps = None):
        """A Hybrid Monte Carlo move:
        Combines Hamiltonian Dynamics and Momentum Refreshment
        to generate a the next position for the MCMC chain
        
        Optional Inputs
            step_size   :: float    :: step_size for integrator
            n_steps     :: integer  :: number of integrator steps
        
        Expectations
            self.p, self.x
            self.dynamics.integrate
            self.momentum.fullRefresh
        
        Returns
            (p,x) :: (float, float) :: new momentum and position
        """
        
        p,x = self.p,self.x # initial temp. proposal p,x
        h_old = self.potential.hamiltonian(p, x)
        
        p = -self.momentum.fullRefresh(p) # mixing matrix adds a flip
        
        if (step_size is not None): self.dynamics.step_size = step_size
        if (n_steps is not None): self.dynamics.n_steps = step_size
        
        p, x = self.dynamics.integrate(p, x)
        # p = self.momentum.flip(p) # not necessary as full refreshment
        h_new = self.potential.hamiltonian(p, x)
        
        try:
            accept = self.accept.metropolisHastings(h_old=h_old, h_new=h_new)
            if not accept: p, x = self.p, self.x
        except Exception as e:
            print h_new
            print h_old
            raise e
            
        return p,x
    
    def moveGHMC(self, mixing_angle, step_size = None, n_steps = None):
        """A generalised Hybrid Monte Carlo move:
        As HMC but includes partial momentum refreshment
        
        Required Inputs
            mixing_angle :: float :: see Momentum()._rotationMatrix()
        
        Optional Inputs
            step_size   :: float    :: step_size for integrator
            n_steps     :: integer  :: number of integrator steps
        
        Expectations
            self.p, self.x
            self.dynamics.integrate
            self.momentum.fullRefresh
        
        Returns
            (p,x) :: (float, float) :: new momentum and position
        """
        p,x = self.p,self.x # initial temp. proposal p,x
        h_old = self.potential.hamiltonian(p, x)
        
        p = self.momentum.generalisedRefresh(p, theta=mixing_angle)
        
        if (step_size is not None): self.dynamics.step_size = step_size
        if (n_steps is not None): self.dynamics.n_steps = step_size
        
        p, x = self.dynamics.integrate(p, x)
        p = self.momentum.flip(p)
        h_new = self.potential.hamiltonian(p, x)
        
        accept = self.accept.metropolisHastings(h_old=h_old, h_new=h_new)
        if not accept: p, x = self.p, self.x
        
        return p,x
    
    
#
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
#
class Momentum(object):
    """Momentum Routines
    
    Required Inputs
        rng :: np.random.RandomState :: random number generator
    
    """
    def __init__(self, rng):
        self.rng = rng
        pass
    
    def fullRefresh(self, p):
        """Performs full refresh"""
        p = self.generalisedRefresh(p)
        return p
    
    def generalisedRefresh(self, p, mixing_angle=.5*np.pi):
        """Performs partial refresh through mixing angle
        to mix momentum with gaussian noise
        
        Optional Inputs
            mixing_angle :: float :: 0. is no mixing
        
        Required Inputs
            p :: np.array :: momentum to refresh
        """
        
        # Random Gaussian noise with: sdev=scale & mean=loc
        self.noise = self.rng.normal(size=p.shape, scale=1., loc=0.)
        self.mixed = self._refresh(p, self.noise, theta=mixing_angle)
        
        return self.mixed[:p.shape[0],:p.shape[1]]
    
    def _refresh(self, p, noise, theta):
        """Mixes noise with momentum
        
        Required Inputs
            p       :: np.array :: momentum to refresh
            noise   :: np.array :: noise to mix with momentum
            theta   :: float    :: mixing angle
        """
        
        self.rot = self._rotationMatrix(n_dim=p.shape[0], theta=theta)
        
        unmixed = np.bmat([[p],[noise]])
        flipped = self.flip(unmixed)
        
        # matix multiplication
        mixed = self.rot*flipped
        
        return mixed
    
    def _rotationMatrix(self, n_dim, theta):
        """A rotation matrix
        
        Required Input
            theta :: float :: angle in radians for mixing
            n_dim :: tuple :: creates n_dim^2 blocks (total: 4*n_dim^2)
        """
        i = np.identity(n_dim)
        c, s = np.cos(theta)*i, np.sin(theta)*i
        rotation = np.bmat([[c, s], [-s, c]])
        return rotation
    
    def flip(self, p):
        """Reverses the momentum
        
        Required Inputs
            p       :: np.array :: momentum to refresh
        """
        return -p
    
    def test(self, print_out=False):
        """tests the momentum refreshment
        
        Required Input
            print_out :: bool :: print results to screen
        
        1. tests 3 input momentum shapes
        2. tests 4 mixing angles
        """
        
        if print_out: np.set_printoptions(precision=2, suppress=True)
        
        passed = True
        rng =  np.random.RandomState(1234)
        
        rand4 = np.random.random(4)
        p41 = np.mat(rand4.reshape(4,1))
        p22 = np.mat(rand4.reshape(2,2))
        p14 = np.mat(rand4.reshape(1,4))
        
        passed *= self._testVectors(p41, "4x1 vector + fullRefresh()", print_out)
        passed *= self._testVectors(p22, "2x2 vector + fullRefresh()", print_out)
        passed *= self._testVectors(p14, "1x4 vector + fullRefresh()", print_out)
        
        passed *= self._testMixing(print_out)
        return passed
    
    def _testMixing(self, print_out):
        """Tests the mixing matrix
        
        Required Input
            print_out :: bool :: print results to screen
        """
        
        def display(test_name, p, expected, result): # print to terminal
            print '\n\n TEST: {}'.format(test_name)
            print '\n original:\n',np.bmat([[p],[self.noise]])
            print '\n rotation matrix:\n',self.rot
            print '\n mix + flip:\n',self.mixed
            print '\n expected:\n',expected
            print '\n outcome: {}'.format(['Failed','Passed'][result])
            pass
        
        p14 = np.mat([[1.,7.,-1., 400.]])
        self.noise = np.mat([[0.1, 3., 1., -2]])
        
        expected_0 = np.mat([[-1., -7., 1., -400.], [-0.1, -3., -1., 2.]])
        expected_halfpi = np.mat([[-0.1, -3., -1., 2.], [1., 7., -1., 400.]])
        expected_pi = -expected_0
        expected_quartpi = np.mat( [[-0.778, -7.071, 0., -281.428],
                                    [0.636, 2.828, -1.414, 284.257]])
        
        self.mixed = self._refresh(p14, self.noise, theta = 0.)
        result = (np.around(self.mixed, 3) == expected_0).all()
        if print_out: display("Theta = 0. (No Mix)", p14, expected_0, result)
        
        self.mixed = self._refresh(p14, self.noise, theta = np.pi)
        result = (np.around(self.mixed, 3) == expected_pi).all()
        if print_out: display("Theta = pi (- No Mix)", p14, expected_pi, result)
        
        self.mixed = self._refresh(p14, self.noise, theta = np.pi/2.)
        result = (np.around(self.mixed, 3) == expected_halfpi).all()
        if print_out: display("Theta = pi/2. (Total Mix)", p14, expected_halfpi, result)
        
        self.mixed = self._refresh(p14, self.noise, theta = np.pi/4.)
        result = (np.around(self.mixed, 3) == expected_quartpi).all()
        if print_out: display("Theta = pi/4. (Partial Mix)", p14, expected_quartpi, result)
        
        return result
    
    def _testVectors(self, p, test_name, print_out):
        """Tests different momentum vector shape for correct mixing
        
        Required Inputs
            p           :: np.array :: momentum to refresh
            test_name   :: string   :: name of test
            print_out   :: bool     :: print results to screen
        """
        
        def display(test_name, p, p_mixed, result): # print to terminal
            print '\n\n TEST: {}'.format(test_name)
            print '\n original vector:\n',np.bmat([[p],[self.noise]])
            print '\n rotation matrix:\n',self.rot
            print '\n mixed momentum + flip:\n',p_mixed
            print '\n outcome: {}'.format(['Failed','Passed'][result])
            pass
        
        p_mixed = self.fullRefresh(p=p)
        result = (np.around(self.flip(p_mixed), 6) == np.around(self.noise, 6)).all()
        if print_out: display(test_name, p, p_mixed, result)
        
        return result
#
class Test_HMC(Pretty_Plotter):
    """Tests for the HMC class
    
    Required Inputs
        rng :: np.random.RandomState :: random number generator
    """
    def __init__(self, rng):
        self.rng = rng
        
        self.sho = Simple_Harmonic_Oscillator()
        self.bg = Multivariate_Gaussian()
        self.lf = Leap_Frog(duE = self.sho.duE, step_size = 0.1, n_steps = 20)
        x0 = np.asarray([[0.]]) # start at 0 by default
        
        self.hmc = Hybrid_Monte_Carlo(x0, self.lf, self.sho, self.rng)
        pass
    
    def hmcSho1d(self, tol = 5e-2, print_out = True, save = 'HMC_oscillator_1d.png'):
        """A test to sample the Simple Harmonic Oscillator
        
        Optional Inputs
            tol     ::  float   :: tolerance level allowed
            print_out   :: bool     :: print results to screen
            save    :: string   :: file to save plot. False or '' gives no plot
        """
        passed = True
        def display(test_name, act_mean, act_cov, mean, cov, mean_tol, cov_tol, result):
            print '\n\n TEST: {}'.format(test_name)
            print ' target mean: ', act_mean
            print ' target cov: ', act_cov
            print '\n empirical mean: ', mean
            print ' empirical_cov: ', cov
            print '\n mean tol: ', mean_tol
            print ' cov tol: ', cov_tol
            print '\n outcome: {}'.format(['Failed','Passed'][result])
            pass
        
        x0 = np.asarray([[1.]])
        
        self.lf.duE = self.sho.duE # reassign leapfrog gradient
        self.hmc.__init__(x0, self.lf, self.sho, self.rng)
        
        n_samples = 15000
        n_burn_in = 1000
        
        act_mean = self.hmc.potential.mean
        act_cov = self.hmc.potential.cov
        
        mean_tol = np.full(act_mean.shape, tol)
        cov_tol = np.full(act_cov.shape, tol)
        
        p_samples, samples = self.hmc.sample(n_samples = n_samples, n_burn_in=n_burn_in)
        burn_in, samples = samples
        samples = np.asarray(samples)
        
        mean = samples.mean(axis=0)
        cov = np.cov(samples)
        
        passed *= (np.abs(mean - act_mean) <= mean_tol).all()
        passed *= (np.abs(cov - act_cov) <= cov_tol).all()
        
        if print_out: 
            np.set_printoptions(precision=2, suppress=True)
            display("HMC: Simple Harmonic Oscillator",
                act_mean, act_cov, mean, cov, mean_tol, cov_tol, passed)
        
        def plot(burn_in, samples, save=save):
            
            self._teXify() # LaTeX
            self.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
            self._updateRC()
            
            fig = plt.figure(figsize = (8*self.s, 8*self.s)) # make plot
            ax =[]
            ax.append(fig.add_subplot(111))
            fig.suptitle(r'Sampling the Simple Harmonic Oscillator',
                fontsize=16)
            ax[0].set_title(
                r'{} Burn-in Samples shown in orange'.format(burn_in.shape[0]-1))
            ax[0].set_ylabel(r'Sample, $n$')
            ax[0].set_xlabel(r"Position, $x$")
            
            offst = burn_in.shape[0]+1 # burn-in samples
            ax[0].plot(burn_in, np.arange(1, offst), #marker='x',
                linestyle='-', color='orange', label=r'Burn In')
            ax[0].plot(samples, np.arange(offst, offst + samples.shape[0]), #marker='x',
                linestyle='-', color='blue', label=r'Sampling')
            
            if save:
                save_dir = './plots/'
                subprocess.call(['mkdir', './plots/'])
            
                fig.savefig(save_dir+save)
            else:
                plt.show()
            pass
        
        if save: 
            burn_in = np.asarray(burn_in).reshape(n_burn_in+1)
            samples = np.asarray(samples).reshape(n_samples+1)
            
            plot(burn_in, samples)
        
        return passed
    
    def hmcGaus2d(self, tol = 5e-2, print_out = True, save = 'HMC_gauss_2d.png'):
        """A test to sample the 2d Gaussian Distribution
        
        Optional Inputs
            tol     ::  float   :: tolerance level allowed
            print_out   :: bool     :: print results to screen
            save    :: string   :: file to save plot. False or '' gives no plot
        """
        passed = True
        def display(test_name, act_mean, act_cov, mean, cov, mean_tol, cov_tol, result):
            print '\n\n TEST: {}'.format(test_name)
            print ' target mean: ', act_mean
            print ' target cov: ', act_cov
            print '\n empirical mean: ', mean
            print ' empirical_cov: ', cov
            print '\n mean tol: ', mean_tol
            print ' cov tol: ', cov_tol
            print '\n outcome: {}'.format(['Failed','Passed'][result])
            pass
        
        x0 = np.asarray([[-3.5], [4.]])
        
        self.lf.duE = self.bg.duE # reassign leapfrog gradient
        self.hmc.__init__(x0, self.lf, self.bg, self.rng)
        
        n_samples = 0
        n_burn_in = 50
        
        act_mean = self.hmc.potential.mean
        act_cov = self.hmc.potential.cov
        
        mean_tol = np.full(act_mean.shape, tol)
        cov_tol = np.full(act_cov.shape, tol)
        
        p_samples, samples = self.hmc.sample(n_samples = n_samples, n_burn_in=n_burn_in)
        burn_in, samples = samples
        samples = np.asarray(samples).T.reshape(2, -1).T
        burn_in = np.asarray(burn_in).T.reshape(2, -1).T
        
        mean = samples.mean(axis=0)
        cov = np.cov(samples.T)
        
        passed *= (np.abs(mean - act_mean) <= mean_tol).all()
        # passed *= (np.abs(cov - act_cov) <= cov_tol).all()
        
        if print_out: 
            np.set_printoptions(precision=2, suppress=True)
            display("HMC: Bivariate Gaussian Distribution",
                act_mean, act_cov, mean, cov, mean_tol, cov_tol, passed)
        
        def plot(burn_in, samples, cov, mean, save=save):
            
            self._teXify() # LaTeX
            self.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
            self.params['figure.subplot.top'] = 0.85
            self._updateRC()
            
            n = 100    # n**2 is the number of points
            
            x = np.linspace(-5., 5., n, endpoint=True)
            x,y = np.meshgrid(x, x)
            
            z = np.exp(-np.asarray([self.bg.uE(np.matrix([[i],[j]])) for i,j in zip(np.ravel(x), np.ravel(y))]))
            z = np.asarray(z).reshape(n,n)
            
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111)
            
            c = ax.contourf(x, y, z, 100, cmap=viridis)
            l1 = ax.plot(burn_in[:,0], burn_in[:,1], 
                color='blue',
                marker='o', markerfacecolor='red'
                )
            # l2 = ax.plot(samples[:,0], samples[:,1],
            #     color='blue',
            #     # marker='o', markerfacecolor='r'
            #     )
                
            ax.set_xlabel(r'$\mathrm{x_1}$')
            ax.set_ylabel(r'$\mathrm{x_2}$')
            
            fig.suptitle(r'Sampling Multivariate Gaussian with HMC', fontsize=self.ttfont)
            ax.set_title(r'Showing the first 50 HMC moves for:\ $\mu=\begin{pmatrix}0 & 0\end{pmatrix}$, $\Sigma = \begin{pmatrix} 1.0 & 0.8\\ 0.8 & 1.0 \end{pmatrix}$', fontsize=self.tfont-4)
            
            plt.grid(True)
            
            if save:
                save_dir = './plots/'
                subprocess.call(['mkdir', './plots/'])
            
                fig.savefig(save_dir+save)
            else:
                plt.show()
            pass
        
        if save:
            plot(burn_in, samples, cov, mean)
            
        return burn_in, samples
#
if __name__ == '__main__':
    rng = np.random.RandomState(1234)
    m = Momentum(rng)
    r1 = m.test(print_out=False)
    test = Test_HMC(rng)
    # r2 = test.hmcSho1d(tol = 5e-2, print_out = True, save = 'HMC_oscillator_1d.png')
    r3 = test.hmcGaus2d(tol = 5e-2, print_out = True, save = 'HMC_gauss_2d.png')