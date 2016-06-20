import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import subprocess

import utils

# these directories won't work unless 
# the commandline interface for python unittest is used
from hmc.potentials import Simple_Harmonic_Oscillator, Multivariate_Gaussian, Quantum_Harmonic_Oscillator, Klein_Gordon
from hmc.hmc import *
from plotter import Pretty_Plotter, PLOT_LOC
from plotter import *

TEST_ID = 'HMC'

class Test(Pretty_Plotter):
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
    
    def hmcSho1d(self, n_samples = 10000, n_burn_in = 50, tol = 5e-2, print_out = True, save = 'HMC_sho_1d.png'):
        """A test to sample the Simple Harmonic Oscillator
        
        Optional Inputs
            tol     ::  float   :: tolerance level allowed
            print_out   :: bool     :: print results to screen
            save    :: string   :: file to save plot. False or '' gives no plot
        """
        passed = True
        
        x0 = np.asarray([[1.]])
        dim = x0.shape[0] # dimension the column vector
        self.lf.duE = self.sho.duE # reassign leapfrog gradient
        self.hmc.__init__(x0, self.lf, self.sho, self.rng)
        
        act_mean = self.hmc.potential.mean
        act_cov = self.hmc.potential.cov
        
        mean_tol = np.full(act_mean.shape, tol)
        cov_tol = np.full(act_cov.shape, tol)
        
        p_samples, samples = self.hmc.sample(n_samples = n_samples, n_burn_in = n_burn_in)
        burn_in, samples = samples # return the shape: (n, dim, 1)
        
        # flatten last dimension to a shape of (n, dim)
        samples = np.asarray(samples).T.reshape(dim, -1).T
        burn_in = np.asarray(burn_in).T.reshape(dim, -1).T
        
        mean = samples.mean(axis=0)
        # covariance assumes observations in columns
        # we have observations in rows so specify rowvar=0
        cov = np.cov(samples, rowvar=0)
        
        passed *= (np.abs(mean - act_mean) <= mean_tol).all()
        passed *= (np.abs(cov - act_cov) <= cov_tol).all()
        
        if print_out:
            minimal = (print_out == 'minimal')
            utils.display("HMC: Simple Harmonic Oscillator", passed,
                details = {
                    'mean':[
                        'target:    {}'.format(     act_mean),
                        'empirical  {}'.format(     mean),
                        'tolerance  {}'.format(     mean_tol)
                        ],
                    'covariance':[
                        'target:    {}'.format(     act_cov ),
                        'empirical  {}'.format(     cov),
                        'tolerance  {}'.format(     cov_tol)
                        ]
                    },
                minimal = minimal)
        
        def plotPath(burn_in, samples, save=save):
            """Note that samples and burn_in contain the initial conditions"""
            self._teXify() # LaTeX
            self.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
            self._updateRC()
            
            fig = plt.figure(figsize = (8*self.s, 8*self.s)) # make plot
            ax =[]
            ax.append(fig.add_subplot(111))
            fig.suptitle(r'Example HMC path sampling the SHO potential',
                fontsize=16)
            ax[0].set_title(
                r'{} Burn-in Samples shown in orange'.format(burn_in.shape[0]))
            ax[0].set_ylabel(r'Sample, $n$')
            ax[0].set_xlabel(r"Position, $x$")
            
            offst = burn_in.shape[0]+1 # burn-in samples
            ax[0].plot(burn_in, np.arange(1, offst), #marker='x',
                linestyle='-', color='orange', label=r'Burn In')
            ax[0].plot(samples, np.arange(offst, offst + samples.shape[0]), #marker='x',
                linestyle='-', color='blue', label=r'Sampling')
            
            if save:
                save_dir = PLOT_LOC + 'plots/'
                subprocess.call(['mkdir', PLOT_LOC + 'plots/'])
                save = save.split('.')
                save = save[0] + '_path' + '.' + save[1] # add identifier
                fig.savefig(save_dir+save)
            else:
                plt.show()
            pass
        def plotPot(samples, save=save):
            """Note that samples and burn_in contain the initial conditions"""
            self._teXify() # LaTeX
            self.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
            self._updateRC()
            
            n = 100 # size of linear space
            x = np.linspace(-5,5,n)
            
            fig = plt.figure(figsize = (8*self.s, 8*self.s)) # make plot
            ax =[]
            ax.append(fig.add_subplot(111))
            fig.suptitle(r'Sampled SHO Potential',
                fontsize=16)
            ax[0].set_title(
                r'{} HMC samples. True potential in Blue.'.format(samples.shape[0]))
            ax[0].set_ylabel(r'Sampled Potential, $e^{-V(x)}$')
            ax[0].set_xlabel(r"Position, $x$")
            
            # fitted normal dist. parameters p[0] = mean, p[1] = stdev
            p = norm.fit(samples)
            fitted = norm.pdf(x, loc=p[0], scale=p[1])
            actual = norm.pdf(x)
            
            n, bins, patches = ax[0].hist(samples, 50, normed=1, # histogram
                facecolor='green', alpha=0.5, label=r'Sampled Data')
            
            ax[0].plot(x, fitted, # marker='x', # best fit
                linestyle='-', color='orange', label=r'Fitted Potential')
            
            ax[0].plot(x, actual, # marker='x',
                linestyle='-', color='blue', label=r'True Potential')
            
            if save:
                save_dir = PLOT_LOC + 'plots/'
                subprocess.call(['mkdir', PLOT_LOC + 'plots/'])
                
                save = save.split('.')
                save = save[0] + '_pot' + '.' + save[1] # add identifier
                fig.savefig(save_dir+save)
            else:
                plt.show()
            pass
        
        
        if save:
            # burn_in includes initial cond.
            # samples inclues final burn_in as initial cond.
            burn_in = np.asarray(burn_in).reshape(n_burn_in+1)
            samples = np.asarray(samples).reshape(n_samples+1)
        
        if save:
            if save == 'plot': 
                plotPath(burn_in[:50], samples=np.asarray([]), save = False)
                plotPot(samples, save = False)
            else:
                plotPath(burn_in[:50], samples=np.asarray([]), save = save)
                plotPot(samples, save = save)
        
        return passed, burn_in, samples
    
    def hmcGaus2d(self, n_samples = 10000, n_burn_in = 50, tol = 5e-2, print_out = True, save = 'HMC_gauss_2d.png'):
        """A test to sample the 2d Gaussian Distribution
        
        Optional Inputs
            tol     ::  float   :: tolerance level allowed
            print_out   :: bool     :: print results to screen
            save    :: string   :: file to save plot. False or '' gives no plot
        """
        passed = True
        
        x0 = np.asarray([[-3.5], [4.]])
        
        dim = x0.shape[0] # dimension the column vector
        self.lf.duE = self.bg.duE # reassign leapfrog gradient
        self.hmc.__init__(x0, self.lf, self.bg, self.rng)
        
        act_mean = self.hmc.potential.mean
        act_cov = self.hmc.potential.cov
        
        mean_tol = np.full(act_mean.shape, tol)
        cov_tol = np.full(act_cov.shape, tol)
        
        p_samples, samples = self.hmc.sample(n_samples = n_samples, n_burn_in=n_burn_in)
        burn_in, samples = samples # return the shape: (n, 2, 1)
        
        # flatten last dimension to a shape of (n, 2)
        samples = np.asarray(samples).T.reshape(dim, -1).T
        burn_in = np.asarray(burn_in).T.reshape(dim, -1).T
        
        mean = samples.mean(axis=0)
        # covariance assumes observations in columns
        # we have observations in rows so specify rowvar=0
        cov = np.cov(samples, rowvar=0)
        
        passed *= (np.abs(mean - act_mean) <= mean_tol).all()
        # passed *= (np.abs(cov - act_cov) <= cov_tol).all()
        
        if print_out:
            minimal = (print_out == 'minimal')
            utils.display("HMC: Bivariate Gaussian", passed,
                details = {
                    'mean':[
                        'target:    {}'.format(     act_mean.reshape(np.prod(act_mean.shape))),
                        'empirical  {}'.format(     mean.reshape(np.prod(mean.shape))),
                        'tolerance  {}'.format(     mean_tol.reshape(np.prod(mean_tol.shape)))
                        ],
                    'covariance':[
                        'target:    {}'.format(     act_cov.reshape(np.prod(act_cov.shape))),
                        'empirical  {}'.format(     cov.reshape(np.prod(cov.shape))),
                        'tolerance  {}'.format(     cov_tol.reshape(np.prod(cov_tol.shape)))
                        ]
                    },
                minimal = minimal)
        
        def plotPath(burn_in, samples, cov, mean, save=save):
            """Note that samples and burn_in contain the initial conditions"""
            
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
            
            fig.suptitle(r'Sampling Multivariate Gaussian with HMC',
                fontsize=self.s*self.ttfont)
            ax.set_title(r'Showing the first 50 HMC moves for:\ $\mu=\begin{pmatrix}0 & 0\end{pmatrix}$, $\Sigma = \begin{pmatrix} 1.0 & 0.8\\ 0.8 & 1.0 \end{pmatrix}$',
                fontsize=(self.tfont-4)*self.s)
            
            plt.grid(True)
            
            if save:
                save_dir = PLOT_LOC + 'plots/'
                subprocess.call(['mkdir', PLOT_LOC + 'plots/'])
            
                fig.savefig(save_dir+save)
            else:
                plt.show()
            pass
        
        if save:
            if save == 'plot':
                plotPath(burn_in, samples, cov, mean, save = False)
            else:
                plotPath(burn_in, samples, cov, mean, save = save)
        
        return passed, burn_in, samples
    def hmcQho(self, n_samples = 10000, n_burn_in = 50, tol = 5e-2, print_out = True, save = 'HMC_qho_1d.png'):
        """A test to sample the Quantum Harmonic Oscillator
        
        Optional Inputs
            tol     ::  float   :: tolerance level allowed
            print_out   :: bool     :: print results to screen
            save    :: string   :: file to save plot. False or '' gives no plot
        """
        passed = True
        
        x_nd = np.random.random((n,)*dim)
        p0 = np.random.random((n,)*dim)
        x0 = Periodic_Lattice(array=copy(self.x_nd), spacing=spacing)
        
        self.lf.duE = self.qho.duE # reassign leapfrog gradient
        self.lf.lattice = True
        
        self.hmc.__init__(x0, self.lf, self.sho, self.rng)
        
        act_mean = self.hmc.potential.mean
        act_cov = self.hmc.potential.cov
        
        p_samples, samples = self.hmc.sample(n_samples = n_samples, n_burn_in = n_burn_in)
        burn_in, samples = samples # return the shape: (n, dim, 1)
        
        print np.asarray(p_samples).shape
        print np.asarray(samples).shape
        print np.asarray(burn_in).shape
        # # flatten last dimension to a shape of (n, dim)
        # samples = np.asarray(samples).T.reshape(dim, -1).T
        # burn_in = np.asarray(burn_in).T.reshape(dim, -1).T
        #
        # mean = samples.mean(axis=0)
        # # covariance assumes observations in columns
        # # we have observations in rows so specify rowvar=0
        # cov = np.cov(samples, rowvar=0)
        #
        # passed *= (np.abs(mean - act_mean) <= mean_tol).all()
        # passed *= (np.abs(cov - act_cov) <= cov_tol).all()
        #
        # if print_out:
        #     minimal = (print_out == 'minimal')
        #     utils.display("HMC: Simple Harmonic Oscillator", passed,
        #         details = {
        #             'mean':[
        #                 'target:    {}'.format(     act_mean),
        #                 'empirical  {}'.format(     mean),
        #                 'tolerance  {}'.format(     mean_tol)
        #                 ],
        #             'covariance':[
        #                 'target:    {}'.format(     act_cov ),
        #                 'empirical  {}'.format(     cov),
        #                 'tolerance  {}'.format(     cov_tol)
        #                 ]
        #             },
        #         minimal = minimal)
        #
        # def plotPath(burn_in, samples, save=save):
        #     """Note that samples and burn_in contain the initial conditions"""
        #     self._teXify() # LaTeX
        #     self.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
        #     self._updateRC()
        #
        #     fig = plt.figure(figsize = (8*self.s, 8*self.s)) # make plot
        #     ax =[]
        #     ax.append(fig.add_subplot(111))
        #     fig.suptitle(r'Example HMC path sampling the SHO potential',
        #         fontsize=16)
        #     ax[0].set_title(
        #         r'{} Burn-in Samples shown in orange'.format(burn_in.shape[0]))
        #     ax[0].set_ylabel(r'Sample, $n$')
        #     ax[0].set_xlabel(r"Position, $x$")
        #
        #     offst = burn_in.shape[0]+1 # burn-in samples
        #     ax[0].plot(burn_in, np.arange(1, offst), #marker='x',
        #         linestyle='-', color='orange', label=r'Burn In')
        #     ax[0].plot(samples, np.arange(offst, offst + samples.shape[0]), #marker='x',
        #         linestyle='-', color='blue', label=r'Sampling')
        #
        #     if save:
        #         save_dir = PLOT_LOC + 'plots/'
        #         subprocess.call(['mkdir', PLOT_LOC + 'plots/'])
        #         save = save.split('.')
        #         save = save[0] + '_path' + '.' + save[1] # add identifier
        #         fig.savefig(save_dir+save)
        #     else:
        #         plt.show()
        #     pass
        # def plotPot(samples, save=save):
        #     """Note that samples and burn_in contain the initial conditions"""
        #     self._teXify() # LaTeX
        #     self.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
        #     self._updateRC()
        #
        #     n = 100 # size of linear space
        #     x = np.linspace(-5,5,n)
        #
        #     fig = plt.figure(figsize = (8*self.s, 8*self.s)) # make plot
        #     ax =[]
        #     ax.append(fig.add_subplot(111))
        #     fig.suptitle(r'Sampled SHO Potential',
        #         fontsize=16)
        #     ax[0].set_title(
        #         r'{} HMC samples. True potential in Blue.'.format(samples.shape[0]))
        #     ax[0].set_ylabel(r'Sampled Potential, $e^{-V(x)}$')
        #     ax[0].set_xlabel(r"Position, $x$")
        #
        #     # fitted normal dist. parameters p[0] = mean, p[1] = stdev
        #     p = norm.fit(samples)
        #     fitted = norm.pdf(x, loc=p[0], scale=p[1])
        #     actual = norm.pdf(x)
        #
        #     n, bins, patches = ax[0].hist(samples, 50, normed=1, # histogram
        #         facecolor='green', alpha=0.5, label=r'Sampled Data')
        #
        #     ax[0].plot(x, fitted, # marker='x', # best fit
        #         linestyle='-', color='orange', label=r'Fitted Potential')
        #
        #     ax[0].plot(x, actual, # marker='x',
        #         linestyle='-', color='blue', label=r'True Potential')
        #
        #     if save:
        #         save_dir = PLOT_LOC + 'plots/'
        #         subprocess.call(['mkdir', PLOT_LOC + 'plots/'])
        #
        #         save = save.split('.')
        #         save = save[0] + '_pot' + '.' + save[1] # add identifier
        #         fig.savefig(save_dir+save)
        #     else:
        #         plt.show()
        #     pass
        #
        #
        # if save:
        #     # burn_in includes initial cond.
        #     # samples inclues final burn_in as initial cond.
        #     burn_in = np.asarray(burn_in).reshape(n_burn_in+1)
        #     samples = np.asarray(samples).reshape(n_samples+1)
        #
        # if save:
        #     if save == 'plot':
        #         plotPath(burn_in[:50], samples=np.asarray([]), save = False)
        #         plotPot(samples, save = False)
        #     else:
        #         plotPath(burn_in[:50], samples=np.asarray([]), save = save)
        #         plotPot(samples, save = save)
        #
        # return passed, burn_in, samples
    
#
if __name__ == '__main__':
    utils.newTest(TEST_ID)
    rng = np.random.RandomState(1234)
    
    test = Test(rng)
    # test.hmcSho1d(n_samples = 10000, n_burn_in = 1000,
    #     tol = 5e-2,
    #     print_out = True,
    #     # save = 'plot'
    #     save = False,
    #     # save = 'HMC_oscillator_1d.png'
    #     )[0]
    # test.hmcGaus2d(n_samples = 10000, n_burn_in = 50,
    #     tol = 5e-2,
    #     print_out = True,
    #     # save = 'plot'
    #     save = False,
    #     # save = 'HMC_gauss_2d.png'
    #     )[0]
    test.hmcQho(n_samples = 100, n_burn_in = 50,
        tol = 5e-2,
        print_out = True,
        # save = 'plot'
        save = False,
        # save = 'HMC_gauss_2d.png'
        )[0]