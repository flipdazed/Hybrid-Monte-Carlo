import numpy as np
import matplotlib.pyplot as plt
import subprocess

import test_utils

# these directories won't work unless 
# the commandline interface for python unittest is used
from hmc.lattice import Periodic_Lattice
from hmc.potentials import Multivariate_Gaussian, Quantum_Harmonic_Oscillator
from plotter import Pretty_Plotter, viridis, magma, inferno, plasma, PLOT_LOC

TEST_ID = 'potentials'

class Test(Pretty_Plotter):
    def __init__(self):
        self.mean = np.asarray([[0.], [0.]])
        self.cov = np.asarray([[1.0,0.8],[0.8,1.0]])
        self.bg = Multivariate_Gaussian(mean = self.mean, cov = self.cov)
        pass
    
    def testBG(self, save = 'potentials_Gaussian_2d.png', print_out = True):
        """Plots a test image of the Bivariate Gaussian"""
        passed = True
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
        
        if print_out:
            minimal = (print_out == 'minimal')
            test_utils.display('Bivariate Gaussian Potential', passed,
                details = {
                    'Not a unit test':[]
                    },
                minimal = minimal)
        
        def plot(save=save):
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
                save_dir = PLOT_LOC + 'plots/'
                subprocess.call(['mkdir', PLOT_LOC + 'plots/'])
            
                fig.savefig(save_dir+save)
            else:
                plt.show()
            pass
        
        if save:
            if save == 'plot':
                plot(save=False)
            elif save:
                plot(save=save)
        
        return passed
    def testQHO(self, dim = 4, sites = 10, spacing = 1., save = False, print_out = True):
        """checks that QHO can be initialised and all functions run"""
        np.set_printoptions(suppress=True)
        
        passed = True
        shape = (sites,)*dim
        raw_lattice = np.arange(sites**dim).reshape(shape)
        self.lattice = Periodic_Lattice(array = raw_lattice, spacing = 1.)
        self.qho = Quantum_Harmonic_Oscillator(self.lattice)
        
        pot_energy = self.qho.potentialEnergy()
        gradient_i = self.qho.gradPotentialEnergy((0,)*dim)
        gradient_f = self.qho.gradPotentialEnergy((sites-1,)*dim)
        
        if print_out:
            minimal = (print_out == 'minimal')
            test_utils.display('QHO Potential', passed,
                details = {
                    'Not a unit test':[],
                    'Gradient':[
                        '{}: {}'.format((0,)*dim, gradient_i),
                        '{}: {}'.format((sites-1,)*dim, gradient_f)],
                    'Potential Energy: {}'.format(pot_energy):[]
                    },
                minimal = minimal)
        return passed
#
if __name__ == '__main__':
    test_utils.newTest(TEST_ID)
    test = Test()
    test.testBG(
            save = False
            # save = 'plot'
            # save = 'potentials_Gaussian_2d.png'
            )
    test.testQHO()
    
