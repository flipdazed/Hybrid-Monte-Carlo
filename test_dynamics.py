import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import subprocess

import utils

# these directories won't work unless 
# the commandline interface for python unittest is used
from hmc.dynamics import Leap_Frog
from hmc.potentials import Simple_Harmonic_Oscillator
from plotter import Pretty_Plotter

class Tests(Pretty_Plotter):
    """Tests energy conservation"""
    def __init__(self, dynamics):
        self.pot = Simple_Harmonic_Oscillator(k=1.)
        self.dynamics = dynamics
        self.dynamics.duE = self.pot.duE
        pass
    def constantEnergy(self, step_sample, step_sizes, tol = 1e-2, print_out = True, save = 'energy_conservation.png'):
        """Checks that the change in hamiltonian ~0
        for varying step_sizes and step_lengths
        
        Can also plot a pretty 2d contour plot
        
        Required Inputs
            step_sample :: np.array :: array of steps lengths to test
            step_sizes  :: np.array :: array of step sizes to test
        
        Optional Inputs
            tol         :: float    :: tolerance level for hamiltonian changes
            save        :: string   :: file to save plot. False or '' gives no plot
            print_out   :: bool     :: if True prints to screen
        
        Returns
            passed :: bool :: True if passed
        """
        passed = True
        diffs = []
        
        # calculate original hamiltonian and set starting vals
        pi,xi = np.asarray([[4.]]), np.asarray([[1.]])
        h_old = self.pot.hamiltonian(pi, xi)
        
        # initial vals required to print out values associated
        # with the worst absolute deviation from perfect energy conservation
        # (0 = no energy loss)
        w_bmk = 1.  # worst benchmark value
        diff = 0.   # absolute difference
        w_step = 0  # worst steps
        w_size = 0  # worst step size
        
        # set up a mesh grid of the steps and sizes
        step_sample, step_sizes = np.meshgrid(step_sample, step_sizes)
        
        for n_steps_i, step_size_i in zip(np.ravel(step_sample), np.ravel(step_sizes)):
            
            # set new parameters
            self.dynamics.n_steps = n_steps_i
            self.dynamics.step_size = step_size_i
            
            # obtain new duynamics and resultant hamiltonian
            pf,xf = self.dynamics.integrate(pi,xi)
            h_new = self.pot.hamiltonian(pf,xf)
            
            bench_mark = np.exp(-(h_old-h_new))
            
            # stores the worst for printing to terminal
            if print_out:
                # avoid calc every time when no print out
                if (np.abs(1. - bench_mark) > diff): # compare to last diff
                    w_bmk = bench_mark
                    w_h_new = h_new
                    w_step = n_steps_i
                    w_size = step_size_i
            
            diff = np.abs(1. - bench_mark) # set new diff
            
            passed *= (diff <= tol).all()
            diffs.append(diff) # append to list for plotting
        
        if print_out:
            utils.display(test_name='Constant Energy', outcome=passed,
                details = {
                    'initial H(p, x): {}'.format(h_old):[],
                    'worst   H(p, x): {}'.format(h_new):[
                            'steps: {}'.format(w_step),
                            'step size: {}'.format(w_size)],
                    'np.abs(exp(-dH): {}'.format(w_h_new - h_old, w_bmk):[]
                },
                minimal=False)
        
        def plot(x = step_sample, y = step_sizes, z = diffs, save = save):
            self._teXify() # LaTeX
            fig = plt.figure(figsize=(8*self.s,8*self.s)) # make plot
            ax =[]
            ax.append(fig.add_subplot(111))
            
            
            fig.suptitle(r'Energy Drift as a function of Integrator Parameters', 
                fontsize=self.ttfont)
            ax[0].set_title(r'Potential:SHO, tolerance level: {}'.format(tol),
                fontsize=self.ttfont-4)
            ax[0].set_xlabel(r'Number of Integrator Steps, $n$')
            ax[0].set_ylabel(r'Integrator Step Size, $\epsilon$')
            
            z = np.asarray(z).reshape(*x.shape)
            p = ax[0].contourf(x, y, z, 500)
            
            # add colorbar and label
            cbar = plt.colorbar(p, ax=ax[0], shrink=0.9)
            cbar.ax.set_ylabel(r'Absolute change in Hamiltonian, $|{1 - e^{-\delta H(p,x)}}|$')
            
            # ax[0].plot(step_sample, np.asarray(diffs), linestyle='-', color='blue')
            # ax[0].plot(step_sample, np.full(step_sample.shape, tol),
            # linestyle='--', color='red', label='tolerance')
            ax[0].axhline(tol, color='red', linestyle='--')
            
            if save:
                save_dir = '../results/plots/'
                subprocess.call(['mkdir', '../results/plots/'])
            
                fig.savefig(save_dir+save)
            else:
                plt.show()
            pass
        
        if save:
            if save == 'plot':
                plot(save=False)
            else:
                plot(save=save)
        
        return passed
    
    def reversibility(self, tol = 1e-3, steps = 2000, print_out = True, save = 'reversibility.png'):
        """Checks the integrator is reversible
        by running and then reversing the integration and 
        verifying the same point in phase space
        
        Optional Inputs
            tol         :: float    :: tolerance level for hamiltonian changes
            steps       :: integer  :: number of integration steps
            save        :: string   :: file to save plot. False or '' gives no plot
            print_out   :: bool     :: if True prints to screen
        """
        passed = True
        
        # params and ensure dynamic params correct
        p0, x0 = np.asarray([[4.]]), np.asarray([[1.]])
        self.dynamics.n_steps = steps
        self.dynamics.step_size = 0.1
        self.dynamics.save_path = True
        
        pf, xf = self.dynamics.integrate(p0, x0)
        p0f, x0f = self.dynamics.integrate(-pf, xf) # time flip
        
        p0f = -p0f # time flip to point in right time again
        
        phase_change = np.linalg.norm( # calculate frobenius norm
            np.asarray([[p0f], [x0f]]) - np.asarray([[p0], [x0]])
            )
        passed = (phase_change < tol)
        if print_out: 
            utils.display(test_name="Reversibility of Integrator", 
            outcome=passed,
            details={
                'initial (p, x): ({}, {})'.format(p0, x0):[],
                'middle  (p, x): ({}, {})'.format(pf, xf):[],
                'final   (p, x): ({}, {})'.format(p0f, x0f):[],
                'phase change:    {}'.format(phase_change):[],
                'number of steps: {}'.format(self.dynamics.n_steps):[]
                },
            minimal=False)
        
        def plot(steps, norm, save=save):
            
            self._teXify() # LaTeX
            self.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
            self._updateRC()
            
            fig = plt.figure(figsize=(8*self.s,8*self.s)) # make plot
            ax =[]
            ax.append(fig.add_subplot(111))
            # fig.suptitle(r"Hamiltonian Dynamics of the SHO using Leap-Frog Integrator",
            #     fontsize=16)
            ax[0].set_title(r'Magnitude of Change in Phase Space, $\Delta\mathcal{P}(x,p)$')
            ax[0].set_xlabel(r'Integration Step, $n$')
            ax[0].set_ylabel(r"$|\Delta\mathcal{P}(x,p)| = \sqrt{(p_{t} + p_{\text{-}t})^2 + (x_{t} - x_{\text{-}t})^2}$")
            
            ax[0].plot(steps, norm, #marker='x',
                linestyle='-', color='blue')
            # ax[0].plot(-p[p.shape[0]//2 : ], x[x.shape[0]//2 : ],
            #     linestyle='-', color='red', marker='+')
            
            if save:
                save_dir = '../results/plots/'
                subprocess.call(['mkdir', '../results/plots/'])
            
                fig.savefig(save_dir+save)
            else:
                plt.show()
            pass
        if save:
            p_path, x_path = self.dynamics.p_ar, self.dynamics.x_ar
            p, x = np.asarray(p_path), np.asarray(x_path)
            
            # curious why I need to clip one step on each?
            # something to do with the way I sample the steps...
            # clip last step on forward and first step on backwards
            # solved!... because I didn't save the zeroth step in the integrator
            # integrator nowsaves zeroth steps
            change_p = (-p[ : p.shape[0]//2] - p[p.shape[0]//2 : ][::-1])**2
            change_x = (x[ : x.shape[0]//2] - x[x.shape[0]//2 : ][::-1])**2
            
            norm = np.sqrt(change_p + change_x)
            steps = np.linspace(0, steps, steps+1, True)
            
            if save == 'plot':
                plot(steps, norm[:,0], save=False)
            else:
                plot(steps, norm[:,0])
            
            self.dynamics.save_path = False
            
        return passed

#
if __name__ == '__main__':
    integrator = Leap_Frog(duE = None, n_steps = 100, step_size = 0.1) # grad set in test
    tests = Tests(dynamics = integrator)
    
    ##### 'save' option details
    # Comment out all to save image
    # 'plot' plots to screen
    # False = do nothing
    #####
    r1 = tests.constantEnergy(
        tol = 0.05,
        step_sample = np.linspace(1, 100, 10, True, dtype=int),
        step_sizes = np.linspace(0.01, 0.1, 5, True),
        save = False,
        # save='plot',
        print_out = True # shows a small print out
        )
    r2 = tests.reversibility(
        steps = 1000,
        tol = 0.01,
        save = False,
        # save = 'plot',
        print_out = True # shows a small print out)
        )