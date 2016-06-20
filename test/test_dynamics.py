import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
import subprocess
from copy import copy

import utils
from hmc import checks

# these directories won't work unless 
# the commandline interface for python unittest is used
from hmc.dynamics import Leap_Frog
from hmc.lattice import Periodic_Lattice
from hmc.potentials import Simple_Harmonic_Oscillator
from hmc.potentials import Quantum_Harmonic_Oscillator, Klein_Gordon
from plotter import Pretty_Plotter, PLOT_LOC

from hmc.lattice import Periodic_Lattice

TEST_ID_Continuum   = 'Continuum dynamics'
TEST_ID_Lattice     = 'Lattice dynamics'

class Continuum(Pretty_Plotter):
    """Tests energy conservation for Continuum"""
    def __init__(self, dynamics, pot):
        self.pot = pot
        self.dynamics = dynamics
        self.dynamics.duE = self.pot.duE
        pass
    
    def constantEnergy(self, p0, x0, step_sample, step_sizes, tol = 1e-2, print_out = True, save = 'energy_conservation_continuum.png'):
        """Checks that the change in hamiltonian ~0
        for varying step_sizes and step_lengths
        
        Can also plot a pretty 2d contour plot
        
        Required Inputs
            p0          :: lattice/np :: momentum
            x0          :: lattice/np :: momentum
            step_sample :: np.array   :: array of steps lengths to test
            step_sizes  :: np.array   :: array of step sizes to test
        
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
        h_old = self.pot.hamiltonian(p0, x0)
        
        # initial vals required to print out values associated
        # with the worst absolute deviation from perfect energy conservation
        # (0 = no energy loss)
        w_bmk  = 1.  # worst benchmark value
        diff   = 0.  # absolute difference
        w_step = 0   # worst steps
        w_size = 0   # worst step size
        
        # set up a mesh grid of the steps and sizes
        step_sample, step_sizes = np.meshgrid(step_sample, step_sizes)
        
        for n_steps_i, step_size_i in zip(np.ravel(step_sample), np.ravel(step_sizes)):
            
            # set new parameters
            self.dynamics.n_steps = n_steps_i
            self.dynamics.step_size = step_size_i
            print "     ... n_steps: {}, size: {}".format(n_steps_i, step_size_i)
            
            # obtain new duynamics and resultant hamiltonian
            self.dynamics.newPaths()
            pf, xf = self.dynamics.integrate(copy(p0), copy(x0))
            h_new = self.pot.hamiltonian(pf, xf)
            
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
                    'worst   H(p, x): {}'.format(w_h_new):[
                            'steps: {}'.format(w_step),
                            'step size: {}'.format(w_size)],
                    'np.abs(exp(-dH)): {}'.format(w_bmk):[]
                })
        
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
                save_dir = PLOT_LOC + 'plots/'
                subprocess.call(['mkdir', PLOT_LOC + 'plots/'])
            
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
    
    def reversibility(self, p0, x0, tol = 1e-3, steps = 2000, print_out = True, save = 'reversibility_continuum.png'):
        """Checks the integrator is reversible
        by running and then reversing the integration and 
        verifying the same point in phase space
        
        Required Inputs
            p0          :: lattice/np :: momentum
            x0          :: lattice/np :: momentum
        
        Optional Inputs
            tol         :: float    :: tolerance level for hamiltonian changes
            steps       :: integer  :: number of integration steps
            save        :: string   :: file to save plot. False or '' gives no plot
            print_out   :: bool     :: if True prints to screen
        """
        passed = True
        
        # params and ensure dynamic params correct
        self.dynamics.n_steps = steps
        self.dynamics.step_size = 0.1
        self.dynamics.save_path = True
        
        self.dynamics.newPaths()
        pm, xm = self.dynamics.integrate(copy(p0), copy(x0))
        p0f, x0f = self.dynamics.integrate(-pm, xm) # time flip
        
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
                'middle  (p, x): ({}, {})'.format(pm, xm):[],
                'final   (p, x): ({}, {})'.format(p0f, x0f):[],
                'phase change:    {}'.format(phase_change):[],
                'number of steps: {}'.format(self.dynamics.n_steps):[]
                })
        
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
                save_dir = PLOT_LOC + 'plots/'
                subprocess.call(['mkdir', PLOT_LOC + 'plots/'])
            
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
class Lattice(Pretty_Plotter):
    """Tests energy conservation for Lattice"""
    def __init__(self, dynamics, pot):
        self.pot = pot
        self.dynamics = dynamics
        self.dynamics.duE = self.pot.duE
        pass
    
    def constantEnergy(self, p0, x0, step_sample, step_sizes, tol = 1e-2, print_out = True, save = 'energy_conservation_lattice.png'):
        """Checks that the change in hamiltonian ~0
        for varying step_sizes and step_lengths
        
        Can also plot a pretty 2d contour plot
        
        Required Inputs
            p0          :: lattice/np :: momentum
            x0          :: lattice/np :: momentum
            step_sample :: np.array   :: array of steps lengths to test
            step_sizes  :: np.array   :: array of step sizes to test
        
        Optional Inputs
            tol         :: float    :: tolerance level for hamiltonian changes
            save        :: string   :: file to save plot. False or '' gives no plot
            print_out   :: bool     :: if True prints to screen
        
        Returns
            passed :: bool :: True if passed
        """
        passed = True
        diffs = []
        kins  = []
        pots  = []
        
        # calculate original hamiltonian and set starting vals
        h_old = self.pot.hamiltonian(p0, x0)
        kE0 = self.pot.kE(p0)
        uE0 = self.pot.uE(x0)
        
        checks.tryAssertEqual(h_old, kE0+uE0[0],
             ' kin: {}, pot:{}, h:{}'.format(kE0, uE0[0], h_old) \
             +'\n Diff: {}'.format(h_old - kE0+uE0[0])
             )
        
        pots.append(uE0)
        kins.append(kE0)
        # initial vals required to print out values associated
        # with the worst absolute deviation from perfect energy conservation
        # (0 = no energy loss)
        w_bmk  = 1.  # worst benchmark value
        diff   = 0.  # absolute difference
        w_step = 0   # worst steps
        w_size = 0   # worst step size
        
        # set up a mesh grid of the steps and sizes
        step_sample, step_sizes = np.meshgrid(step_sample, step_sizes)
        
        for n_steps_i, step_size_i in zip(np.ravel(step_sample), np.ravel(step_sizes)):
            
            # set new parameters
            self.dynamics.n_steps = n_steps_i
            self.dynamics.step_size = step_size_i
            
            # obtain new duynamics and resultant hamiltonian
            self.dynamics.newPaths()
            pf, xf = self.dynamics.integrate(copy(p0), copy(x0))
            
            h_new = self.pot.hamiltonian(pf, xf)
            
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
            
            if save:
                kE_path = [self.pot.kE(i) for i in self.dynamics.p_ar]
                uE_path = []
                for i in self.dynamics.x_ar:
                    x0.get = i
                    uE_path.append(self.pot.uE(x0))
                    
                # print np.asarray([i for i in self.dynamics.x_ar])
                # print np.asarray(uE_path)
            kins = np.asarray([kE0] + kE_path)
            pots = [uE0] + uE_path
            
            
        if print_out:
            minimal = (print_out == 'minimal')
            utils.display(test_name='Constant Energy', outcome=passed,
                details = {
                    'initial H(p, x): {}'.format(h_old):[],
                    'worst   H(p, x): {}'.format(w_h_new):[
                            'steps: {}'.format(w_step),
                            'step size: {}'.format(w_size)],
                    'np.abs(exp(-dH): {}'.format(w_h_new - h_old, w_bmk):[]
                },
                minimal=minimal)
        
        def plot2d(x = step_sample, y = step_sizes, z = diffs, save = save):
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
            p = ax[0].contourf(x, y, z, 200,
                norm=LogNorm(vmin=z.min(), vmax=z.max()))
            
            # add colorbar and label
            cbar = plt.colorbar(p, ax=ax[0], shrink=0.9)
            cbar.ax.set_ylabel(r'Absolute change in Hamiltonian, $|{1 - e^{-\delta H(p,x)}}|$')
            
            # ax[0].plot(step_sample, np.asarray(diffs), linestyle='-', color='blue')
            # ax[0].plot(step_sample, np.full(step_sample.shape, tol),
            # linestyle='--', color='red', label='tolerance')
            ax[0].axhline(tol, color='red', linestyle='--')
            
            if save:
                save_dir = PLOT_LOC + 'plots/'
                subprocess.call(['mkdir', PLOT_LOC + 'plots/'])
            
                fig.savefig(save_dir+save)
            else:
                plt.show()
            pass
        def plot1d(x = step_sample, y1 = kins, y2 = pots, save = save):
            self._teXify() # LaTeX
            self.params['text.latex.preamble'] = r"\usepackage{amsmath}"
            self._updateRC()
            fig = plt.figure(figsize=(8*self.s,8*self.s)) # make plot
            ax =[]
            ax.append(fig.add_subplot(111))
            
            fig.suptitle(r'Increasing KE, $\sum T_S$, in the Action with Leap Frog moves', 
                fontsize=self.ttfont)
            ax[0].set_title(r'Potential: {}'.format(self.pot.name),
                fontsize=self.ttfont-4)
            ax[0].set_xlabel(r'Number of Integrator Steps, $n$')
            ax[0].set_ylabel(r'Energy')
            
            steps = np.linspace(0, n_steps_i+1, n_steps_i+2, True)
            action, k, u = zip(*y2)
            
            h = ax[0].plot(steps, y1+np.asarray(action), label=r"$H' = T(\pi) + S(x,t)$",
                color='blue', linewidth=5., alpha=0.2)
            kE = ax[0].plot(steps, np.asarray(y1), label=r'$T(\pi)$',
                color='darkred', linewidth=3., linestyle='-', alpha=0.2)
            uE = ax[0].plot(steps, np.asarray(action), label=r'$S(x,t) = \sum_{n} (T_S + V_S)$',
                color='darkgreen', linewidth=3., linestyle='-', alpha=0.2)
            uE = ax[0].plot(steps, np.asarray(k), label=r'$\sum_{n} T_S$',
                color='red', linestyle='--')
            uE = ax[0].plot(steps, np.asarray(u), label=r'$\sum_{n} V_S$',
                color='green', linestyle='--')
            ax[0].legend(loc='upper left', shadow=True, fontsize = self.axfont)
            # ax[0].set_yscale("log", nonposx='clip')
            if save:
                save_dir = PLOT_LOC + 'plots/'
                subprocess.call(['mkdir', PLOT_LOC + 'plots/'])
            
                fig.savefig(save_dir+save)
            else:
                plt.show()
            pass
        
        if save:
            if save == 'plot':
                plot2d(save=False)
            else:
                plot2d(save=save)
        
        return passed
    
    def reversibility(self, p0, x0, tol = 1e-3, steps = 2000, print_out = True, save = 'reversibility_lattice.png'):
        """Checks the integrator is reversible
        by running and then reversing the integration and 
        verifying the same point in phase space
        
        Required Inputs
            p0          :: lattice/np :: momentum
            x0          :: lattice/np :: momentum
        
        Optional Inputs
            tol         :: float    :: tolerance level for hamiltonian changes
            steps       :: integer  :: number of integration steps
            save        :: string   :: file to save plot. False or '' gives no plot
            print_out   :: bool     :: if True prints to screen
        """
        passed = True
        
        # params and ensure dynamic params correct
        self.dynamics.n_steps = steps
        self.dynamics.step_size = 0.1
        self.dynamics.save_path = True
        
        self.dynamics.newPaths()
        pm, xm = self.dynamics.integrate(copy(p0), copy(x0))
        p0f, x0f = self.dynamics.integrate(-pm, xm) # time flip
        
        p0f = -p0f # time flip to point in right time again
        
        phase_change = np.linalg.norm( # calculate frobenius norm
            np.asarray([[p0f], [x0f.get]]) - np.asarray([[p0], [x0.get]])
            )
        passed = (phase_change < tol)
        if print_out: 
            utils.display(test_name="Reversibility of Integrator", 
            outcome=passed,
            details={
                'initial (p, x): ({}, {})'.format(p0, x0.get):[],
                'middle  (p, x): ({}, {})'.format(pm, xm):[],
                'final   (p, x): ({}, {})'.format(p0f, x0f.get):[],
                'phase change:    {}'.format(phase_change):[],
                'number of steps: {}'.format(self.dynamics.n_steps):[]
                })
        
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
                save_dir = PLOT_LOC + 'plots/'
                subprocess.call(['mkdir', PLOT_LOC + 'plots/'])
            
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
    # utils.newTest(TEST_ID_Continuum)
    #
    # integrator = Leap_Frog(duE = None, n_steps = 100, step_size = 0.1) # grad set in test
    # pot = Simple_Harmonic_Oscillator(k = 1.)
    # tests = Test(dynamics = integrator, pot=pot)
    # p, x = np.asarray([[4.]]), np.asarray([[1.]])
    #
    # ##### 'save' option details
    # # Comment out all to save image
    # # 'plot' plots to screen
    # # False = do nothing
    # #####
    # tests.constantEnergy(p0 = copy(p), x0 = copy(x),
    #     tol = 0.05,
    #     step_sample = np.linspace(1, 100, 10, True, dtype=int),
    #     step_sizes = np.linspace(0.01, 0.1, 5, True),
    #     save = False,
    #     # save='plot',
    #     print_out = True # shows a small print out
    #     )
    #
    # tests.reversibility(p0 = copy(p), x0 = copy(x),
    #     steps = 1000,
    #     tol = 0.01,
    #     save = False,
    #     # save = 'plot',
    #     print_out = True # shows a small print out)
    #     )
    
    utils.newTest(TEST_ID_Lattice)
    dim         = 1
    n           = 10
    spacing     = 1.
    step_size   = 0.01
    n_steps     = 600
    
    # check that change in step_size time:
    # ~1/(lattice time length)
    # cond = np.abs(1. -  n/spacing*step_size)
    # checks.tryAssertLtEqual(cond, .1,
    #      ' n:{}, a:{}, step_size:{}'.format(n, spacing, step_size) \
    #      +'\n Result: {}'.format(cond)
    #      )
    
    v = n**dim
    x = np.random.random((n,)*dim)
    p0 = np.random.random((n,)*dim)
    
    # x0 = Periodic_Lattice(array=copy(x), spacing=spacing)
    # print 'QHO'
    # qho = Quantum_Harmonic_Oscillator()
    # integrator = Leap_Frog(duE = None,
    #     n_steps = n_steps,
    #     step_size = step_size,
    #     lattice = True,
    #     save_path=True)
    #
    # tests = Lattice(dynamics = integrator, pot=qho)
    # tests.constantEnergy(p0 = copy(p0), x0 = copy(x0),
    #     tol = 0.05,
    #     step_sample = [n_steps],
    #     step_sizes = [step_size],
    #     # save = False,
    #     # save='plot',
    #     save = 'energy_cons_QHO_pot_bug.png',
    #     print_out = True # shows a small print out
    #     )
    
    print 'KG'
    x0 = Periodic_Lattice(array=copy(x), spacing=spacing)
    qho = Klein_Gordon()
    integrator = Leap_Frog(duE = None,
        n_steps = n_steps,
        step_size = step_size,
        lattice = True,
        save_path=True)
    
    tests = Lattice(dynamics = integrator, pot=qho)
    tests.constantEnergy(p0 = copy(p0), x0 = copy(x0),
        tol = 0.05,
        # step_sample = [n_steps],
        # step_sizes = [step_size],
        step_sample = np.linspace(1, 1000, 10, True, dtype=int),
        step_sizes = np.linspace(0.01, 0.1, 10, True),
        # save = False,
        # save='plot',
        save = 'energy_conservation_lattice_KG.png',
        print_out = True # shows a small print out
        )
    # tests.reversibility(p0 = copy(p), x0 = copy(x),
    #     steps = 100,
    #     tol = 0.01,
    #     save = False,
    #     # save = 'plot',
    #     print_out = True # shows a small print out)
    #     )