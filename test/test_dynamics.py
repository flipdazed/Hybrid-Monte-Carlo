import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from matplotlib.colors import LogNorm
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

TEST_ID     = 'dynamics'

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
            ax[0].set_title(r'Potential: {}, Tolerance level: {}'.format(self.pot.name, tol),
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
            # ax[0].axhline(tol, color='red', linestyle='--')
            
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
            fig.suptitle(r'Magnitude of Change in Phase Space, $\Delta\mathcal{P}(x,p)$',
                fontsize=self.ttfont)
            ax[0].set_title(r'Potential: {}, Tolerance level: {}'.format(self.pot.name, tol),
                fontsize=self.ttfont-4)
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
        
        if len(uE0) > 1:
            check_uE0 = uE0[0]
        else:
            check_uE0 = uE0
        
        checks.tryAssertEqual(h_old, kE0+check_uE0,
             ' kin: {}, pot:{}, h:{}'.format(kE0, check_uE0, h_old) \
             +'\n Diff: {}'.format(h_old - kE0+check_uE0)
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
            ax[0].set_title(r'Potential: {}, Tolerance level: {}'.format(self.pot.name, tol),
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
            # ax[0].axhline(tol, color='red', linestyle='--')
            
            if save:
                save_dir = PLOT_LOC + 'plots/'
                subprocess.call(['mkdir', PLOT_LOC + 'plots/'])
            
                fig.savefig(save_dir+save)
            else:
                plt.show()
            pass
        
        def plot1d(x = step_sample, y1 = kins, y2 = pots, save = save, all_lines=False):
            self._teXify() # LaTeX
            self.params['text.latex.preamble'] = r"\usepackage{amsmath}"
            self._updateRC()
            
            fig = plt.figure(figsize=(8*self.s,8*self.s)) # make plot
            ax =[]
            ax.append(fig.add_subplot(111))
            
            fig.suptitle(r"Components of Hamiltonian, $H'$, during Leap Frog integration",
                fontsize=self.ttfont)
            ax[0].set_title(r'Potential: {}, Tolerance level: {}'.format(self.pot.name, tol),
                fontsize=self.ttfont-4)
            ax[0].set_xlabel(r'Number of Integrator Steps, $n$')
            ax[0].set_ylabel(r'Energy')
            
            steps = np.linspace(0, n_steps_i+1, n_steps_i+2, True)
            action, k, u = zip(*y2)
            
            action = np.asarray(action)
            k = np.asarray(k)
            u = np.asarray(u)
            
            try:
                h = ax[0].plot(steps, y1+np.asarray(action), label=r"$H' = T(\pi) + S(x,t)$",
                    color='blue', linewidth=5., alpha=0.2)
                if all_lines:
                    kE = ax[0].plot(steps, np.asarray(y1), label=r'$T(\pi)$',
                        color='darkred', linewidth=3., linestyle='-', alpha=0.2)
                    uE = ax[0].plot(steps, np.asarray(action), label=r'$S(x,t) = \sum_{n} (T_S + V_S)$',
                        color='darkgreen', linewidth=3., linestyle='-', alpha=0.2)
                    uE = ax[0].plot(steps, np.asarray(k), label=r'$\sum_{n} T_S$',
                        color='red', linestyle='--')
                    uE = ax[0].plot(steps, np.asarray(u), label=r'$\sum_{n} V_S$',
                        color='green', linestyle='--')
                ax[0].legend(loc='upper left', shadow=True, fontsize = self.axfont)
                
            except Exception as e:
                print "\n\n___Shape___"
                print "conj mom: %s" % y1.shape
                print "act: %s,  k: %s, u: %s" % (action.shape, k.shape, u.shape)
                print "steps: %s" % steps.shape
                print "___________\n\n"
                raise e
            
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
                if len(step_sample) > 1:
                    plot2d(save=False)
                else:
                    plot1d(save=False,
                        all_lines=True
                        )
            else:
                if len(step_sample) > 1:
                    plot2d(save=save)
                else:
                    plot1d(save=save,
                        all_lines=True
                        )
        
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
            fig.suptitle(r'Magnitude of Change in Phase Space, $\Delta\mathcal{P}(x,p)$',
                fontsize=self.ttfont)
            ax[0].set_title(r'Potential: {}, Tolerance level: {}'.format(self.pot.name, tol),
                fontsize=self.ttfont-4)
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
            
        return passed
#
class Test(object):
    """this is a wrapper for the majority of the tests above so that they
    can be run with just a call to this file.
    
    Optional Inputs
        tol         :: float    :: tolerance of all tests
        n_steps     :: float    :: number of integration steps
        step_size   :: float    :: step size
        save_name   :: string   :: the identifier name for saving
    
    Note: To save specify at the function level
    """
    def __init__(self, tol=1e-2, n_steps=100, step_size=0.1, save_name='energy'):
        self.save_name = save_name
        self.tol = tol
        self.n_steps = n_steps
        self.step_size = step_size
        
        self.integrator = Leap_Frog(
            duE = None, # grad set in test
            n_steps = 100,
            step_size = 0.1)
        
        
        pass
    
    def _save(self, save, save_name):
        """
        Parses the name to save the file
        
        Required Inputs
            save        :: string/bool :: see below
            save_name   :: string      :: the name to save as if saving
        
        Notes:
            save :: 'plot' plots to screen. False is no plot. True saves a file.
        """
        if save == 'plot':
            return save
        elif save:
            return save_name
        elif save == False:
            return False
        else:
            checks.tryAssertEqual(False, save,
                 ' save error error' \
                 + '\nsave: {}'.format(save))
            
    
    def continuum(self, test_type = 'Continuum', save='plot'):
        """
        Runs tests:
            constantEnergy
            reversibility (step-size vs. )
        for:
            SHO
        
        Optional Inputs
            test_type   :: string :: identifier for screen output
            save        :: string/bool :: see below
        
        Notes:
            save :: 'plot' plots to screen. False is no plot. True saves a file.
        """
        self.integrator.lattice = False
        self.integrator.save_path = True
        # initial x,p
        self.p_1d, self.x_1d = np.asarray([[4.]]), np.asarray([[1.]])
        
        self.sho = Simple_Harmonic_Oscillator(k = 1.)
        potential = self.sho
        tests = Continuum(dynamics = self.integrator, pot=potential)
        
        utils.newTest(TEST_ID + ": " + test_type + ": " + potential.name)
        
        self.integrator.save_path = False
        save_name = self.save_name + '_conservation_{}.png'.format(potential.name)
        tests.constantEnergy(p0 = copy(self.p_1d), x0 = copy(self.x_1d),
            tol = self.tol,
            step_sample = np.linspace(1, self.n_steps*2, 100, True, dtype=int),
            step_sizes = np.linspace(0.001, self.step_size*2, 100, True),
            save = self._save(save, save_name))
        
        save_name = self.save_name + '_reversibility_{}.png'.format(potential.name)
        tests.reversibility(p0 = copy(self.p_1d), x0 = copy(self.x_1d),
            steps = self.n_steps,
            tol = self.tol,
            save = self._save(save, save_name))
    
    def lattice(self, n=10, dim=1, spacing=1, test_type = 'Lattice', save='plot'):
        """
        Runs tests:
            constantEnergy
            reversibility
        for:
            Klein Gordon
            QHO
        
        Optional Inputs
            n           :: int      :: number of lattice sites
            dim         :: int      :: number of dimensions
            spacing     :: float    :: lattice spacing
            test_type   :: string :: identifier for screen output
            save        :: string/bool :: see below
        
        Notes:
            save :: 'plot' plots to screen. False is no plot. True saves a file.
        
        """
        self.integrator.lattice = True
        self.integrator.save_path = True
        
        self.x_nd = np.random.random((n,)*dim)
        self.p0 = np.random.random((n,)*dim)
        self.x0 = Periodic_Lattice(array=copy(self.x_nd), spacing=spacing)
        
        self.qho = Quantum_Harmonic_Oscillator(debug=True)
        self.kg = Klein_Gordon(debug=True)
        
        for potential in [self.kg, self.qho]:
            utils.newTest(TEST_ID + ": " + test_type + ": " + potential.name)
            
            tests = Lattice(dynamics = self.integrator, pot=potential)
            save_name = self.save_name + '_conservation_1d_{}.png'.format(potential.name)
            tests.constantEnergy(
                p0 = copy(self.p0),
                x0 = copy(self.x0),
                tol = self.tol,
                step_sample = [self.n_steps],
                step_sizes = [self.step_size],
                save=self._save(save, save_name))
            
            # save_name = self.save_name + '_conservation_2d_{}.png'.format(potential.name)
            # tests.constantEnergy(
            #     p0 = copy(self.p0),
            #     x0 = copy(self.x0),
            #     tol = self.tol,
            #     step_sample = np.linspace(1, self.n_steps*2, 50, True, dtype=int),
            #     step_sizes = np.linspace(0.001, self.step_size*2, 50, True),
            #     save=self._save(save, save_name))
            
            save_name = self.save_name + '_reversibility_{}.png'.format(potential.name)
            tests.reversibility(
                p0 = copy(self.p0),
                x0 = copy(self.x0),
                steps = self.n_steps,
                tol = self.tol,
                save = self._save(save, save_name))
        pass
        
#
if __name__ == '__main__':
    
    dim         = 1
    n           = 10
    spacing     = 1.
    step_size   = .01
    n_steps     = 500
    
    test = Test(n_steps=n_steps, step_size=step_size)
    
    # test.continuum(save=True)
    test.lattice(save=True, dim=dim, n=n, spacing=spacing)