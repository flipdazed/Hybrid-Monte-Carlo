import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import subprocess
from copy import copy

from potentials import Simple_Harmonic_Oscillator
from pretty_plotting import Pretty_Plotter

class Leap_Frog(object):
    """Leap Frog Integrator
    
    Required Inputs
        duE  :: func :: Gradient of Potential Energy
    
    Optional Inputs
        d   :: integration step size
        l  :: leap frog integration steps (trajectory length)
    
    Note: Do not confuse x0,p0 with initial x0,p0 for HD
    """
    def __init__(self, duE, step_size = 0.1, n_steps = 250, save_path = False):
        self.step_size = step_size
        self.n_steps = n_steps
        self.duE = duE
        
        self.save_path = save_path
        self.p_ar = [] # data for plots
        self.x_ar = [] # data for plots
        pass
    
    def integrate(self, p0, x0):
        """The Leap Frog Integration
        
        Required Input
            p0  :: float :: initial momentum to start integration
            x0  :: float :: initial position to start integration
        
        Expectations
            save_path :: Bool :: save (p,x). IN PHASE: Start at (1,1)
            self.x_step = x0 when class is instantiated
            self.p_step = p0 when class is instantiated
        
        Returns
            (x,p) :: tuple :: momentum, position
        """
        self.p, self.x = p0,x0
        if self.save_path: self._storeSteps() # store zeroth step
        
        for step in xrange(0, self.n_steps):
            self._moveP(frac_step=0.5)
            self._moveX()
            self._moveP(frac_step=0.5)
            if self.save_path: self._storeSteps() # store moves
        
        # make into a 3-tensor of (path, column x/p, 1)
        # last shape required to preserve as column vector for np. matrix mul
        if self.save_path: 
            self.p_ar = np.asarray(self.p_ar).reshape(
                (len(self.p_ar), self.p.shape[0], self.p.shape[1]))
            self.x_ar = np.asarray(self.x_ar).reshape(
                (len(self.x_ar), self.x.shape[0], self.x.shape[1]))
        
        # remember that any usage of self.p,self.x will be stored as a pointer
        # must slice or use a copy(self.p) to "freeze" the current value in mem
        return self.p, self.x
    
    def integrateAlt(self, p0, x0):
        """The Leap Frog Integration
        
        Required Input
            p0  :: float :: initial momentum to start integration
            x0  :: float :: initial position to start integration
        
        Expectations
            save_path :: Bool :: save (p,x). OUT OF PHASE: Start at (.5,1)
            self.x_step = x0 when class is instantiated
            self.p_step = p0 when class is instantiated
        
        Returns
            (x,p) :: tuple :: momentum, position
        """
        
        self.p, self.x = p0,x0
        self._moveP(frac_step=0.5)
        self._moveX()
        if self.save_path: self._storeSteps() # store moves
        
        for step in xrange(1, self.n_steps):
            self._moveP()
            self._moveX()
            if self.save_path: self._storeSteps() # store moves
        
        self._moveP(frac_step=0.5)
        
        return self.p, self.x
    
    def _moveX(self, frac_step=1.):
        """Calculates a POSITION move for the Leap Frog integrator 
        
        Required Inputs
            p :: float :: current momentum
            x :: float :: current position
        """
        self.x += frac_step*self.step_size*self.p
        pass
    
    def _moveP(self, frac_step=1.):
        """Calculates a MOMENTUM move for the Leap Frog integrator 
        
        Required Inputs
            p :: float :: current momentum
            x :: float :: current position
        """
        self.p -= frac_step*self.step_size*self.duE(self.x)
        pass
    def _storeSteps(self):
        """Stores current momentum and position in lists
        
        Expectations
            self.x_step :: float
            self.p_step :: float
        """
        self.p_ar.append(copy(self.p))
        self.x_ar.append(copy(self.x))
        pass

#
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
        
        def display(test_name, steps, size, h_new, h_old, bench_mark, result): # print
            print '\n\n TEST: {}'.format(test_name)
            print ' initial H(p, x):',h_old
            print ' worst   H(p, x):',h_new
            print ' at steps: {}'.format(steps)
            print ' at step size: {}'.format(size)
            print ' np.abs(exp(-dH): {}'.format(h_new - h_old, bench_mark)
            print ' outcome: {}'.format(['Failed','Passed'][result])
            pass
        
        diffs = []
        
        # calculate original hamiltonian and set starting vals
        pi,xi = np.asarray([[4.]]), np.asarray([[1.]])
        h_old = self.pot.hamiltonian(pi,xi)
        
        # initial vals required
        w_bmk = 1.
        diff = 0.
        w_step = 0
        w_size = 0
        
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
            display("Constant Energy", w_step, w_size, w_h_new, h_old, w_bmk, passed)
        
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
                save_dir = './plots/'
                subprocess.call(['mkdir', './plots/'])
            
                fig.savefig(save_dir+save)
            else:
                plt.show()
            pass
        if save: plot(save=save)
        
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
        def display(test_name, steps, x0, p0, pf, xf, p0f, x0f, pc, result): # print
            print '\n\n TEST: {}'.format(test_name)
            print ' initial (p, x): ({}, {})'.format(p0, x0)
            print ' int.    (p, x): ({}, {})'.format(pf, xf)
            print ' final   (p, x): ({}, {})'.format(p0f, x0f)
            print ' phase change:    {}'.format(pc)
            print ' number of steps: {}'.format(steps)
            print ' outcome: {}'.format(['Failed','Passed'][passed])
            pass
        
        # params and ensure dynamic params correct
        p0, x0 = 4., 1.
        self.dynamics.n_steps = steps
        self.dynamics.step_size = 0.1
        self.dynamics.save_path = True
        
        pf,xf = self.dynamics.integrate(p0, x0)
        p0f,x0f = self.dynamics.integrate(-pf, xf) # time flip
        
        p0f = -p0f # time flip to point in right time again
        
        phase_change = np.linalg.norm( # calculate frobenius norm
            np.asarray([[p0f], [x0f]]) - np.asarray([[p0], [x0]])
            )
        
        passed = (phase_change < tol)
        
        if print_out: display("Reversibility of Integrator",
            self.dynamics.n_steps, x0, p0, pf, xf, p0f, x0f, phase_change, passed)
        
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
                save_dir = './plots/'
                subprocess.call(['mkdir', './plots/'])
            
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
            plot(steps, norm)
            
            self.dynamics.save_path = False
            
        return passed
#
class Demo_Hamiltonian_Dynamics(Pretty_Plotter):
    """Simulates Hamiltonian Dynamics using arbitrary integrator
    This class plots a test animation as a sort of Unit Test
    
    The code is really really messy towards the end.
    This is mainly because of the numerous tweaks to parameters
    made in the matplotlib script so was unavoidable!
    
    Required Inputs
        p0  :: float :: initial momentum
        x0  :: float :: initial position
        dynamics :: func :: integration function for Hamiltonian Dynamics
        potential :: class :: contains all potential funcs required
    
    Optional Input
        save :: string :: loc. and type of file to save options: see below
    
    Expectations
        Kinetic energy is a function of momentum
        Potential energy is a function of position
        Either may have additional parameters
    
    Notes
        Support for '.gif' :: A temp '.mp4' file will be created
                              and will be converted to '.png' files
                              with 'ffmpeg'. These are the made into a
                              '.gif' with 'convert'. Temp files del'd.
    """
    def __init__(self, p0, x0, dynamics, potential):
        # code test based on the following matlab blog entry
        prefix = 'https://'
        self.blog = 'TheCleverMachine.wordpress.com'
        url_path = '/2012/11/18/mcmc-hamiltonian-monte-carlo-a-k-a-hybrid-monte-carlo/'
        self.ref = prefix + self.blog + url_path
        
        self.potential = potential
        self.kE, self.uE = self.potential.kE, self.potential.uE
        
        self.p0,self.x0 = np.matrix(p0), np.matrix(x0)
        self.dynamics = dynamics
        
        self.p,self.x = self.p0,self.x0 # initial conditions
        pass
    
    def run(self):
        """Calculate a trajectory with Hamiltonian Dynamics"""
        self.p, self.x = self.dynamics.integrate(self.p, self.x)
        pass
    
    
    def energy_drift(self, p_ar, x_ar, save = 'energy_drift.png'):
        """Plot the Hamiltonian at each integrator integration"""
        
        # convert position/momentum to Energies
        kE_ar = np.apply_along_axis(self.potential.kE, 2, np.asarray(p_ar))
        uE_ar = np.apply_along_axis(self.potential.uE, 2, np.asarray(x_ar))
        assert len(kE_ar) == len(uE_ar)
        steps = np.arange(len(kE_ar))
        h = kE_ar + uE_ar
        
        fig = plt.figure(figsize=(8, 8)) # make plot
        ax =[]
        ax.append(fig.add_subplot(111))
        self._teXify() # LaTeX
        # fig.suptitle(r"Hamiltonian Dynamics of the SHO using Leap-Frog Integrator",
        #     fontsize=16)
        ax[0].set_title(r'Oscillating Hamiltonian as a function of Integration Steps: \textit{Energy Drift}')
        ax[0].set_xlabel(r'Integration Steps, $n$')
        ax[0].set_ylabel(r'Hamiltonian, $H(p,x)$')
        
        ax[0].plot(steps, kE_ar + uE_ar, linestyle='-', color='blue')
        
        if save:
            save_dir = './plots/'
            subprocess.call(['mkdir', './plots/'])
            
            fig.savefig(save_dir+save)
        else:
            plt.show()
        pass
    
    def full_anim(self, p_ar, x_ar, save = 'ham_dynamics.gif'):
        """Display a demo animation of Hamiltonian Dynamics
        
        Required Inputs
            p_ar :: np.array(float) :: momentum at each integration step
            x_ar :: np.array(float) :: position at each integration step
        
        Expectations
            self.save :: string
        
        This function is a bit messy because of the numerous tweaks
        and also the fact that there are three nested plots of 
        various formats in the final product.
        
        
        """
        
        kE,uE,dkE,duE = self.potential.all
        
        n = 1000 # Resolution of the spring (< 500 is shit)
        fig = plt.figure(figsize=(8, 8)) # make plot
        self._teXify() # make it lookm like LaTeX
        # fig.suptitle(r"Hamiltonian Dynamics of the SHO using Leap-Frog Integrator",
            # fontsize=16)
        fig.text(.02, .02,
        r"... Based on a blog entry from %s" % self.blog, fontsize=8)
        ax = [] # container list for the axes
        gs = gridspec.GridSpec(2, 2) # set up a 2x2 grid for axes
        ax.append(plt.subplot(gs[0, :]))    # Top spanning all
        ax.append(plt.subplot(gs[1,:-1]))   # Bottom Left
        ax.append(plt.subplot(gs[1:,-1]))   # Bottom Right
        
        ### (Top) Spring Plot
        ax[0].set_title(r'Oscillating Mass on a Spring: $V(x) = \frac{1}{2}x^2$')
        for tic in ax[0].yaxis.get_major_ticks(): # remove all y tics
            tic.tick1On = tic.tick2On = False     # I have no idea how
            tic.label1On = tic.label2On = False   # I also don't care \(*_*)/
        ax[0].set_xlabel(r"$x$")
        ax[0].set_xlim([-6.,6.])
        ax[0].set_ylim([-1.,1.])
        
        # Intial parameters
        pos = x_ar[0,:,:].reshape((1)) # expect 1D p and x
        mom = p_ar[0,:,:].reshape((1)) # expect 1D p and x
        x = np.linspace(-6., pos, n, endpoint=True) # x range to clip wire
        wire = np.sin(6. * np.linspace(0, 2.*np.pi, n)) # wire is clipped at x[-1]
        # Lines: spring (sin curve); weight (rectangle)
        
        spring, = ax[0].plot(x, wire, color='grey', linestyle='-', linewidth=2)
        weight = plt.Rectangle((0, -0.25), 0.5, 0.5, color='red')
        ### End (Top) Spring Plot
        
        ### (Bottom Left) Hamiltonian
        ax[1].set_title(r"Energy Functions")
        ax[1].set_ylabel(r"Energy, $H(p,x) = T(p)+V(x)$")
        ax[1].set_ylim([0,10])
        w = 0.25            # width of the bars
        i = np.arange(3)    # index of the bars
        ax[1].set_xticklabels(('T(p)', 'V(x)', 'H(p,x)')) # labels corresponding to i
        ax[1].set_xticks(i+2*w) # xtic locations
        
        # Lines: en_t (kinetic); en_v (potential); en_h (hamiltonian)
        en_t = ax[1].bar(0+w, kE(mom), width=2*w, color='magenta') # update kinetic en.
        en_v = ax[1].bar(1+w, uE(pos), width=2*w, color='cyan') # update potential en.
        en_h = ax[1].bar(2+w, kE(mom)+uE(pos), width=2*w, color='green')  # update ham.
        ### End (Bottom Left) Hamiltonian
        
        ### (Bottom Right) Phase Space
        ax[2].set_title(r"Phase Space")
        ax[2].set_xlabel(r"Position, $x$")
        ax[2].set_ylabel(r"Momentum, $p$")
        ax[2].set_xlim([-6.,6.])
        ax[2].set_ylim([-6.,6.])
        
        # Lines: all_traj (all trajectories) ; curr_traj (where we are now)
        # must flatten last column index to convert to a row vector
        # expect shape of (array_len, col_vector_dims, 1)
        all_traj, = ax[2].plot(x_ar[:,0,0], p_ar[:,0,0], linestyle='-', color='blue'
        # ,marker='o'
        )
        curr_traj = plt.Circle((0,0), radius=0.25, color='red')
        ### End (Bottom Right) Phase Space
        
        
        def init(): # initial conditions
            weight.set_x(pos)      # center weight on first point
            ax[0].add_patch(weight)         # add weight to plot
            
            curr_traj.center = (pos, mom)   # centers the circle
            ax[2].add_patch(curr_traj)      # add the patch to the plot
            return weight,curr_traj,        # must return for animation in order
        
        def animate(i): # animation func
            pos = x_ar[i,0,0]              # update x pos
            mom = p_ar[i,0,0]              # update x pos
            x = np.linspace(-6., pos, n)    # create x-pos array
            weight.set_x(pos)               # update weight pos
            spring.set_xdata(x)             # refresh spring x-axis
            
            kE_i,uE_i = kE(mom),uE(pos) # calculate energies
            
            # en_* come out as an interable so must make [du] etc. the same
            for xy in [(en_t, [kE_i]), (en_v, [uE_i]), (en_h, [kE_i+uE_i])]:
                for rect, h in zip(*xy): # for each bar chart get shape and new val
                    rect.set_height(h)  # then set the new height
            
            curr_traj.center = (pos, mom) # current phase space point
            return spring,en_t,en_v,en_h # must return as tuple again in order
        
        anim = animation.FuncAnimation(fig, animate, np.arange(0, len(x_ar)),
                                      interval=50, blit=False, init_func=init)
        
        if save:
            save_dir = './animations/'
            subprocess.call(['mkdir', './animations/'])
            
            if save.split('.')[-1] == 'gif':
                tmp_dir = './temp/' # avoid removal of similar files
                subprocess.call(['mkdir', './temp/'])
                tmp_mov = tmp_dir+save.split('.')[0]+'.mp4'
                
                anim.save(tmp_mov, fps=30, # save temp movie file as .mp4
                    extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
                
                subprocess.call(['ffmpeg','-loglevel','16', # convert .mp4 to pngs
                    '-i',tmp_mov,
                    '-r','10',tmp_dir+'output%05d.png'])
                
                subprocess.call(['convert', # convert pngs to .gif
                    tmp_dir+'output*.png',
                    save_dir+'{0}.gif'.format(save.split('.')[0])])
                
                subprocess.call(['rm', '-r', './temp/']) # remove temp folder
            else:
                anim.save(save_dir+save, fps=30, 
                    extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
        else:
            plt.show()
        pass
#
def fullDemo():
    """Displays two plots for Simple Harmonic Oscillator
    
    1. Energy Drift phenomenon
    2. Animated demo
        - mass on spring
        - phase space for leap-frog integrator
        - energy functions
    """
    pot = Simple_Harmonic_Oscillator(k=1.)
    lf = Leap_Frog(
        duE = pot.duE,
        step_size = 0.1,
        # n_steps = 63,
        n_steps = 250,
        save_path = True
        )
    
    test = Demo_Hamiltonian_Dynamics(
        p0 = [[1.]], x0 = [[4.]],
        dynamics = lf,
        potential = pot
        )
    
    test.run() # run dynamics
    
    test.energy_drift( # show energy drift
        save=False,
        p_ar = test.dynamics.p_ar,
        x_ar = test.dynamics.x_ar
        )
    
    test.full_anim( # animated demo
        save=False,
        p_ar = test.dynamics.p_ar,
        x_ar = test.dynamics.x_ar
        )

if __name__ == '__main__': # demo if run directly
    # fullDemo()
    integrator = Leap_Frog(duE = None, n_steps = 100, step_size = 0.1) # grad set in test
    tests = Tests(dynamics = integrator)
    r1 = tests.constantEnergy(
        tol = 0.05,
        step_sample = np.linspace(1, 100, 10, True, dtype=int),
        step_sizes = np.linspace(0.01, 0.1, 5, True),
        # These values are for pretty pictures
        save=False, # comment out to print figure
        # step_sample = np.linspace(1, 100, 100, True, dtype=int),
        # step_sizes = np.linspace(0.01, 0.5, 100, True),
        print_out = True # shows a small print out
        )
    r2 = tests.reversibility(
        steps = 1000,
        tol = 0.01,
        print_out = True # shows a small print out)
        )