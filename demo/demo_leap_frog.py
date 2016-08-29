import numpy as np
import subprocess

import hmc
from hmc.dynamics import Leap_Frog
from hmc.potentials import Simple_Harmonic_Oscillator
from test.utils.logs import *

from plotter import Pretty_Plotter, ANIM_LOC, PLOT_LOC
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
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
        super(Pretty_Plotter,self).__init__()
        # code test based on the following matlab blog entry
        prefix = 'https://'
        self.blog = 'TheCleverMachine.wordpress.com'
        url_path = '/2012/11/18/mcmc-hamiltonian-monte-carlo-a-k-a-hybrid-monte-carlo/'
        self.ref = prefix + self.blog + url_path
        
        self.potential = potential
        self.kE, self.uE = self.potential.kE, self.potential.uE
        
        self.p0, self.x0 = np.array(p0), np.array(x0)
        self.dynamics = dynamics
        
        self.p,self.x = self.p0, self.x0 # initial conditions
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
            save_dir = '../' + PLOT_LOC
            subprocess.call(['mkdir', save_dir])
            
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
        p = self.potential
        kE,uE,duE = p.kE, p.uE, p.duE
        
        n = 1000 # Resolution of the spring (< 500 is shit)
        fig = plt.figure(figsize=(8, 8), dpi=600) # make plot
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
        p_ar = np.array(p_ar)
        x_ar = np.array(x_ar)
        pos = x_ar[0].reshape((1,)) # expect 1D p and x
        mom = p_ar[0].reshape((1,)) # expect 1D p and x
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
            save_dir = ANIM_LOC
            subprocess.call(['mkdir', save_dir])
            
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
    logger.info('Demonstrating Hamiltonian Dynamics')
    
    logger.debug('Potential: SHO')
    pot = Simple_Harmonic_Oscillator(k=1.)
    
    logger.info('Integrator: Leap Frog')
    lf = Leap_Frog(
        duE = pot.duE,
        step_size = 0.1,
        # n_steps = 63, # 64 = full circle in phase space
        n_steps = 250,
        save_path = True
        )
    
    test = Demo_Hamiltonian_Dynamics(
        p0 = [[1.]], x0 = [[4.]],
        dynamics = lf,
        potential = pot
        )
    
    logger.debug('Running integration')
    test.run() # run dynamics
    
    logger.info('Demonstrating Energy Drift')
    test.energy_drift( # show energy drift
        save=False,
        p_ar = test.dynamics.p_ar,
        x_ar = test.dynamics.x_ar
        )
    
    logger.info('Plotting an animated demonstration')
    logger.debug('Circular phase space characteristic of Symplectic integrators')
    test.full_anim( # animated demo
        save='h_dynamics_anim.mp4',
        p_ar = test.dynamics.p_ar,
        x_ar = test.dynamics.x_ar
        )
#
if __name__ == '__main__': # demo if run directly
    fullDemo()