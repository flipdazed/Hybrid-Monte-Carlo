import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import subprocess

from potentials import Simple_Harmonic_Oscillator
from h_dynamics import Hamiltonian_Dynamics, Leap_Frog

class Test(object):
    """This class plots a test animation as a sort of Unit Test
    
    The code is really really messy towards the end.
    This is mainly because of the numerous tweaks to parameters
    made in the matplotlib script so was unavoidable!
    
    Required Input
        potential :: class :: contains all potential funcs required
    
    Optional Input
        save :: string :: loc. and type of file to save options: see below
    
    Notes
        Support for '.gif' :: A temp '.mp4' file will be created
                              and will be converted to '.png' files
                              with 'ffmpeg'. These are the made into a
                              '.gif' with 'convert'. Temp files del'd.
    """
    def __init__(self, potential):
        
        # code test based on the following matlab blog entry
        prefix = 'https://'
        self.blog = 'TheCleverMachine.wordpress.com'
        url_path = '/2012/11/18/mcmc-hamiltonian-monte-carlo-a-k-a-hybrid-monte-carlo/'
        self.ref = prefix+self.blog+url_path
        
        self.kE, self.uE = potential.kE, potential.uE
        self.potential = potential
        pass
    
    def _teXify(self):
        """makes plots look posh"""
        
        self.s = 1   # Increase plot size by a scale factor
        self.fig_dims = [12*self.s,5*self.s]    # size of plot
        self.axfont = 11*self.s                 # axes
        self.tfont  = 14*self.s                 # subplot titles
        self.ttfont = 16*self.s                 # figure title
        
        plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
        
        # Customising Options
        params = {'text.usetex' : True,
                  'font.size' : 11,
                  'font.family' : 'lmodern',
                  'text.latex.unicode': True,
                  # 'text.latex.preamble': [r"\usepackage{hyperref}"], # doesn't work
                  # 'text.latex.preamble': [r"\usepackage{amsmath}"],
                  'figure.figsize' : self.fig_dims,
                  'figure.subplot.top':    0.85, #0.85 for title
                  'figure.subplot.hspace': 0.40,
                  'figure.subplot.wspace': 0.40,
                  'figure.subplot.bottom': 0.15,
                  'axes.titlesize': self.tfont,
                  'axes.labelsize': self.axfont,
                  'axes.grid': True
                  }
                  
        plt.rcParams.update(params) # updates the default parameters
        pass
    
    def energy_drift(self, p_ar, x_ar, save='energy_drift.png'):
        """Plot the Hamiltonian at each integrator integration"""
        
        # convert position/momentum to Energies
        kE_ar = self.potential.kE(np.asarray(p_ar))
        uE_ar = self.potential.uE(np.asarray(x_ar))
        assert len(kE_ar) == len(uE_ar)
        steps = np.arange(len(kE_ar))
        h = kE_ar + uE_ar
        
        fig = plt.figure(figsize=(8, 8)) # make plot
        ax =[]
        ax.append(fig.add_subplot(111))
        self._teXify() # LaTeX
        fig.suptitle(r"Hamiltonian Dynamics of the SHO using Leap-Frog Integrator",
            fontsize=16)
        ax[0].set_title(r'Oscillating Hamiltonian as a function of Integration Steps: \textit{Energy Drift}')
        ax[0].set_xlabel(r'Integration Steps, $\epsilon$')
        ax[0].set_ylabel(r'Hamiltonian, $H(p,x)$')
        
        ax[0].plot(steps, kE_ar + uE_ar, linestyle='-', color='blue')
        
        if save:
            save_dir = './plots/'
            subprocess.call(['mkdir', './plots/'])
            
            fig.savefig(save_dir+save)
        else:
            plt.show()
        pass
    
    def full_anim(self, p_ar, x_ar, save='ham_dynamics.gif'):
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
        fig.suptitle(r"Hamiltonian Dynamics of the SHO using Leap-Frog Integrator",
            fontsize=16)
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
        pos = x_ar[0]
        mom = p_ar[0]
        x = np.linspace(-6., pos, n, endpoint=True) # x range to clip wire
        wire = np.sin(6.*np.linspace(0,2.*np.pi,1000)) # wire is clipped at x[-1]
        
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
        all_traj, = ax[2].plot(x_ar, p_ar, linestyle='-', color='blue'
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
            pos = x_ar[i]              # update x pos
            mom = p_ar[i]              # update x pos
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
# Spacing (:
#
if __name__ == '__main__': # demo if run directly
    
    ### Simple harmonic Oscillator
    pot = Simple_Harmonic_Oscillator(k=1.)
    lf = Leap_Frog(duE=pot.duE, d=0.1, 
    # l = 63,
    l=250,
    save_path=True)
    
    hd = Hamiltonian_Dynamics(p0=1., x0=-4., integrator=lf)
    hd.integrate()
    
    test = Test(potential=pot)
    
    test.energy_drift(
        # save=False,
        p_ar=hd.integrator.p_ar,
        x_ar=hd.integrator.x_ar
    )
    
    # plot animated tests
    test.full_anim(
        # save=False,
        p_ar=hd.integrator.p_ar,
        x_ar=hd.integrator.x_ar
        )
        
    ### End SHO