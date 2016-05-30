import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import subprocess

# code based on the following matlab blog entry
prefix = 'https://'
blog = 'TheCleverMachine.wordpress.com'
url_path = '/2012/11/18/mcmc-hamiltonian-monte-carlo-a-k-a-hybrid-monte-carlo/'
ref = prefix+blog+url_path

class Hamiltonian_Dynamics(object):
    """Simulates Hamiltonian Dynamics using Leap-Frog Integration
    
    Required Inputs
        p0  :: float :: initial momentum
        x0  :: float :: initial position
        k   :: function :: Kinetic Energy
        u   :: function :: Potential Energy
        dk  :: function :: Gradient of Kinetic Energy
        du  :: function :: Gradient of Potential Energy
    
    Optional Inputs
        delta   :: integration step length
        lf      :: leap frog integration steps (trajectory length)
    
    Expectations
        Kinetic energy is a function of momentum
        Potential energy is a function of position
        Either may have additional parameters
    """
    def __init__(self, p0, x0, k, u, dk, du, delta=0.1, lf = 250):
        self.p0,self.x0 = p0, x0   # initial conditions
        
        self.k, self.u = k,u        # energy functions
        self.dk, self.du = dk,du    # energy gradients
        
        self.delta = delta  # step size
        self.lf = lf        # leap-frog step length
        
        self.pStep_arr = [] # data for plots
        self.xStep_arr = [] # data for plots
        pass
    
    def leapFrog(self):
        """The Leap Frog Integration"""
        
        self.pStep = self.p0 - self.delta/2.*self.du(self.x0) # first half mom. step.
        self.xStep = self.x0 + self.delta*self.pStep # first full pos. step (NEW mom.)
        
        for step in xrange(1, self.lf): # continue with full steps
            self.leapFrogMove(self.pStep,self.xStep)
        
        self.pStep -= self.delta/2.*self.du(self.xStep)  # last half step with momentum
        
        pass
        
    def leapFrogMove(self, p, x):
        """Calculates a move for the Leap Frog integrator 
        
        Required Inputs
            p :: float :: current momentum
            x :: float :: current position
        """
        
        p -= self.delta*self.du(x)  # momentum step with NEW pos.
        x += self.delta*p           # position step with NEW mom.
        
        self.pStep, self.xStep = p,x   # assign for next iteration
        
        self._storeSteps() # store data
        pass
    
    def _storeSteps(self):
        """Stores current momentum and position in lists
        
        Expectations
            self.xStep :: float
            self.pStep :: float
        """
        self.pStep_arr.append(self.pStep)
        self.xStep_arr.append(self.xStep)
        pass
    
class Test(object):
    """This class plots a test animation as a sort of Unit Test
    
    The code is really really messy towards the end.
    This is mainly because of the numerous tweaks to parameters
    made in the matplotlib script so was unaviodable!
    
    Required Input
        params :: [p0, x0, k, u, dk, du] :: defined in Hamiltonian_Dynamics()
    
    Optional Input
        save :: string :: loc. and type of file to save options: see below
    
    Notes
        Support for '.gif' :: A temp '.mp4' file will be created
                              and will be converted to '.png' files
                              with 'ffmpeg'. These are the made into a
                              '.gif' with 'convert'. Temp files del'd.
    """
    def __init__(self, params, save='ham_dynamics.gif'):
        self.save = save
        
        self.Ham_Dyn = Hamiltonian_Dynamics(*params)
        self.Ham_Dyn.leapFrog() # runs the integrator
        
        self.plot()
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
    
    def plot(self):
        """Display a demo animation of Hamiltonian Dynamics
        
        Expectations
            self.save :: string
        
        This function is a bit messy because of the numerous tweaks
        and also the fact that there are three nested plots of 
        various formats in the final product.
        
        
        """
        ### Set up
        
        xStep_arr, pStep_arr = self.Ham_Dyn.xStep_arr, self.Ham_Dyn.pStep_arr
        dk,du = self.Ham_Dyn.dk,self.Ham_Dyn.du
        k,u = self.Ham_Dyn.k,self.Ham_Dyn.u
        
        n = 1000 # Resolution of the spring (< 500 is shit)
        fig = plt.figure(figsize=(8, 8)) # make plot
        self._teXify() # make it lookm like LaTeX
        fig.suptitle(r"Hamiltonian Dynamics of the SHO using Leap-Frog Integrator",
            fontsize=16)
        fig.text(.02, .02,
        r"... Based on a blog entry from %s" % blog, fontsize=8)
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
        pos = xStep_arr[0]
        mom = pStep_arr[0]
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
        en_t = ax[1].bar(0+w, k(mom), width=2*w, color='magenta') # update kinetic en.
        en_v = ax[1].bar(1+w, u(pos), width=2*w, color='cyan') # update potential en.
        en_h = ax[1].bar(2+w, k(mom)+u(pos), width=2*w, color='green')  # update ham.
        ### End (Bottom Left) Hamiltonian
        
        ### (Bottom Right) Phase Space
        ax[2].set_title(r"Phase Space")
        ax[2].set_xlabel(r"Position, $x$")
        ax[2].set_ylabel(r"Momentum, $p$")
        ax[2].set_xlim([-6.,6.])
        ax[2].set_ylim([-6.,6.])
        
        # Lines: all_traj (all trajectories) ; curr_traj (where we are now)
        all_traj, = ax[2].plot(xStep_arr, pStep_arr, linestyle='-', color='blue')
        curr_traj = plt.Circle((0,0), radius=0.25, color='red')
        ### End (Bottom Right) Phase Space
        
        
        def init(): # initial conditions
            weight.set_x(pos)      # center weight on first point
            ax[0].add_patch(weight)         # add weight to plot
            
            curr_traj.center = (pos, mom)   # centers the circle
            ax[2].add_patch(curr_traj)      # add the patch to the plot
            return weight,curr_traj,        # must return for animation in order
        
        def animate(i): # animation func
            pos = xStep_arr[i]              # update x pos
            mom = pStep_arr[i]              # update x pos
            x = np.linspace(-6., pos, n)    # create x-pos array
            weight.set_x(pos)               # update weight pos
            spring.set_xdata(x)             # refresh spring x-axis
            
            k_i,u_i = k(mom),u(pos) # calculate gradients
            
            # en_* come out as an interable so must make [du] etc. the same
            for xy in [(en_t, [k_i]), (en_v, [u_i]), (en_h, [k_i+u_i])]:
                for rect, h in zip(*xy): # for each bar chart get shape and new val
                    rect.set_height(h)  # then set the new height
            
            curr_traj.center = (pos, mom) # current phase space point
            return spring,en_t,en_v,en_h # must return as tuple again in order
        
        anim = animation.FuncAnimation(fig, animate, np.arange(0, len(xStep_arr)),
                                      interval=50, blit=False, init_func=init)
        
        if self.save:
            save_dir = './animations/'
            subprocess.call(['mkdir', './animations/'])
            
            if self.save.split('.')[-1] == 'gif':
                tmp_dir = './temp/' # avoid removal of similar files
                subprocess.call(['mkdir', './temp/'])
                tmp_mov = tmp_dir+self.save.split('.')[0]+'.mp4'
                
                anim.save(tmp_mov, fps=30, # save temp movie file as .mp4
                    extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
                
                subprocess.call(['ffmpeg','-loglevel','16', # convert .mp4 to pngs
                    '-i',tmp_mov,
                    '-r','10',tmp_dir+'output%05d.png'])
                
                subprocess.call(['convert', # convert pngs to .gif
                    tmp_dir+'output*.png',
                    save_dir+'{0}.gif'.format(self.save.split('.')[0])])
                
                subprocess.call(['rm', '-r', './temp/']) # remove temp folder
            else:
                anim.save(save_dir+self.save, fps=30, 
                    extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
        else:
            plt.show()
        pass

#
# Spacing (:
#
if __name__ == '__main__': # demo if run directly
    ### Hamiltonian functions (SHO)
    k = lambda p: p**2/2.  # kinetic energy
    dk = lambda p: p       # gradient
    u = lambda x, k=1: k*x**2/2.   # potential (def. k=1)
    du = lambda x, k=1: k*x        # gradient
    ### End Hamiltonian functions
    
    ### Initial conditions
    x0 = -4.    # position
    p0 = 1.     # momentum
    ### End Initial conditions
    
    params = [p0, x0, k, u, dk, du] # package as list
    
    test = Test(params=params, save='ham_dynamics.gif') # animate!