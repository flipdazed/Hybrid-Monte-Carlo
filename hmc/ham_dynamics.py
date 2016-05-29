import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import subprocess

### Parameters
delta = 0.1  # step size
lf = 250     # leap-frog step length
pStep_arr = []  # data for plots
xStep_arr = []  # data for plots
### End Parameters

### Hamiltonian functions (SHO)
tE = lambda p: p**2/2.  # kinetic energy
dtE = lambda p: p       # gradient
vE = lambda x, k=1: k*x**2/2.   # potential (def. k=1)
dvE = lambda x, k=1: k*x        # gradient
### End Hamiltonian functions

### Initial conditions
x0 = -4.    # position
p0 = 1.     # momentum
### End Initial conditions

### Hamiltonian Dynamics: Leap-Frog
pStep = p0 - delta/2.*dvE(x0)  # first half step with momentum
xStep = x0 + delta*pStep       # first full step with position

pStep_arr.append(pStep) # plotting
xStep_arr.append(xStep) # plotting

# continvE with full steps
for i_lf in xrange(1, lf):
    pStep = pStep - delta*dvE(xStep) # update momentum
    xStep = xStep + delta*pStep      # update position
    pStep_arr.append(pStep) # plotting
    xStep_arr.append(xStep) # plotting

pStep = pStep - delta/2.*dvE(xStep)  # last half step with momentum
pStep_arr.append(pStep) # plotting
xStep_arr.append(xStep) # plotting
### End Hamiltonian Dynamics

class Test(object):
    
    def __init__(self, save='ham_dynamics.gif'):
        self.plot(save=save)
        pass
    
    def teXify(self):
        """makes plots look posh"""
        #Direct input 
        self.s = 1   # Increase plot size
        self.fig_dims = [12*self.s,5*self.s]   # size of plot
        self.axfont = 11*self.s           # axes
        self.tfont  = 14*self.s           # subplot titles
        self.ttfont = 16*self.s           # figure title
        
        plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
        #Options
        params = {'text.usetex' : True,
                  'font.size' : 11,
                  'font.family' : 'lmodern',
                  'text.latex.unicode': True,
                  'figure.figsize' : self.fig_dims,
                  'figure.subplot.top':    0.85, #0.85 for title
                  'figure.subplot.hspace': 0.40,
                  'figure.subplot.wspace': 0.40,
                  'figure.subplot.bottom': 0.15,
                  'axes.titlesize': self.tfont,
                  'axes.labelsize': self.axfont,
                  'axes.grid': True
                  }
        plt.rcParams.update(params)
        pass
    
    def plot(self, save):
        """Display a demo animation of Hamiltonian Dynamics"""
        # Set up
        n = 1000
        fig = plt.figure(figsize=(8, 8)) # instanciate plot
        self.teXify() # make nice
        fig.suptitle(r"Hamiltonian Dynamics of the SHO using Leap-Frog Integrator", fontsize=16)
        gs = gridspec.GridSpec(2, 2) # set up a 2x2 grid
        ax = []
        ax.append(plt.subplot(gs[0, :]))    # Top
        ax.append(plt.subplot(gs[1,:-1]))   # Bottom Left
        ax.append(plt.subplot(gs[1:,-1]))   # Bottom Right
        
        ### (Top) Spring Plot Setup
        ax[0].set_title(r'Oscillating Mass on a Spring: $V(x) = \frac{1}{2}x^2$')
        for tic in ax[0].yaxis.get_major_ticks(): # remove all y tics
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
            
        ax[0].set_xlabel("x")
        ax[0].set_xlim([-6.,6.])
        ax[0].set_ylim([-1.,1.])
        
        # intial parameters
        pos = xStep_arr[0]
        mom = pStep_arr[0]
        x = np.linspace(-6., pos, n, endpoint=True) # x range to clip wire
        wire = np.sin(6.*np.linspace(0,2.*np.pi,1000)) # wire is clipped at x[-1]
        
        # lines: spring (sin curve); weight (rectangle)
        spring, = ax[0].plot(x, wire, color='grey', linestyle='-', linewidth=2)
        weight = plt.Rectangle((0, -0.25), 0.5, 0.5, color='r')
        ### End (Top) Spring Plot
        
        ### (Bottom Left) Hamiltonian
        ax[1].set_title(r"Hamiltonian Function")
        ax[1].set_ylabel(r"Energy, $E(x) = T(p)+V(x)$")
        ax[1].set_ylim([0,10])
        w = 0.25
        i = np.arange(3)
        ax[1].set_xticks(i+2*w)
        ax[1].set_xticklabels(('T(p)', 'V(x)', 'E(x)'))
        
        en_t = ax[1].bar(0+w, tE(mom), width=2*w, color='magenta') # update kinetic en.
        en_v = ax[1].bar(1+w, vE(pos), width=2*w, color='cyan') # update potential en.
        en_h = ax[1].bar(2+w, tE(mom)+vE(pos), width=2*w, color='green')  # update ham.
        ### End (Bottom Left) Hamiltonian
        
        ### (Bottom Right) Phase Space
        ax[2].set_title(r"Phase Space")
        ax[2].set_xlabel(r"Position, $x$")
        ax[2].set_ylabel(r"Momentum, $p$")
        ax[2].set_xlim([-6.,6.])
        ax[2].set_ylim([-6.,6.])
        
        all_traj, = ax[2].plot(xStep_arr, pStep_arr, linestyle='-', color='blue')
        curr_traj = plt.Circle((0,0), radius=0.25, color='r')
        ### End (Bottom Right) Phase Space
        
        
        def init(): # initial conditions
            weight.set_x(pos)      # center weight on first point
            ax[0].add_patch(weight)         # add weight to plot
            
            curr_traj.center = (pos, mom)
            ax[2].add_patch(curr_traj)
            return weight,curr_traj,
        
        def animake(i): # animation func
            pos = xStep_arr[i]              # update x pos
            mom = pStep_arr[i]              # update x pos
            x = np.linspace(-6., pos, n)    # create x-pos array
            weight.set_x(pos)               # update weight pos
            spring.set_xdata(x)             # refresh spring x-axis
            
            for rect, h in zip(en_t, [tE(mom)]):
                    rect.set_height(h)
            for rect, h in zip(en_v, [vE(pos)]):
                    rect.set_height(h)
            for rect, h in zip(en_h, [tE(mom)+vE(pos)]):
                    rect.set_height(h)
            
            curr_traj.center = (pos, mom)   # current phase space point
            return spring,en_t,en_v,en_h
        
        anim = animation.FuncAnimation(fig, animake, np.arange(0, len(xStep_arr)),
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

if __name__ == '__main__': # demo if run directly
    test = Test()