from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
from plotter import Pretty_Plotter, PLOT_LOC

class Cube(Pretty_Plotter):
    """ plots a cube """
    def __init__(self):
        
        # set up figure
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_aspect("equal")
        
        # make pretty
        self._teXify() # LaTeX
        self.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
        self._updateRC()
        self.ax.axis('off')
        
        self.labels = {}
        pass
    
    def drawBase(self):
        """draws the cube"""
        # draw cube of 1,1,1 dimensions
        r = [-1, 1]
        for s, e in combinations(np.array(list(product(r,r,r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                self.ax.scatter(*zip(s,e), color="b", marker='o')
                self.ax.plot3D(*zip(s,e), color="r", linestyle='--')
        plt.draw()
        pass
    
    def plotLabels(self, text, x, y, z, offset=(-20, 20)):
        """ adds labels to the 3d figure"""
        
        # create tranfsromation of cube dimensions to 2D
        x2, y2, _ = proj3d.proj_transform(x,y,z, self.ax.get_proj())
        
        # create labels
        label = self.ax.annotate(
            text, 
            xy = (x2, y2), xytext = offset,
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            # bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0.2')
            )
        
        # store label
        self.labels[text] = label
        
        def update_position(e):
            """click and the labels snap the correct place
            http://stackoverflow.com/a/10394128/4013571
            """
            x2, y2, _ = proj3d.proj_transform(x,y,z, self.ax.get_proj())
            self.labels[text].xy = x2,y2
            self.labels[text].update_positions(self.fig.canvas.renderer)
            self.fig.canvas.draw()
            pass
        
        # enable the updated position
        self.fig.canvas.mpl_connect('button_release_event', update_position)
        plt.draw()
        pass
    
    def title(self, text):
        self.ax.set_title(text, fontsize=14)
        pass
    
    def save(self, name = 'lattice_momenta.png'):
        self.fig.savefig(PLOT_LOC + 'plots/' + name)
        pass

if __name__ == '__main__':
    
    cube = Cube()
    cube.drawBase()
    # x, y, z = (width, height, depth)
    cube.plotLabels(r'$\vec{e}_{x_1}$', 0, -1, -1, (-20, -10))
    cube.plotLabels(r'$\vec{e}_{x_2} \equiv \vec{e}_{\tau}$', -1, -1, 0, (-20, 10))
    cube.plotLabels(r'$\vec{e}_{\pi}$', 1, 0, 1, (-15, 10))
    cube.plotLabels(r'$\phi(x_1, \tau, \pi)$', -1, -1, 1, (-15, 10))
    cube.title(r'Introducing conjugate momenta, $\pi$, to a 2D $(x,\tau)$ Lattice')
    cube.save()