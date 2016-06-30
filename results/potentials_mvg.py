import numpy as np
import os
import matplotlib.pyplot as plt

from hmc.potentials import Multivariate_Gaussian
from plotter import Pretty_Plotter, viridis, magma, inferno, plasma, PLOT_LOC

def plot(x, y, z, save):
    """Plots a test image of the Bivariate Gaussian
    
    Required Inputs
        x,y,z   :: arrays        :: must all be same length
        save    :: string / bool :: save location or '' / False
    """
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp.params['figure.subplot.top'] = 0.85
    pp._updateRC()
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    
    # create contour plot
    c = ax.contourf(x, y, z, 100, cmap=plasma)
    
    # axis labels
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.grid(False) # remove grid
    
    fig.suptitle( # title
    r'Test plot of a 2D Multivariate (Bivariate) Gaussian', fontsize=pp.ttfont)
    
    ax.set_title( # subtitle
    r'Parameters: $\mu=\begin{pmatrix}0 & 0\end{pmatrix}$, $\Sigma = \begin{pmatrix} 1.0 & 0.8\\ 0.8 & 1.0 \end{pmatrix}$',
        fontsize=pp.tfont-4)
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
if __name__ == '__main__':
    
    print "Running Model: {}".format(__file__) 
    # define the mean and covariance
    mean = np.asarray([[0.], [0.]])
    cov = np.asarray([[1.0,0.8], [0.8,1.0]])
    
    # n**2 is the number of points
    # defines the resolution of the plot
    n = 200
    
    bg = Multivariate_Gaussian(mean = mean, cov = cov)
    
    # create a mesh grid of NxN
    x = np.linspace(-5., 5., n, endpoint=True)
    x,y = np.meshgrid(x,x)
    
    # ravel() flattens the arrays into 1D vectors
    # and then they are passed as (x,y) components to the
    # potential term to form z = f(x,y)
    z = np.exp(-np.asarray([bg.uE(np.matrix([[i],[j]])) \
        for i,j in zip(np.ravel(x), np.ravel(y))]))
    
    # reshape back into an NxN
    z = np.asarray(z).reshape(n, n)
    print "Finished Running Model: {}".format(__file__)
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    plot(x,y,z,
        save = save_name
        # save = False
        )
    
