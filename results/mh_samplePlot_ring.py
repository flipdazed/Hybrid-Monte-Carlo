#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from plotter import Pretty_Plotter, PLOT_LOC, magma, inferno, plasma, viridis

from matplotlib.collections import LineCollection
from hmc.potentials import Ring_Potential
from numpy.random import uniform as U
rng = np.random.RandomState()
def plot(burn_in, samples, bg_xyz, save):
    """Note that samples and burn_in contain the initial conditions"""
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['figure.subplot.top'] = 0.85
    pp._updateRC()
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    
    ax.set_xlabel(r'$\phi_{1}$', fontsize=pp.axfont)
    ax.set_ylabel(r'$\phi_{2}$', fontsize=pp.axfont)
    
    # fig.suptitle(r'Sampling a ring potential with HMC',
        # fontsize=pp.ttfont)
    pot = r'$Q(\phi_{x}) = e^{-50\left|\phi_{x}^2 + 1/10\right|}$'
    ax.set_title(r'{} burn-in \& {} MH moves'.format(
    max(burn_in.shape), max(samples.shape)),
        fontsize=pp.tfont)
    
    plt.grid(False)
    
    x, y, z = bg_xyz
    
    # small = 1e-1
    # mask = z<small
    # z = np.ma.MaskedArray(-z, mask, fill_value=0)
    # x = np.ma.MaskedArray(x, mask, fill_value=np.nan)
    # y = np.ma.MaskedArray(y, mask, fill_value=np.nan)
    
    cnt = ax.contourf(x, y, z, 100, cmap=viridis,
        alpha=.3, antialiased=True)
    
    l0 = ax.plot(burn_in[0,0], burn_in[0,1], 
        marker='o', markerfacecolor='green'
        )
    
    l1 = ax.plot(burn_in[:,0], burn_in[:,1], 
        color='green', linestyle='--',alpha=0.3,
        # marker='o', markerfacecolor='red'
        )
    
    x,y = samples[:,0], samples[:,1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=plt.get_cmap('magma_r'),norm=plt.Normalize(250, 1500))
    lc.set_array(np.arange(samples[:,1].size))
    lc.set_linewidth(1.)
    lc.set_alpha(0.3)
    ax.add_collection(lc)
    
    l2 = ax.scatter(samples[:,0], samples[:,1],
        color='blue',alpha=0.7, s=3,
        marker='o',
        )
    
    pp.save_or_show(save, PLOT_LOC)
    pass
#
def getPotential(potFn, n_points=100):
    """Returns x,y,z of the analytic potential
    
    Required Inputs
        potFn   :: function :: the potential energy function
    
    Optional Inputs
        n_points :: int :: defines the resolution
    
    Expected
        potFn takes a 1x2 column matrix in and returns a point
    """
    
    n_points = 1000    # n**2 is the number of points
    x = np.linspace(-1.5, 1.5, n_points, endpoint=True)
    x,y = np.meshgrid(x, x)
    
    z = np.exp(-model.potential.uE(np.asarray([x,y])))
    z = np.asarray(z).reshape(n_points, n_points)
    return x,y,z

class Model(object):
    
    def __init__(self, x0, pot):
        self.x0 = x0
        self.potential = pot
        self.pot = lambda x: np.exp(-self.potential.uE(np.asarray(x)))
        pass
    def metropolisHastings(self, samples,q,phi):
        """# MH sampling from a generic distribution, q"""
        phi = np.array(phi)
        chain = [phi]
        accRej = lambda old,new : min([q(new)/q(old),1])
        for i in xrange(samples):
            proposal = phi + U(-1,1,phi.size).reshape(phi.shape)
            if U(0,1) < accRej(phi,proposal):
                phi = proposal
            chain.append(phi) # append old phi if rejected
        return np.array(chain) # phi_{t,x}
    
    def run(self, n_samples, n_burn_in):
        run = self.metropolisHastings(n_samples+n_burn_in, self.pot, self.x0)
        self.burn_in = run[:n_burn_in]
        self.samples = run[n_burn_in:]
        pass
#
if __name__ == '__main__':
    
    n_burn_in = 100
    n_samples = 1000
    
    pot = Ring_Potential(scale=5, bias=-1)
    
    x0 = np.asarray([[0.],[0.]])
    model = Model(x0, pot=pot)
    
    # adjust for nice plotting
    print "Running Model: {}".format(__file__)
    model.run(n_samples=n_samples, n_burn_in=n_burn_in)
    print "Finished Running Model: {}".format(__file__)
    
    # change shape from (n, 2) -> (2, n)
    samples = model.samples
    burn_in = model.burn_in
    
    xyz = getPotential(pot.uE)
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    plot(burn_in=burn_in, samples=samples, bg_xyz=xyz,
        save = save_name
        # save = False
        )
