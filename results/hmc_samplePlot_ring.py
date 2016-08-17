#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from plotter import Pretty_Plotter, PLOT_LOC, magma, inferno, plasma, viridis

from matplotlib.collections import LineCollection
from models import Basic_HMC as Model
from hmc.potentials import Ring_Potential

def plot(burn_in, samples, bg_xyz, mdmc,save):
    """Note that samples and burn_in contain the initial conditions"""
    
    pp = Pretty_Plotter()
    pp._teXify() # LaTeX
    pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    pp.params['figure.subplot.top'] = 0.85
    pp._updateRC()
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    
    ax.set_xlabel(r'$\phi_{1}$')
    ax.set_ylabel(r'$\phi_{2}$')
    
    # fig.suptitle(r'Sampling a ring potential with HMC',
        # fontsize=pp.ttfont)
    # pot = r'$Q(\phi_{x}) = e^{-50|\phi_{x}^2 + \tfrac{1}{10}|}$'
    # ax.set_title(r'Showing {} burn-in \& {} HMC moves sampling from {}'.format(
    # max(burn_in.shape), max(samples.shape), pot),
    #     fontsize=(pp.tfont-4))
    #
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
    l2 = ax.scatter(samples[:,0], samples[:,1],
        color='blue',alpha=0.7, s=3,
        marker='o',
        )
    
    if mdmc is not None: samples = mdmc
    
    x,y = samples[:,0], samples[:,1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, cmap=plt.get_cmap('magma_r'),norm=plt.Normalize(250, 1500))
    lc.set_array(np.arange(samples[:,1].size))
    lc.set_linewidth(1.)
    lc.set_alpha(0.3)
    ax.add_collection(lc)
    
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
    
    z = np.exp(-model.sampler.potential.uE(np.asarray([x,y])))
    z = np.asarray(z).reshape(n_points, n_points)
    return x,y,z
#
if __name__ == '__main__':
    
    n_burn_in =100
    n_samples = 1000
    
    pot = Ring_Potential(scale=5, bias=-1)
    
    x0 = np.asarray([[0.],[0.]])
    rng = np.random.RandomState()
    accept_kwargs={'store_acceptance':True, 'accept_all':True}
    model = Model(x0, rng=rng, pot=pot, step_size=0.1, n_steps=20,
        rand_steps=True, save_path=True)
    
    # adjust for nice plotting
    print "Running Model: {}".format(__file__)
    model.run(n_samples=n_samples, n_burn_in=n_burn_in)
    print "Finished Running Model: {}".format(__file__)
    
    # the MDMC steps
    p_mdmc = model.sampler.dynamics.p_ar # contains start step from each previous
    x_mdmc = np.asarray(model.sampler.dynamics.x_ar).reshape(-1,2) # so shape is n_steps + 1
    
    samples = model.samples
    burn_in = model.burn_in
    
    xyz = getPotential(pot.uE)
    
    f_name = os.path.basename(__file__)
    save_name = os.path.splitext(f_name)[0] + '.png'
    plot(burn_in=burn_in, samples=samples, bg_xyz=xyz,mdmc=None,
        save = save_name
        # save = False
        )
