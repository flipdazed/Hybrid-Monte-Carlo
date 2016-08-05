#!/usr/bin/env python
import numpy as np

from results.data.store import load, store
from hmc.potentials import Klein_Gordon as KG
from correlations.corr import twoPoint
from correlations.acorr import acorr as myAcorr
from correlations.errors import gW, windowing, uWerr, getW, autoWindow, acorrnErr

from common.acorr import plot
from common.utils import saveOrDisplay

file_name = __file__
save      = True
m         = 1.0
n_steps   = 40
step_size = 1/((3.*np.sqrt(3)-np.sqrt(15))*m/2.)/float(n_steps)

n, dim    = 10, 1
spacing   = 1.

n_samples = 100000
n_burn_in = 25
c_len     = 10000

pot       = KG(m=m)
x0 = np.random.random((n,)*dim)
separations = range(c_len)       # define how many separations for a/c
tau = n_steps*step_size

opFn    = lambda samples: twoPoint(samples, separation=0)
op_name = r'$\hat{O} = \sum_{pq} \Omega \phi_p\phi_q :\Omega = \delta_{p0}\delta_{q0}$'

my_loc         = "results/data/numpy_objs/acorr_xx_comparison_alex.json"
comparison_loc = 'results/data/other_objs/acorr_xx_comparison_christian.pkl'

mixing_angle = .5*np.pi

subtitle = r"Potential: {}; Lattice: ${}$; $a={:.1f}; \delta\tau={:.2f}; n={}$".format(
    pot.name, x0.shape, spacing, step_size, n_steps)

def reObtain():
    """Re-obtains the MCMC samples - note that these are position samples
    not the sampled operator!
    
    Expectations :: this func relies on lots of variables above!
    """
    print 'Running Model: {}'.format(file_name)
    rng = np.random.RandomState()
    model = Model(x0, pot=pot, spacing=spacing, rng=rng, step_size = step_size,
      n_steps = n_steps, rand_steps=rand_steps)
      
    c = acorr.Autocorrelations_1d(model)
    c.runModel(n_samples=n_samples, n_burn_in=n_burn_in, mixing_angle = mixing_angle, verbose=True)
    
    c._getSamples()
    acs = c.getAcorr(separations, opFn, norm = False)   # non norm for uWerr
    store(c.samples, file_name, '_alex')
    print 'Finished Running Model: {}'.format(file_name)
    
    traj = c.trajs
    p = c.model.p_acc
    xx = c.op_mean
    return c.samples
#
def preparePlot(my_data, comparison_data, my_err=None, comparison_err=None):
    """Compares and plots the two datasets
    
    Required Inputs
        my_data          :: np array :: y data as a benchmark
        comparison_data  :: np array :: y data to compare to my_data
        my_err           :: np array :: y errs as a benchmark
        comparison_err   :: np array :: y errs to compare to my_data
    """
    x = np.asarray(separations)*step_size*n_steps
    my_x = x[:my_data.size]
    comparision_x = x[:my_data.size]
    
    # create the dictionary item to pass to plot()
    acns = {}
    acns[r'My $C_{\phi^2}(\theta = \pi/2)$'] = (my_x, my_data, my_err) # add my data
    acns[r'C.H. $C_{\phi^2}(\theta = \pi/2)$'] = (my_x, comparison_data, comparison_err) # add christians data
    
    # Bundle all data ready for Plot() and store data as .pkl or .json for future use
    all_plot = {'acns':acns, 'lines':{}, 'subtitle':subtitle, 'op_name':op_name}
    return all_plot

import math as math
class Christian_Autocorrelation(object):
    """Written by Christian Hanauer"""
    def __init__(self,data):
        self.data = np.asarray(data)
        self.N = len(data)
        
    def acf(self,t=None):
        #Normalised autocorrelation funciton according to (E.7) in [2]
        mean = np.mean(self.data)
        var = np.sum((self.data - mean) ** 2) / float(self.N)
        if t!=None:
            return ((self.data[:self.N - t] - mean) * (self.data[t:] - mean)).sum() / float(self.N-t) / var
        else:
            def r(t):
                acf_lag = ((self.data[:self.N - t] - mean) * (self.data[t:] - mean)).sum() / float(self.N-t) / var
                return acf_lag
            x = np.arange(self.N) # Avoiding lag 0 calculation
            acf_coeffs = np.asarray(map(r, x))
            return acf_coeffs
    def tauintW(self,plateau_plot=True,spare=100,S=1):
        #calculation of tauintf with findng W according to [1]
        g = 1.
        mathexp = math.exp
        mathsqrt = math.sqrt
        mathlog = math.log
        tint = 0.5
        tplot = [0.5]
        for i in range(1, self.N):
            tint += self.acf(i)
            tplot = np.append(tplot,tint)
            tau = S/mathlog( (2.*tint+1)/(2.*tint-1) )
            g = mathexp(-i/tau) - tau/mathsqrt(i*self.N)
            if g < 0:
                W = i
                thelp = tint
                for j in range(spare):
                     thelp += self.acf(i)
                     tplot = np.append(tplot,thelp)
                break
        if plateau_plot == True:
            fig_plateau = plt.figure()
            ax_plateau = fig_plateau.add_subplot(111)
            ax_plateau.plot(tplot)
            ax_plateau.plot((W, W), (0,tplot[W]))
            ax_plateau.set_xlabel(r'$W$')
            ax_plateau.set_ylabel(r'$\tau_{int}$')
            ax_plateau.set_title(r'$\tau_{int}$ in dependence of $W$')
        
        dtint = np.sqrt(4*(W + 0.5-tint)/float(self.N)) * tint
        return tint,dtint,W
        

# load data from christian and use his data
comparison_xx = load(comparison_loc)
comparison_xx = np.asarray(comparison_xx)

print 'Comparing autocorrelation calculations...'
# assert that the autocorrelation routine is the same
av_xx = comparison_xx.mean()
norm = ((comparison_xx-av_xx)**2).mean()
my_acorr = np.asarray(map(lambda s: myAcorr(comparison_xx, av_xx, s), np.asarray(separations)))

christian_class = Christian_Autocorrelation(comparison_xx)
christian_acorr = christian_class.acf()[:c_len]

christian_acorr = np.asarray(christian_acorr)
diffs = christian_acorr[:my_acorr.size] - my_acorr
print " > av. difference: {}".format(diffs.mean())

print 'Checking integration window calculation:'
christian_tint, christian_dtint, christian_w = christian_class.tauintW(False)
_,_,my_w = windowing(comparison_xx, av_xx, 1.0, comparison_xx.size, fast=True)

print "Christian Window:{}".format(christian_w)
print "My Window:{}".format(my_w)

ans = uWerr(comparison_xx, acorr=my_acorr)   # get errors
_, _, _, itau, itau_diff, _, acns = ans             # extract data
my_w = getW(itau, itau_diff, n=comparison_xx.size)       # get window length

my_err = acorrnErr(acns, my_w, comparison_xx.size)     # get autocorr errors
christian_acorr = np.asarray(christian_acorr)
christian_err = acorrnErr(christian_acorr, christian_w, comparison_xx.size)

all_plot = preparePlot(christian_acorr[:2*my_w], acns[:2*my_w], my_err[:2*my_w], christian_err[:2*my_w])
store(all_plot, file_name, '_allPlot')
plot(save = saveOrDisplay(save, file_name+"_compareAc"), **all_plot)