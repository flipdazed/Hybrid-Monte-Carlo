import numpy as np
import matplotlib.pyplot as plt

from correlations.errors import uWerr, acorrnErr, getW
from theory.clibs.autocorrelations.exponential import hmc
from theory.operators import magnetisation_sq
from correlations.acorr import acorrMapped
from results.common.utils import prll_map
from results.data.store import load
from plotter.pretty_plotting import Pretty_Plotter

min_sep   = 0.
max_sep   = 1000.
max_x_view = 10.
max_y_view = 2.

file_desc = 'acorrIssues_mag2_hmc_lowM'
acs = load('results/data/numpy_objs/{}_acs.json'.format(file_desc))
t = load('results/data/numpy_objs/{}_trajs.json'.format(file_desc))
p = load('results/data/other_objs/{}_probs.pkl'.format(file_desc))
s = load('results/data/numpy_objs/{}_samples.json'.format(file_desc))

op_samples = magnetisation_sq(s)
av_op = op_samples.mean()

n, dim    = s.shape[1], 1
spacing   = 1.
n_samples, n_burn_in = s.shape[0], 50
c_len     = 100000
m         = 0.1
n_steps   = 1000
step_size = 1./((3.*np.sqrt(3)-np.sqrt(15))*m/2.)/float(n_steps)
tau       = step_size*n_steps
r         = 1/tau
phi       = tau*m

res       = step_size
tolerance = res/2.-step_size*0.1

print 'loaded {} samples'.format(s.shape[0])
print 'lattice: {}'.format(repr(s.shape[0:]))
print 'total t: {}'.format(t[-1])
print 'acceptance: {}'.format(p)
del t
del p
del acs
# calculate autocorrelations
aFn = lambda s: acorrMapped(op_samples, t, s, av_op, norm=1.0, tol=tolerance, counts=True)
separations = np.linspace(min_sep, max_sep, (max_sep-min_sep)/res+1)

# multicore
result = prll_map(aFn, separations, verbose=True)
a,counts = zip(*result)

a = np.asarray(a)           # the autocorrelation array
counts = np.array(counts)   # the counts at each separation
mask = 1-np.isnan(a)

separations = separations[mask]
a = a[mask]

# grab errors
print 'getting errors...'
ans = uWerr(op_samples, a)
_, _, _, itau, itau_diff, _, acns = ans             # extract data
my_w = getW(itau, itau_diff, n=n_samples)       # get window length
my_err = acorrnErr(acns, my_w, n_samples)     # get autocorr errors
my_err *= np.sqrt(n_samples)/np.sqrt(counts)

# theory calclations
th_x = np.linspace(min_sep, max_sep, 1000)
th = np.array([hmc(p, phi, r, i) for i in th_x])

from scipy.optimize import curve_fit
fn = lambda x, a, b, c: a*np.cos(b*x)**2+c
cos_sq = curve_fit(fn, separations[:50], acns[:50])

pp = Pretty_Plotter()
pp._teXify()
pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
pp._updateRC()

fig = plt.figure(figsize=(8, 8)) # make plot
ax =[]
fig, ax = plt.subplots(2, sharex=True, figsize = (8, 8))
ax[-1].set_xlabel(r'Fictitious time separation, $t : t = m\delta\tau$ for $m\in\mathbb{N}$')

main_title = r'HMC Autocorrelation Data for $M^2$ and $\tau\sim\text{Exp}(1/r)$'
info_title = r'Samples: $10^{:d}$ Lattice: ({:d},{:d}), $m={:3.1f}$, $n={:d}$, '.format(
    int(np.log10(n_samples)), n,dim, m, n_steps)+r'$\delta\tau = \frac{2}{m(3\sqrt{3} - \sqrt{15})}\frac{1}{n}$'
th_label = r'Theory: $\mathcal{C}_{\text{HMC}}(t; P_{\text{acc}}='+ r'{:5.3f}, \phi = {:5.3f}, r={:5.3f}'.format(p, phi, r) +r')$'

fig.suptitle(main_title, fontsize = 14)
ax[0].set_title(info_title, fontsize = 12)

ax[0].errorbar(separations, acns, yerr=my_err, ecolor='k', ms=3, fmt='o', alpha=0.6, label='Measured $\mathcal{C}_{HMC}(t)$')
ax[0].plot(th_x, th, linewidth=2.0, alpha=0.6, label=th_label)
ax[0].plot(th_x, fn(th_x, *cos_sq[0]), linewidth=2.0, alpha=0.6, label=r'${:4.2}\cos^2{:4.2}x + {:4.2}$'.format(*cos_sq[0]))
ax[0].legend(loc='best', shadow=True, fontsize = 12, fancybox=True)
ax[0].set_ylabel(r'Normalised Correlation Function, $\mathcal{C}(t)$')
ax[0].set_xlim([0,max_x_view])
ax[0].set_ylim([0,max_y_view])
ax[0].relim()
ax[0].autoscale_view()

ax[1].set_title(r'Counts of trajectories at each separation - $10^6$ at $t=0$ omitted', fontsize = 12)
ax[1].bar(separations[1:], counts[1:], linewidth=0, alpha=0.6, width = tolerance, label=r'Separation Tolerance = $\delta\tau/2$ i.e. Bins of $\tau$')
ax[1].legend(loc='best', shadow=True, fontsize = 12, fancybox=True)
ax[1].set_ylabel(r'Count')

plt.show()