import numpy as np
import matplotlib.pyplot as plt

from correlations.errors import uWerr, acorrnErr, getW
from theory.operators import x_sq
from correlations.acorr import acorrMapped
from results.common.utils import prll_map
from results.data.store import load
from plotter.pretty_plotting import Pretty_Plotter

min_sep   = 0.
max_sep   = 50
max_x_view = 10.
max_y_view = 2.

file_desc = 'acorr_xx_hmc_nonExp_kg'
acs = load('results/data/numpy_objs/{}_acs.json'.format(file_desc))
t = load('results/data/numpy_objs/{}_trajs.json'.format(file_desc))
p = load('results/data/other_objs/{}_probs.pkl'.format(file_desc))
s = load('results/data/numpy_objs/{}_samples.json'.format(file_desc))

op_samples = x_sq(s)
av_op = op_samples.mean()
tsum = np.cumsum(t)

n, dim    = s.shape[1], 1
spacing   = 1.
n_samples, n_burn_in = s.shape[0], 50
c_len     = 10000
m         = 1.0
n_steps   = 20
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
del acs
# calculate autocorrelations
aFn = lambda s: acorrMapped(op_samples, tsum, s, av_op, norm=1.0, tol=tolerance, counts=True)
separations = np.linspace(min_sep, max_sep, (max_sep-min_sep)/res+1)

# multicore
result = prll_map(aFn, separations, verbose=True)
a,counts = zip(*result)

a = np.asarray(a)           # the autocorrelation array
counts = np.array(counts)   # the counts at each separation

# grab errors
print 'getting errors...'
ans = uWerr(op_samples, a*counts)
_, _, _, itau, itau_diff, _, acns = ans             # extract data
my_w = getW(itau, itau_diff, n=n_samples)       # get window length
my_err = acorrnErr(acns, my_w, n_samples)     # get autocorr errors
my_err *= np.sqrt(n_samples)/np.sqrt(counts)

pp = Pretty_Plotter()
pp._teXify()
pp._updateRC()

fig = plt.figure(figsize=(8, 8)) # make plot
ax =[]
fig, ax = plt.subplots(2, sharex=True, figsize = (8, 8))
ax[-1].set_xlabel(r'Fictitious time separation, $s : s = m\delta\tau$ for $m\in\mathbb{N}$', fontsize=14)

main_title = r'HMC Autocorrelation Data for $X^2$ and $\tau\sim\text{Exp}(1/r)$'
info_title = r'Samples: $10^{:d}$ Lattice: ({:d},{:d}), $m={:3.1f}$, $n={:d}$, '.format(
    int(np.log10(n_samples)), n,dim, m, n_steps)+r'$\delta\tau = \frac{2}{m(3\sqrt{3} - \sqrt{15})}\frac{1}{n}$'

# fig.suptitle(main_title, fontsize = 14)
ax[0].set_title(info_title, fontsize = 14)
mask = ~np.isnan(a)

ax[0].errorbar(separations[mask], a[mask]/a[0], yerr=my_err[mask], ecolor='k', ms=3,
 fmt='o', alpha=0.6, label=r'Measured $\mathcal{C}_{\langle \mathscr{X}^2_{x} \rangle_x}(s)$')
 
ax[0].legend(loc='best', shadow=True, fontsize = 14, fancybox=True)
ax[0].set_ylabel(r'Normalised $\mathcal{C}_{\langle \mathscr{X}^2_{x} \rangle_x}(s)$', fontsize = 14)
ax[0].set_xlim([0,max_x_view])
ax[0].set_ylim([0,max_y_view])
ax[0].relim()
ax[0].autoscale_view()

ax[1].set_title(r'Counts of trajectories at each separation - $10^{:d}$ at $s=0$ omitted'.format(int(np.log10(n_samples))), fontsize = 14)
ax[1].bar(separations[1:], counts[1:], linewidth=0, alpha=0.6, width = tolerance, label=r'Separation Tolerance = $\delta\tau/2$ i.e. Bins of $\tau$')

# ax[1].plot(th_x, fn(th_x, *f[0]), linewidth=2.0, alpha=0.6, label=r'f(s) = ${:4.2}'.format(f[0][0]) \
#     +r'e^{-'+r'{:4.2f}'.format(cos_sq[0][1]) + r's}' + r'{:4.2}$'.format(f[0][2]), c='r')

ax[1].legend(loc='best', shadow=True, fontsize = 14, fancybox=True)
ax[1].set_ylabel(r'Count', fontsize=14)
ax[0].set_ylim([0,1.1])
ax[1].set_ylim(ymin=0)
plt.show()