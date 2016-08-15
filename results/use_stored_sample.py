import numpy as np
import matplotlib.pyplot as plt

try:
    from theory.clibs.autocorrelations.exponential import hmc
    cpp = True
except:
    cpp = False
    print 'failed to import c++'
from theory.operators import magnetisation_sq
from correlations.acorr import acorrMapped, acorrMapped_noDups
from results.common.utils import prll_map
from results.data.store import load
from plotter import Pretty_Plotter, PLOT_LOC
from common.utils import saveOrDisplay, multiprocessing
import ctypes

acs = load('results/data/numpy_objs/acorrIssues_mag2_hmc_lowM_acs.json')
t   = load('results/data/numpy_objs/acorrIssues_mag2_hmc_lowM_trajs.json')
p   = load('results/data/other_objs/acorrIssues_mag2_hmc_lowM_probs.pkl')
s   = load('results/data/numpy_objs/acorrIssues_mag2_hmc_lowM_samples.json')

save      = True
file_name = 'acorrIssues_mag2_hmc_m0p10_acCount'
op = magnetisation_sq(s)
av_op = op.mean()

n, dim    = 10, 1
spacing   = 1.
n_samples, n_burn_in = 1000000, 50
c_len     = 1000
m         = 0.1
n_steps   = 20
step_size = 1./((3.*np.sqrt(3)-np.sqrt(15))*m/2.)/float(n_steps)
tau       = step_size*n_steps
r         = 1/tau
phi       = tau*m

min_sep   = 0.
max_sep   = 500
res       = step_size
tolerance = res/2.-step_size*0.1

# this line ensures that we skip this number of iterations
# to make sure the process doesn't spwn over 32000 processes
sep_skip  = max(1,int(((max_sep-min_sep)/res+1) // c_len))
print 'skipping {}'.format(sep_skip)

# calculate autocorrelations
shared_array = multiprocessing.Array(ctypes.c_double, op.size)
shared_array = np.ctypeslib.as_array(shared_array.get_obj())
shared_array = shared_array.reshape(op.shape)
shared_array[...] = op[...]

aFn = lambda s: acorrMapped(op, t, s, shared_array, norm=1.0, tol=tolerance, counts=True)
separations = np.linspace(min_sep, max_sep, (max_sep-min_sep)/res+1)
separations = separations[::sep_skip]

# multicore
result = prll_map(aFn, separations, verbose=True)
a,counts = zip(*result)

a = np.asarray(a)           # the autocorrelation array
counts = np.array(counts)   # the counts at each separation

# theory calclations
if cpp:
    th_x = np.linspace(min_sep, max_sep, 1000)
    th = np.array([hmc(p, phi, r, i) for i in th_x])

pp = Pretty_Plotter()
pp._teXify()
pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
pp._updateRC()

fig = plt.figure(figsize=(8, 8)) # make plot
ax =[]
fig, ax = plt.subplots(2, sharex=True, figsize = (8, 8))
ax[-1].set_xlabel(r'Fictitious time separation, $t : t = m\delta\tau$ for $m\in\mathbb{N}$')

main_title = r'HMC Autocorrelation Data for Exponentially Distributed Trajectories'
info_title = r'Samples: $10^{:d}$ Lattice: ({:d},{:d}), $m={:3.1f}$, $n={:d}$, '.format(
    int(np.log10(n_samples)), n,dim, m, n_steps)+r'$\delta\tau = \frac{2}{m(3\sqrt{3} - \sqrt{15})}\frac{1}{n}$'
th_label = r'Theory: $\mathcal{C}_{\text{HMC}}(t; P_{\text{acc}}='+ r'{:5.3f}, \phi = {:5.3f}, r={:5.3f}'.format(p, phi, r) +r')$'

fig.suptitle(main_title, fontsize = 14)
ax[0].set_title(info_title, fontsize = 12)

ax[0].scatter(separations, a, linewidth=2.0, alpha=0.6, label='Measured $\mathcal{C}_{HMC}(t)$')
if cpp: ax[0].plot(th_x, th, linewidth=2.0, alpha=0.6, label=th_label)
ax[0].legend(loc='best', shadow=True, fontsize = 12, fancybox=True)
ax[0].set_ylabel(r'Unnormalised Correlation Function, $\mathcal{C}(t)$')

ax[1].set_title(r'Counts of trajectories at each separation - $10^6$ at $t=0$ omitted', fontsize = 12)
ax[1].bar(separations[1:], counts[1:], linewidth=0, alpha=0.6, width = tolerance, label=r'Separation Tolerance = $\delta\tau/2$ i.e. Bins of $\tau$')
ax[1].legend(loc='best', shadow=True, fontsize = 12, fancybox=True)
ax[1].set_ylabel(r'Count')

pp.save_or_show(saveOrDisplay(save, file_name), PLOT_LOC)