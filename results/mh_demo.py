import numpy as np
from numpy.random import uniform as U
# MH sampling from a generic distribution, q
def metropolisHastings(samples,q,phi):
    phi = np.array(phi)
    chain = [phi]
    accRej = lambda old,new : min([q(*new)/q(*old),1])
    for i in xrange(samples):
        proposal = phi + U(-1,1,phi.size).reshape(phi.shape)
        if U(0,1) < accRej(phi,proposal):
            phi = proposal
        chain.append(phi) # append old phi if rejected
    return np.array(chain) # phi_{t,x}

# sample q
q = lambda x,y: np.exp(-abs(x**2+y**2-50)/10.) # an interesting distribution
chain = np.array(metropolisHastings(1000,q,[0,0])) # make 1000 samples

# the rest is just to make the plot pretty
from plotter import Pretty_Plotter, PLOT_LOC, viridis
import matplotlib.pyplot as plt
from common.utils import saveOrDisplay
res = np.linspace(-10,10,1000)
x,y = np.meshgrid(res,res)

pp = Pretty_Plotter()
pp._teXify() # LaTeX

fig = plt.figure(figsize=(8, 8)) # make plot
fig, ax = plt.subplots(1, figsize = (8, 8))

small = 1e-1
pot = q(x,y).T
mask = pot<small
plot_q = np.ma.MaskedArray(pot, mask, fill_value=np.nan)
plot_x = np.ma.MaskedArray(x, mask, fill_value=np.nan)
plot_y = np.ma.MaskedArray(y, mask, fill_value=np.nan)
ax.contourf(plot_x,plot_y, plot_q, antialiased=True, alpha=0.4, cmap=viridis, label=r'$Q(x,y) = e^{-|x^2+y^2-50|/10}$')
ax.plot(*chain.T, alpha=0.5, color='red') # must transpose to make shape (2,1000) for *samples to work
ax.scatter(*(chain.T[:,0]), marker='o', s=100.0, alpha=0.5, color='green', label='Start') # must transpose to make shape (2,1000) for *samples to work
ax.scatter(*(chain.T[:,-1]), marker='o', s=100.0, alpha=0.5, color='blue', label='End') # must transpose to make shape (2,1000) for *samples to work
ax.legend(loc='best', shadow=True, fontsize = 12)
plt.grid('off')
# ax.set_ylabel('y')
# ax.set_xlabel('x')
plt.axis('off')

save_name = __file__
save      = True
pp.save_or_show(saveOrDisplay(save, save_name), PLOT_LOC)
