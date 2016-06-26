import dynamics_qho_constEn_2d

import os
import numpy as np
from hmc.potentials import Klein_Gordon as KG

step_sizes  = [.001, .5]
steps       = [1, 500]
n_steps     = 25
n_sizes     = 25

steps = np.linspace(steps[0], steps[1], n_steps, True, dtype=int)
step_sizes = np.linspace(step_sizes[0], step_sizes[1], n_sizes, True)
    
print 'Running Model'
pot = KG()
en_diffs = dynamics_qho_constEn_2d.dynamicalEnergyChange(pot, steps, step_sizes)
print 'Finished Running Model'

f_name = os.path.basename(__file__)
save_name = os.path.splitext(f_name)[0] + '.png'

dynamics_qho_constEn_2d.plot(x = steps, y = step_sizes, z = en_diffs,
    save = save_name,
    # save = False
    )