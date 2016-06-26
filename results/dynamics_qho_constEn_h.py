import dynamics_qho_constEn_1d

import os
from hmc.potentials import Quantum_Harmonic_Oscillator as QHO

n_steps   = 500
step_size = .01

print 'Running Model'
pot = QHO()
kins, pots = dynamicalEnergyChange(pot, n_steps, step_size)
print 'Finished Running Model'

f_name = os.path.basename(__file__)
save_name = os.path.splitext(f_name)[0] + '.png'

# all_lines = False only plots Hamiltonian
dynamics_qho_constEn_1d.plot(y1 = kins, y2 = pots, all_lines=False,
    save = save_name
    # save = False
    )