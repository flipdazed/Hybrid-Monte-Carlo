import numpy as np
import sys, traceback

from lattice import Periodic_Lattice

class Lattice_Quantum_Harmonic_Oscillator(object):
    """Quantum Harmonic Oscillator on a lattice
    
    H = \frac{m}{2}\dot{x}^2 + V(x)
    
    V(x) = \frac{1}{2}mx^2 + \frac{1}{3!}\lambda_3 x^3 + \frac{1}{4!}\lambda_4 x^4
    
    Required Inputs
        lattice :: class :: see lattice.py for info
    
    Optional Inputs
        m       :: float :: mass
        phi_3   :: phi_3 coupling constant
        phi_4   :: phi_4 coupling constant
    """
    def __init__(self, lattice, m=1., phi_3=0., phi_4=0.):
        self.m = m
        self.lattice = lattice
        
        self.phi_3 = phi_3      # phi^3 coupling const.
        self.phi_4 = phi_4      # phi^4 coupling const.
        
        self.kE = lambda p: self.kineticEnergy(p)
        self.uE = lambda i: self.potentialEnergy(i)
        self.dkE = lambda p: self.gradKineticEnergy(p)
        self.duE = lambda i: self.gradPotentialEnergy(i)
        
        self.all = [self.kE, self.uE, self.dkE, self.duE]
        pass
    
    def kineticEnergy(self, p):
        """n-dim KE
        
        This is the kinetic energy for the shadow hamiltonian
        
        Required Inputs
            p :: np.matrix (col vector) :: momentum vector
        """
        return .5 * np.square(p).flatten().sum(axis=0)
    
    def potentialEnergy(self):
        """n-dim potential
        
        This is the action. In HMC, the action
        is the potential in the shadow hamiltonian.
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            idx   :: integer :: lattice position
        """
        lattice = self.lattice.get # shortcut for brevity
        
        euclidean_action = 0.
        # sum (integrate) across euclidean-space (i.e. all lattice sites)
        for idx in np.ndindex(lattice.shape):
            x_sq = np.dot(lattice[idx].T, lattice[idx])
            # k_e = .5 * self.m # optionally can use velocity but less stable
            # k_e *= self.lattice.gradSquared(idx, a_power=2) # (dx/dt)^2
            
            #### free action S_0: 1/2 \phi(m^2 - \nabla)\phi 
            k_e = - .5 * self.lattice.laplacian(idx, a_power=1)  # lap^2_L
            k_e *= np.dot(lattice[idx], k_e) # 2nd x term not acted on by Lap_L^2
            
            # gradient should be an array of the length of degrees of freedom 
            assert (k_e.shape == (self.lattice.d,))
            
            u_0 = .5 * self.m**2    # mass term: 1/2 * m^2
            u_0 *= x_sq             # position at t=i: x(t)^2
            u_0 *= self.lattice.spacing
            ### End free action
            
            # phi^3 term
            u_3 = self.phi_3 * x_sq * lattice[idx]
            u_3 /= np.math.factorial(3)
            
            # phi^4 terms
            u_4 = self.phi_4 * np.dot(x_sq.T, x_sq)
            u_4 /= np.math.factorial(4)
            
            euclidean_action += k_e + u_0 + u_3 + u_4
        
        return euclidean_action
    
    def gradKineticEnergy(self, p):
        """Gradient w.r.t. conjugate momentum"""
        return p
    
    def gradPotentialEnergy(self, idx):
        """Gradient of the action
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            idx   :: integer :: lattice position
        """
        lattice = self.lattice.get # shortcut
        
        x_sq = np.dot(lattice[idx].T, lattice[idx])
        
        # kinetic term:  p/m = m * v[i] = sqrt(v[i]^2)
        # dh_dp = self.m * np.sqrt(self.lattice.gradSquared(idx, a_power=2))
        dke_dx = -.5* self.lattice.laplacian(idx, a_power=1)
        
        # mass term (phi^2)
        du0_dx = self.m**2 * lattice[idx] * self.lattice.spacing
        
        # phi^3
        du3_dx = self.phi_3 * x_sq
        du3_dx /= np.math.factorial(2)
        
        # phi^4
        du4_dx = self.phi_4 * x_sq * lattice[idx]
        du4_dx /= np.math.factorial(3)
        
        dS_dx = dke_dx + du0_dx + du3_dx + du4_dx
        return dS_dx
    
    def hamiltonian(self, p):
        h = self.kineticEnergy(p) + self.potentialEnergy()
        try:
            assert h.shape == (1,)*len(h.shape) # check 1 dimensional
        except Exception, e:
            _, _, tb = sys.exc_info()
            print '\n hamiltonian() not scalar:'
            traceback.print_tb(tb) # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            print 'line {} in {}'.format(line, text)
            print 'shape: {}'.format(h.shape)
            sys.exit(1)
        return h.reshape(1)