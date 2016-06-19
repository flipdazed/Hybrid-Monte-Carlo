import numpy as np

from lattice import Periodic_Lattice
from . import checks

class Quantum_Harmonic_Oscillator(object):
    """Quantum Harmonic Oscillator on a lattice
    
    H = \frac{m}{2}\dot{x}^2 + V(x)
    
    V(x) = \frac{1}{2}mx^2 + \frac{1}{3!}\lambda_3 x^3 + \frac{1}{4!}\lambda_4 x^4
    
    Optional Inputs
        m       :: float :: mass
        phi_3   :: phi_3 coupling constant
        phi_4   :: phi_4 coupling constant
    """
    def __init__(self, m=1., phi_3=0., phi_4=0., klein_gordon=False):
        self.m = m
        self.klein_gordon = klein_gordon
        self.phi_3 = phi_3      # phi^3 coupling const.
        self.phi_4 = phi_4      # phi^4 coupling const.
        
        # constructed like this so that can switch between
        # klein_gordon and non klein_gordon action by reseting self.klein_gordon = False|True
        # at the class.object level
        self.kE  = lambda p: self.kineticEnergy(p)
        self.uE  = lambda x: self.potentialEnergy(x)
        self.dkE = lambda p: self.gradKineticEnergy(p)
        self.duE = lambda i, x: self.gradPotentialEnergy(i, x)
        
        self.all = [self.kE, self.uE, self.dkE, self.duE]
        pass
    
    def kineticEnergy(self, p):
        """n-dim KE
        
        This is the kinetic energy for the shadow hamiltonian
        
        Required Inputs
            p :: np.array (nd) :: momentum array
        """
        return .5 * np.square(p).flatten().sum(axis=0)
    
    def potentialEnergy(self, positions):
        """n-dim potential
        
        This is the action. In HMC, the action
        is the potential in the shadow hamiltonian.
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            positions :: class :: see lattice.py for info
        """
        lattice = positions.get # shortcut for brevity
        
        x_sq_sum = np.power(lattice.flatten(), 2).sum()
        
        # avoids the need to iteratively evaluate in the loop
        # although more lines of code
        if klein_gordon: 
            
            p_sq_sum = np.array(0.)
            # sum (integrate) across euclidean-space (i.e. all lattice sites)
            for idx in np.ndindex(lattice.shape):
                
                x = lattice[idx] # iterates single points of the lattice
                
                # kinetic term: - x * (Lattice laplacian of x) * lattice spacing
                p_sq = self.lattice.laplacian(idx, a_power=1) 
                
                # gradient should be an array of the length of degrees of freedom 
                checks.tryAssertEqual(p_sq.shape, (),
                     ' laplacian shape should be scalar' \
                     + '\n> p_sq shape: {}'.format(p_sq.shape))
                
                p_sq_sum += np.dot(x.T, p_sq)
                
                # x.p_sq is a scalar
                checks.tryAssertEqual(p_sq_sum.shape, (),
                     'p_sq * x should be scalar.' \
                     + '\n> p_sq: {} \n> x: {}'.format(p_sq, x.T)
                     + '\n> p_sq_sum {}'.format(p_sq_sum)
                     )
            
            #### free action S_0: 1/2 \phi(m^2 - \klein_gordon)\phi 
            kinetic = - .5 * p_sq_sum
            u_0 = .5 * self.m**2 * x_sq_sum
            ### End free action
        
        else: # this part use the action not integrated by parts
            
            v_sq_sum = np.array(0.) # initiate velocity squared
            # sum (integrate) across euclidean-space (i.e. all lattice sites)
            for idx in np.ndindex(lattice.shape):
                
                # sum velocity squared
                v_sq = self.lattice.gradSquared(idx, a_power=1)
                
                # gradient should be an array of the length of degrees of freedom 
                checks.tryAssertEqual(v_sq.shape, (),
                     ' derivative^2 shape should be scalar' \
                     + '\n> v_sq shape: {}'.format(v_sq_sum.shape)
                     )
                     
                # sum to previous
                v_sq_sum +=  v_sq
            
            #### free action S_0: m/2 \phi(v^2 + m)\phi
            kinetic = .5 * self.m * v_sq_sum
            u_0 = .5 * self.m * x_sq_sum
            ### End free action
        
        # Add interation terms if required
        if self.phi_3: # phi^3 term
            x_3_sum = np.power(lattice.flatten(), 3).sum()
            u_3 = self.phi_3 * x_3_sum / np.math.factorial(3)
        else:
            u_3 = 0.
        
        if self.phi_4: # phi^4 term
            x_4_sum = np.power(lattice.flatten(), 4).sum()
            u_3 = self.phi_3 * x_4_sum / np.math.factorial(4)
        else:
            u_4 = 0.
        
        # the potential terms in the action
        potential = u_0 + u_3 + u_4
        
        # multiply the potential by the lattice spacing as required
        euclidean_action = kinetic + positions.spacing * potential
        
        return euclidean_action
    
    def gradKineticEnergy(self, p):
        """Gradient w.r.t. conjugate momentum
        
        Required Inputs
            p :: np.array (nd) :: momentum array
        """
        return p
    
    def gradPotentialEnergy(self, lattice, idx):
        """Gradient of the action
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            idx   :: integer :: lattice position
            positions :: class :: see lattice.py for info
        """
        lattice = positions.get # shortcut for brevity
        
        # used in both klein_gordon and non klein_gordon
        x_sum = lattice.flatten().sum()
        
        # avoids the need to iteratively evaluate in the loop
        # although more lines of code
        if klein_gordon: 
            
            p_sq_sum = np.array(0.)
            # sum (integrate) across euclidean-space (i.e. all lattice sites)
            for idx in np.ndindex(lattice.shape):
                x = lattice[idx] # iterates single points of the lattice
            
                # x is a scalar
                checks.tryAssertEqual(x.shape, (),
                     ' x should be scalar.' + '\n> x: {}'.format(x))
            
                # gradient of kinetic term x \klein_gordon^2 x = 2 \klein_gordon^2 x
                p_sq = self.lattice.gradLaplacian(idx, a_power=1) 
            
                # gradient should be an array of the length of degrees of freedom 
                checks.tryAssertEqual(p_sq.shape, (),
                     ' laplacian shape should be scalar' \
                     + '\n> p_sq shape: {}'.format(p_sq.shape))
                
                # sum across indices
                p_sq_sum += p_sq
            
                # x.p_sq is a scalar
                checks.tryAssertEqual(p_sq_sum.shape, (),
                     'p_sq * x should be scalar.' \
                     + '\n> p_sq: {}'.format(p_sq)
                     + '\n> p_sq_sum {}'.format(p_sq_sum)
                     )
            
            #### grad of free action S_0: 2/2 * (m^2 - \klein_gordon^2)\phi
            kinetic = - p_sq_sum
            u_0 = self.m**2 * x_sum # derivative taken
            ### End free action
        
        else: # this part use the action not integrated by parts
            
            k_e = np.array(0.)
            # sum (integrate) across euclidean-space (i.e. all lattice sites)
            for idx in np.ndindex(lattice.shape):
                
                x = lattice[idx] # iterates single points of the lattice
                
                # x is a scalar
                checks.tryAssertEqual(x.shape, (),
                     ' x should be scalar.' + '\n> x: {}'.format(x))
                
                # derivative of velocity squared
                # the derivative of the velocity squared is actually
                # identical to - \klein_gordon^2
                v_sq =  - self.lattice.gradLaplacian(idx, a_power=1)
                
                # gradient should be an array of the length of degrees of freedom 
                checks.tryAssertEqual(v_sq.shape, (),
                     ' derivative^2 shape should be scalar' \
                     + '\n> v_sq shape: {}'.format(v_sq_sum.shape)
                     )
                     
                # sum to previous
                v_sq_sum +=  v_sq
            
            #### free action S_0: m/2 \phi(v^2 + m)\phi
            kinetic = self.m * v_sq_sum
            u_0 = self.m * x_sum
            ### End free action
        
        # Add interation terms if required
        if self.phi_3: # phi^3 term
            x_sq_sum = np.power(lattice.flatten(), 2).sum()
            u_3 = self.phi_3 * x_sq_sum / np.math.factorial(2)
        else:
            u_3 = 0.
        
        if self.phi_4: # phi^4 term
            x_3_sum = np.power(lattice.flatten(), 3).sum()
            u_4 = self.phi_4 * x_3_sum / np.math.factorial(3)
        else:
            u_4 = 0.
        
        # the potential terms in the action
        potential = u_0 + u_3 + u_4
            
        # multiply the potential by the lattice spacing as required
        derivative = k_e + kinetic + positions.spacing * potential
            
        return derivative
    
    def hamiltonian(self, p, positions):
        """Returns the Hamiltonian
        
        Required Inputs
            p :: np.array (nd) :: momentum array
            positions :: class :: see lattice.py for info
        """
        h = self.kineticEnergy(p) + self.potentialEnergy(lattice)
        # check 1 dimensional
        checks.tryAssertEqual(h.shape, (1,)*len(h.shape),
             ' hamiltonian() not scalar.\n> shape: {}'.format(h.shape))
        return h.reshape(1)