import numpy as np
import sys, traceback

from lattice import Periodic_Lattice
import .checks

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
    def __init__(self, lattice, m=1., phi_3=0., phi_4=0., nabla=False):
        self.m = m
        self.lattice = lattice
        self.nabla = nabla
        self.phi_3 = phi_3      # phi^3 coupling const.
        self.phi_4 = phi_4      # phi^4 coupling const.
        
        # constructed like this so that can switch between
        # nabla and non nabla action by reseting self.nabla = False|True
        # at the class.object level
        self.kE  = lambda p: self.kineticEnergy(p)
        self.uE  = lambda i: self.potentialEnergy(i, nabla=self.nabla)
        self.dkE = lambda p: self.gradKineticEnergy(p)
        self.duE = lambda i: self.gradPotentialEnergy(i, nabla=self.nabla)
        
        self.all = [self.kE, self.uE, self.dkE, self.duE]
        pass
    
    def kineticEnergy(self, p):
        """n-dim KE
        
        This is the kinetic energy for the shadow hamiltonian
        
        Required Inputs
            p :: np.matrix (col vector) :: momentum vector
        """
        return .5 * np.square(p).flatten().sum(axis=0)
    
    def potentialEnergy(self, nabla):
        """n-dim potential
        
        This is the action. In HMC, the action
        is the potential in the shadow hamiltonian.
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            idx   :: integer :: lattice position
        """
        lattice = self.lattice.get # shortcut for brevity
        
        x_sq_sum = np.power(lattice.flatten(), 2).sum()
        
        # avoids the need to iteratively evaluate in the loop
        # although more lines of code
        if nabla: 
            
            p_sq_sum = 0.
            # sum (integrate) across euclidean-space (i.e. all lattice sites)
            for idx in np.ndindex(lattice.shape):
                
                x = lattice[idx] # iterates single points of the lattice
                
                # kinetic term: - x * (Lattice laplacian of x) * lattice spacing
                p_sq = self.lattice.laplacian(idx, a_power=1) 
                
                # gradient should be an array of the length of degrees of freedom 
                checks.tryAssertEqual(val1=p_sq.shape, val2=(self.lattice.d,),
                     ' laplacian shape should be 1D array' \
                     + '\nequal in length to the number of lattice dimensions.' \
                     + '\n> p_sq shape: {}'.format(p_sq.shape) \
                     + '\n> lattice dimensions: {}'.format((self.lattice.d,))
                     )
                
                p_sq_sum += np.dot(x.T, p_sq)
                
                # x.p_sq is a scalar
                checks.tryAssertEqual(val1=p_sq_sum, val2=(),
                     'p_sq * x should be scalar.' \
                     + '\n> p_sq \n> x: {}'.format(p_sq, x)
                     )
            
            #### free action S_0: 1/2 \phi(m^2 - \nabla)\phi 
            kinetic *= - .5 * p_sq_sum
            u_0 = .5 * self.m**2 * x_sq_sum
            ### End free action
        
        else: # this part use the action not integrated by parts
            
            v_sq_sum = 0. # initiate velocity squared
            # sum (integrate) across euclidean-space (i.e. all lattice sites)
            for idx in np.ndindex(lattice.shape):
                
                # sum velocity squared
                v_sq = self.lattice.gradSquared(idx, a_power=1)
                
                # gradient should be an array of the length of degrees of freedom 
                checks.tryAssertEqual(val1=v_sq.shape, val2=(),
                     ' derivative^2 shape should be scalar' \
                     + '\n> v_sq shape: {}'.format(v_sq_sum.shape)
                     )
                
                # sum to previous
                v_sq_sum +=  v_sq
            
            #### free action S_0: m/2 \phi(v^2 + m)\phi
            kinetic *= .5 * self.m * v_sq_sum
            u_0 = .5 * self.m**2 * x**2
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
        euclidean_action = kinetic + self.lattice.spacing * potential
        
        return euclidean_action
    
    def gradKineticEnergy(self, p):
        """Gradient w.r.t. conjugate momentum"""
        return p
    
    def gradPotentialEnergy(self, idx, nabla):
        """Gradient of the action
        
        Here the laplacian in the action is used with 1/a
        the potential is then Va
        
        Required Inputs
            idx   :: integer :: lattice position
        """
        lattice = self.lattice.get # shortcut for brevity
        
        derivative = 0.
        
        # avoids the need to iteratively evaluate in the loop
        # although more lines of code
        if nabla: 
                
            x = lattice[idx] # iterates single points of the lattice
            
            # x is a scalar
            checks.tryAssertEqual(val1=x, val2=(),
                 ' x should be scalar.' + '\n> x: {}'.format(x_sq))
            
            #### free action S_0: 1/2 \phi(m^2 - \nabla)\phi 
            
            # kinetic term: - x * (Lattice laplacian of x) * lattice spacing /2
            k_e = - .5 * x * self.lattice.laplacian(idx, a_power=1)
            
            # gradient should be an array of the length of degrees of freedom 
            checks.tryAssertEqual(val1=k_e.shape, val2=(self.lattice.d,),
                 ' kinetic energy shape should be 1D array' \
                 + '\nequal in length to the number of lattice dimensions.' \
                 + '\n> k_e shape: {}'.format(h.shape) \
                 + '\n> lattice dimensions: {}'.format((self.lattice.d,))
                 )
            
            u_0 = .5 * self.m**2    # mass term: 1/2 * m^2
            u_0 *= x**2             # position at t=i: x(t)^2
            
            ### End free action
            
            # phi^3 term
            u_3 = self.phi_3 * x**3 / np.math.factorial(3)
            
            # phi^4 term
            u_4 = self.phi_4 * x**4 / np.math.factorial(4)
            
            # the potential in the action U(x)
            u = u_0 + u_3 + u_4
            
            # multiply the potential by the lattice spacing as required
            euclidean_action = k_e + u
    
    else: # this part use the action not integrated by parts
        
        # sum (integrate) across euclidean-space (i.e. all lattice sites)
        for idx in np.ndindex(lattice.shape):
            
            x = lattice[idx] # iterates single points of the lattice
            
            # x is a scalar
            checks.tryAssertEqual(val1=x, val2=(),
                 ' x should be scalar.' + '\n> x: {}'.format(x_sq))
                 
            #### free action S_0: m/2 \phi(v^2 + m)\phi
            
            # kinetic term 1/2 m v^2 * lattice spacing
            v =  self.lattice.gradSquared(idx, a_power=1)
            k_e *= .5 * self.m * v**2
            
            # gradient should be an array of the length of degrees of freedom 
            checks.tryAssertEqual(val1=k_e.shape, val2=(self.lattice.d,),
                 ' kinetic energy shape should be 1D array' \
                 + '\nequal in length to the number of lattice dimensions.' \
                 + '\n> k_e shape: {}'.format(h.shape) \
                 + '\n> lattice dimensions: {}'.format((self.lattice.d,))
                 )
            
            # mass term: 1/2 m^2 x^2
            u_0 = .5 * self.m**2 * x**2
            ### End free action
            
            # phi^3 term
            u_3 = self.phi_3 * x**3 / np.math.factorial(3)
            
            # phi^4 term
            u_4 = self.phi_4 * x**4 / np.math.factorial(4)
            
            # the potential in the action U(x)
            u = u_0 + u_3 + u_4
            
            # multiply the potential by the lattice spacing as required
            euclidean_action = k_e + self.lattice.spacing * u
        
        return derivative
    
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