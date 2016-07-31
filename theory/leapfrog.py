from numpy import exp, pi, sin, cos, sqrt, array, ndarray

__doc__ == """
References
    [1] : ADK, BP, `Acceptances and Auto Correlations in Hybrid Monte Carlo'
    [2] : ADK, BP, `Cost of the generalised hybrid Monte Carlo algorithm for free field theory'
    
::Abbreviations::
av  :: average
H   :: Hamiltonian
si  :: the spectral averaged frequency for $\omega^{2i}$

:: Assumptions ::
L   :: samples directly form the lattice
"""

# coefficients for p as specified in table 1 of [2]
# the index specifies the nth value in $p_{i,1}$ and $k_{i,1}$
pi1 =  array([0.1250,  0.3800, -0.0457,  0.0038,  0.0118, -0.0483,  0.0371,  0.1178, -0.0813])
ki1 =  array([0.0417, -0.0661,  0.0217, -0.0204, -0.0258,  0.0437, -0.0137, -0.0528,  0.1545])

def siL(m, tau, lattice_p, i=2, **kwargs):
    """Take the spectral average directly form the lattice
    
    Required Inputs
        m   :: float :: mass
        tau :: float :: av. trajectory length
        lattice_p :: np.ndarray :: momentum at each lattice site
    
    Optional Inputs
        i   ::  integer :: i in the $\omega^{2i}$ that is to be averaged
    
    Notes:
    $\omega_p^2$ is calculated as in Eq. (65) of [2]
    Then used in Eq. (63)
    """
    n  = lattice_p.size
    w2 = m**2 + 4*sin(np.pi*lattice_p.ravel()/n)**2
    w  = sqrt(w2)
    wi = w2**i
    
    return (w4*sin(tau*w)).mean()
    

def s2(m, tau, t, **kwargs):
    """2nd order spectral average of frequencies
    
    Required Inputs
        m   :: float :: mass
        tau :: float :: av. trajectory length
        t   :: float :: fictitious time
    """
    c = 4.*tau
    m2 = m*m
    m4 = m2*m2
    sigma = 3 - 3*j0(c) + jn(2, c) - jn(4,c) \
            m2*(2 - 2*j0(c) + 4*t*jn(1, c) + jn(2, c)) \
            .5*m4*(1 + (2*t**2 - 1)*j0(c) + 3*t*jn(1, c))
    return sigma
#
def avH(tau, dtau, m, n, i=1, **kwargs):
    """Average hamiltonian in Leap Frog MDMC
    
    Required Inputs
        tau     :: float :: av. trajectory length
        dtau    :: float :: step size
        m       :: float :: mass
        n       :: integer :: number of lattice sites
    """
    raise NotImplemented("What is t?!?")
    if i > 1: 
        raise NotImplemented("Requires implementation for i>1")
    x = n*dtau**(4*i + 4)
    2*pi1[0]**2*x*s2(m, tau, )
    return 