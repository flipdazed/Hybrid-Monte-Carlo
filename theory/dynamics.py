from scipy.special import j0, jn, erfc
from numpy import exp, pi, sin, cos, sqrt, array

__doc__ == """
References
    [1] : ADK, BP, `Acceptances and Auto Correlations in Hybrid Monte Carlo'
    [2] : ADK, BP, `Cost of the generalised hybrid Monte Carlo algorithm for free field theory'
"""

def accKHMC1dfree_opt(dtau, m, n):
    """Acceptance probability for KHMC as given on pg.37 of [2]
    This is only valid at the optimal parameter choices
    for the mixing angle, m and theta
    
    Required Input
        dtau :: float :: the Leap Frog step length
        m    :: float :: mass
        n   :: int   :: number of lattice sites    
    """
    x = n*dtau**6
    ans = 1. - sqrt(.625/pi)*(1 + 9./20.*m**2 + 39./800.*m**4)*sqrt(x)
    return ans

def accLMC1dFree(dtau, m, n):
    """Acceptance probability routines for Langevin Monte Carlo: 
        (HMC with 1 Leap Frog step) in one dimension
         -- for the case of the free field
    
    As defined in [1], Eq. (2.3).
    
    Required Input
        dtau :: float :: the Leap Frog step length
        m    :: float :: mass
        n   :: int   :: number of lattice sites    
    """
    sigma = 20. + 18.*m**2 + 6.*m**4 + m**6
    return erfc(.125*dtau**3*sqrt(.5*n*sigma))

def accHMC1dFree(tau, dtau, m, lattice_p=array([])):
    """Acceptance probability routine for Hybrid Monte Carlo
     -- for the case of the free field
    
    Required Input
        dtau :: float :: the Leap Frog step length
        tau  :: float :: the trajectory length, $n*\delta \tau$
        m    :: float :: mass
    
    Optional Input
        lattice_p :: np.ndarray :: momentum at each lattice site
    
    The massless soln is defined in [1] as $\tau_0$. The subscript
    denoting that this is valid for 0th order 
    Leap Frog integration
    
    Notes
        1. If m == 0: no lattice required
           If m != 0: requires lattice
        3. To clarify: $\tau / \delta\tau = n$ where $n$ is the 
        number of Leap Frog steps for each trajectory
    """
    
    n = lattice_p.size
    if (m == 0) & (lattice_p.size >= 10):
        c = 4.*tau
        sigma = .75 - .75*j0(c) + jn(2, c) - .25*jn(4,c)
    else:
        if lattice_p.size == 0:
            raise ValueError('lattice velocities required if m != 0 or n < 10')
        p = lattice_p.ravel()
        a = m**2 + 4*sin(np.pi*p/n)**2
        sigma = .125*(a**2*(1. - cos(2.*tau*sqrt(a)))).mean()
    
    return erfc(.25*dtau**2*sqrt(.5*n*sigma))