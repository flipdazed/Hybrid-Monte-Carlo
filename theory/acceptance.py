from scipy.special import j0, jn, erfc
from numpy import exp, pi, sin, cos, sqrt, array, ndarray
import leapfrog

__doc__ == """
References
    [1] : ADK, BP, `Acceptances and Auto Correlations in Hybrid Monte Carlo'
    [2] : ADK, BP, `Cost of the generalised hybrid Monte Carlo algorithm for free field theory'
    
::Abbreviations::
1d  - 1 dimensional lattice theories
f   - free field theory
o   - expects that optimal values are chosen for $m$, $\theta$ and $\phi$
V   - only valid for large lattices ( > 10 sites)
L   - samples from the lattice momenta directly
m0  - m=0 assumed
lfi - order i  of leapfrog assumed
"""

def acceptance(dtau, delta_h=None, **kwargs):
    """Theoretical acceptance rate. Eq (22) of [2]
    
    Required Inputs
        dtau :: float :: MDMC step size
    
    Optional Inputs
        delta_h :: float :: average change in hamiltonian
        tau  :: float :: av. trajectory length  : required if delta_h = None
        m    :: float :: mass                   : required if delta_h = None
        n    :: int   :: number of lattice sites: required if delta_h = None
    """
    
    if delta_h is None:
        reqs = ['tau','m','n']
        if all(k in kwargs for k in reqs):
            delta_h = leapfrog.avH(dtau=dtau, **kwargs)
        else:
            raise ValueError('Need to define: tau, m, m if delta_h is None')
    x = .5*sqrt(delta_h)
    return erfc(x)

def KMC1dfolf0(dtau, m, n, **kwargs):
    """Acceptance probability for KMC as given on pg.37 of [2]
    
    This is only valid at the optimal parameter choices
    for the mixing angle, m and phi
    
    Required Input
        dtau :: float :: the Leap Frog step length
        m    :: float :: mass
        n   :: int   :: number of lattice sites
    
    Assumptions
        1 dimensional
        free field
        optimal values
        leapfrog order: 0
    """
    x = n*dtau**6
    ans = 1. - sqrt(.625/pi)*(1 + 9./20.*m**2 + 39./800.*m**4)*sqrt(x)
    return ans

def LMC1dflf0(dtau, m, n, **kwargs):
    """Acceptance probability routines for Langevin Monte Carlo: 
        (HMC with 1 Leap Frog step) in one dimension
    
    As defined in [1], Eq. (2.3).
    
    Required Input
        dtau :: float :: the Leap Frog step length
        m    :: float :: mass
        n   :: int   :: number of lattice sites
    
    Assumptions
        1 dimensional
        free field
        leapfrog order: 0
    """
    sigma = 20. + 18.*m**2 + 6.*m**4 + m**6
    return acceptance(dtau=dtau,delta_h=.5*n*sigma)

def HMC1dfVm0lf0(tau, dtau, n, **kwargs):
    """Acceptance for HMC
    
    Required Input
        tau  :: float :: the trajectory length, $n*\delta \tau$
        dtau :: float :: the Leap Frog step length
        n    :: integer :: lattice size (number of sites)
    
    Assumptions
        1 dimensional
        free field
        high volume
        massless
        leapfrog order: 0
    """
    c = 4.*tau
    sigma = .75 - .75*j0(c) + jn(2, c) - .25*jn(4,c)
    return acceptance(dtau=dtau,delta_h=.125*n*sigma*dtau**4)

def GHMC1dfL(tau, dtau, m, lattice_p, i=0, **kwargs):
    """Acceptance probability routine for Hybrid Monte Carlo
    
    Required Input
        tau  :: float :: the trajectory length, $n*\delta \tau$
        dtau :: float :: the Leap Frog step length
        m    :: float :: mass
        lattice_p :: np.ndarray :: momentum at each lattice site
    
    Optional Input
        i   :: integer :: leapfrog order
    
    Assumptions
        1 dimensional
        free field
        sample momentum from lattice
    
    Notes:
        defined in [2] on page 13 Eq (23)
    """
    if not isinstance(lattice_p, ndarray): raise ValueError('Requires numpy array')
    
    n = lattice_p.size
    spec_av = leapfrog.siL(m=m, tau=tau, lattice_p=lattice_p)
    
    p01_2   = leapfrog.pi1[0]**2
    x       = n*dtau**(4*i+4)
    delta_h = 2*x*p01_2*spec_av
    
    return acceptance(dtau=dtau, delta_h=delta_h)