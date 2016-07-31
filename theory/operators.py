from __future__ import division
from numpy import exp, real, cos, sin, pi, array, log, sqrt

__doc__ = """
References
    [1] : ADK, BP, `Acceptances and Auto Correlations in Hybrid Monte Carlo'
    [2] : ADK, BP, `Cost of the generalised hybrid Monte Carlo algorithm for free field theory'
    [3] : C&F, `Statistical Approach to Quantum Mechanics`
    
::Abbreviations::
1d  - 1 dimensional lattice theories
f   - free field theory
"""

# defined in equation (C.14) of [3]
fnW2 = lambda mu, a: mu**2*(1 + .25*(a*mu)**2)
fnW = lambda mu, a: mu*sqrt(1 + .25*(a*mu)**2)

# defined in equation (C.23) of [3]
fnR = lambda mu, a: 1 + 0.5*(a*mu)**2 - a*fnW(mu, a)

def phi2_1df(mu, n, a, sep):
    """1D Magnetisation as calculated Appendix C of [3]
    
    This is the 1D version of <M^2> = <\phi^2>
    the theoretical prediction of the 1D 2-point
    correlation function for w=m
    
    Required Inputs
        mu  :: float :: mass as defined in Eq C.1
        n   :: int   :: size of lattice
        a   :: float :: lattice spacing
        sep :: int   :: lattice separation of the two points
    """
    r = fnR(mu, a)
    w = fnW(mu, a)
    
    if separation == 0:
        ratio = (1. + r**n)/(1. - r**n)
    else:
        ratio = (r**sep + r**(n-sep))/(1. - r**n)
        
    return .5/w*ratio