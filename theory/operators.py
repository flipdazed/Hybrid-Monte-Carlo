from __future__ import division
import numpy as np

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
fnW = lambda mu, a: mu*np.sqrt(1 + .25*(a*mu)**2)

# defined in equation (C.23) of [3]
fnR = lambda mu, a: 1 + 0.5*(a*mu)**2 - a*fnW(mu, a)

def x2_1df(mu, n, a, sep):
    """x^2 as calculated Appendix C of [3]
    
    The theoretical prediction of the 1D 2-point
    correlation function - NOT the magnetisation
    
    Required Inputs
        mu  :: float :: mass as defined in Eq C.1
        n   :: int   :: size of lattice
        a   :: float :: lattice spacing
        sep :: int   :: lattice separation of the two points
    """
    r = fnR(mu, a)
    w = fnW(mu, a)
    
    if sep == 0:
        ratio = (1. + r**n)/(1. - r**n)
    else:
        ratio = (r**sep + r**(n-sep))/(1. - r**n)
        
    return .5/w*ratio

def x_sq(lattice, sum_axes=1):
    """Calculates the magnetisation across a lattice
    
    This is really just a renaming of np.sum(arr ,axis=1)
    
    Required Inputs
        lattice :: np.ndarray :: the lattice
        sum_axes :: int/tuple :: can be tuple of dimensions or integer which
                                assumed samples are in axis 0
    """
    return np.mean(lattice**2, axis=sum_axes)

def magnetisation(lattice, sum_axes=1):
    """Calculates the magnetisation across a lattice
    
    This is really just a renaming of np.sum(arr ,axis=1)
    
    Required Inputs
        lattice :: np.ndarray :: the lattice
        sum_axes :: int/tuple :: can be tuple of dimensions or integer which
                                assumed samples are in axis 0
    """
    return np.mean(lattice, axis=sum_axes)

def magnetisation_sq(lattice, sum_axes=1):
    """Calculates the magnetisation^2 across a lattice
    
    This is really just a renaming of np.sum(arr ,axis=1)^2
    
    Required Inputs
        lattice :: np.ndarray :: the lattice
        sum_axes :: int/tuple :: can be tuple of dimensions or integer which
                                assumed samples are in axis 0
    """
    return np.mean(lattice, axis=sum_axes)**2