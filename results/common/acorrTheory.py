from __future__ import division
from numpy import exp, real
from scipy.signal import tf2zpk

__doc__ = """::References::
[1] Cost of the Generalised Hybrid Monte Carlo Algorithm for Free Field Theory
"""

class expTrajM2(object):
    """\phi^2 :: Theoretical autocorrelation
    (Magnetic Suceptibility)
    
    Intended for exponentially distributed trajectories
    Explicitly calculated in Appendix A.3 of [1]
    
    Required Inputs
        r   :: float    :: inverse average trajectory length
        m   :: float    :: mass parameter - the lowest frequenxy mode
    """
    def __init__(self, r, m):
        self.setRoots(r, m)
        pass
    def lapTfm(self, b):
        """Laplace-transformed function
        
        Required Inputs
            b   :: float :: Laplace-transformed time
        """
        numerator   = b**3 + 2*b**2*b + b*b**2 + 2*m**2*b
        denominator = b**3 + 2*b*b**2 +(b**2 + 4*m**2)*b + 2*m**2*b
        return numerator / denominator
    
    def setRoots(self, r, m, verbose=False):
        """Finds the roots from the un-partialled
        fraction form using scipy
        
        Required Inputs
            r   :: float    :: inverse average trajectory length
            m   :: float    :: mass parameter - the lowest frequenxy mode
        """
        self.r = r
        self.m = m
        self.res, self.poles, self.const = tf2zpk(*self.numDenom(r, m))
        
        if verbose:
            display = lambda t, a, b: '\n{}: {}'.format(t, a, b)
            print display('Residues', self.res)
            print display('Poles', self.poles)
            print display('Constant', self.k)
        self.roots = self.poles[:-1]
        return self.roots
    
    def numDenom(self, r, m):
        """The split numerator and denominator
        containing powers of $\beta$ where list[0]
        is the coefficient for the n-1'th power of
         $\beta$ and list[-1] is the coefficient
        for the 0th power of beta
        """
        numerator = [r, 2*r**2, 2*m**2*r+r**3]
        denominator = [1, 2*r, r**2 + 4*m**2, 2*m**2*r]
        return numerator, denominator
    
    def eval(self, t, r=None, m=None):
        """The Inverted Laplace Transform using partial fractioning
        
        To be explicit, this is the function f(t)
        
        Required Inputs
            t      :: float :: time (the inverse of \beta)
        """
        if r or m is not None:
            if r is not None: self.r = r
            if m is not None: self.m = m
            self.setRoots(self.r, self.m)
        
        b1, b2 = self.roots
        m, r = self.m, self.r
        
        m2 = m**2 ; m4 = m**4
        r2 =r**2; r3 = r**3; r4 = r**4
        b1_2 = b1**2
        b1b2 = b1*b2
        d = 2.*r4 - 13.*m2*r2 + 64.*m4
        
        t1  = 2.*r4                          \
                + 4.*r3*b1                   \
                + (2.*b1_2 - m2)*r2          \
                + b1*r*m2                    \
                + 32.*m4 + 4.*b1_2*m2
        
        t2  = -4.*r3*b1                      \
                + (-2.*b1_2-13.*m2-2.*b1b2)*r2 \
                + (-7.*b2 - 8.*b1)*m2*r       \
                + 16.*m4                     \
                + (-4.*b1_2 - 4.*b1b2)*m2
        
        t3  = (2.*b1b2 + m2)*r2              \
                + (7.*b2 + 7.*b1)*m2*r        \
                + 16.*m4                     \
                + 4.*b1b2*m2
        
        ans = t1*exp(b1*t) + t2*exp(b2*t) + t3*exp(-(2.*r+b1+b2)*t)
        ans *= r/d
        
        return real(ans)