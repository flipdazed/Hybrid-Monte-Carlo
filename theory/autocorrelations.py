from __future__ import division
from numpy import exp, real, cos, sin
from scipy.signal import tf2zpk
from hmc import checks

__doc__ = """::References::
[1] Cost of the Generalised Hybrid Monte Carlo Algorithm for Free Field Theory
"""

class M2_Fix_UnitAcc(object):
    """A/C of M^2 for tau = fixed
    (Magnetic Suceptibility)
    
    Intended for fixed length trajectories
    
    Required Inputs
        tau   :: float    :: average trajectory length
        m   :: float    :: mass parameter - the lowest frequenxy mode
    
    Optional Inputs
        p_acc :: float :: average acceptance rate
        p_thresh :: float :: point where p_acc \approx 1
    """
    def __init__(self, tau, m, theta=.5*np.pi, p_acc=1, p_thresh=0.95):
        self.setRoots(r, m, theta, p_acc)
        pass
    
    def lapTfm(self, b, tau, m, theta=.5*np.pi):
        """Laplace-transformed function
        
        Required Inputs
            b   :: float :: Laplace-transformed time
            tau   :: float    :: average trajectory length
            m   :: float    :: mass parameter - the lowest frequency mode
        
        Optional Inputs
            theta :: float :: mixing angle
        """
        cos_phi2 = cos(m*tau)**2
        
        if theta==.5*np.pi:
            if self.p_acc > self.p_thresh:
                # Example 7.2.2 and on page 28
                numerator = cos_phi2
                denominator = exp(b*tau) - cos_phi2
            else:
                # last Equation on page 27
                a0 = self.p_acc*cos_phi2 + 1 - self.p_acc
                numerator = a0
                denominator = exp(b*tau) - a0
        else:
            bt = b*tau
            cos_theta = cos(theta)
            cos_theta2 = cos_theta**2
            cos_theta3 = cos_theta**3
            e1bt, e2bt, e3bt = [exp(-i*bt) for i in [1,2,3]]
            
            a0 = (-1 + 2*cos_phi2)*cos_theta2
            if self.p_acc > self.p_thresh:
                
                numerator = cos_phi2*e1bt           \
                            + cos_theta3*e3bt     \
                            - exp2bt*a0             \
                            - exp2bt*cos_phi*cos_theta
                denomintor = (cos_theta2*cos_phi2 + a0*cos_theta + cos_phi2)*e1bt \
                            + (-e2bt*cos_phi2 + e3bt)*cos_theta3 \
                            - e2bt*a1*cos_theta2 \
                            - e2bt*cos_phi2*cos_theta - 1
            else:
                pa = self.p_acc # shortcut
                
                a0 = -pa*cos_phi2 - 1 + pa
                a1 = -2*pa + 1 + 2*pa*cos_phi2
                a2 = +pa*cos_phi2 - 1 + pa
                
                numerator = a0*e1bt \
                            - e3bt*(-1 + 2*pa)*cos_theta3 \
                            + e2bt*a1*cos_theta2    \
                            + e2bt*a2*cos_theta
                denomintor = (a0*cos_theta2 + (-2*pa*cos_phi2 + 1)*cos_theta + a0)*e1bt \
                            + (-e2bt + e2bt*pa + e2bt*pa*cos_phi2 + e3bt - 2*e3bt*pa)*cos_theta3 \
                            + e2bt*a1*cos_theta2 \
                            + e2bt*a2*cos_theta + 1
            numerator *= -1 # non HMC are BOTH multiplied by -1
            
        return numerator / denominator
    
    def setRoots(self, tau, m, theta, p_acc):
        """Gets the roots
        
        Required Inputs
            tau   :: float    :: average trajectory length
            m   :: float    :: mass parameter - the lowest frequency mode
        
        This is really simple as the form is already partial fractioned
        """
        self.tau = tau
        self.m = m
        self.theta = theta
        self.p_acc = p_acc
        
        if theta==.5*np.pi:
            if self.p_acc > self.p_thresh:
                self.roots = cos(m*tau)**2
            else:
                self.roots = p_acc*cos(m*tau)**2 + 1 - p_acc
        else:
            if self.p_acc > self.p_thresh:
                raise ValueError("Not implemented")
            else:
                raise ValueError("Not implemented")
        return self.roots
    
    def eval(self, t, tau=None, m=None):
        """The Inverted Laplace Transform using partial fractioning
        
        To be explicit, this is the function f(t)
        
        Required Inputs
            t      :: float :: time (the inverse of \beta)
        
        Optional Inputs
            tau   :: float    :: average trajectory length
            m   :: float    :: mass parameter - the lowest frequency mode
        
        Note the delta function at zero does not exist as C = 0
        """
        if tau or m is not None:
            if tau is not None: self.tau = tau
            if m is not None: self.m = m
            self.setRoots(self.tau, self.m)
        
        b1 = self.roots
        m, tau = self.m, self.tau # save sapce
        
        # note that incidentally the root, b1, is also the numerator
        n = t/tau # the number of HMC trajectories (samples)
        ans = b1*np.exp(-(n+1)*np.log(b1))
        return ans
    
#
class M2_Exp_UnitAcc(object):
    """A/C of M^2 for tau ~ Geom
    (Magnetic Suceptibility)
    
    Intended for exponentially distributed trajectories*
    Explicitly calculated in Appendix A.3 of [1]
    
    Required Inputs
        r   :: float    :: inverse average trajectory length
        m   :: float    :: mass parameter - the lowest frequenxy mode
    
    Optional Inputs
        p_acc :: float :: average acceptance rate
        p_thresh :: float :: point where p_acc \approx 1
    
    * An approximation for small r where Geom(r) \approx Expon(r)
    """
    def __init__(self, r, m, p_acc=1, p_thresh=0.95):
        self.setRoots(r, m)
        pass
    
    def lapTfm(self, b, m):
        """Laplace-transformed function
        
        Required Inputs
            b   :: float :: Laplace-transformed time
            m   :: float    :: mass parameter - the lowest frequency mode
        """
        numerator   = b**3 + 2*b**2*b + b*b**2 + 2*m**2*b
        denominator = b**3 + 2*b*b**2 +(b**2 + 4*m**2)*b + 2*m**2*b
        return numerator / denominator
    
    def setRoots(self, r, m, verbose=False):
        """Finds the roots from the un-partialled
        fraction form using scipy
        
        Required Inputs
            r   :: float    :: inverse average trajectory length
            m   :: float    :: mass parameter - the lowest frequency mode
        
        Optional Inputs
            verbose :: bool :: prints out residues, poles and constant from
                                scipy's tf2zpk function
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
        
        Required Inputs
            r   :: float    :: inverse average trajectory length
            m   :: float    :: mass parameter - the lowest frequency mode
        """
        numerator = [r, 2*r**2, 2*m**2*r+r**3]
        denominator = [1, 2*r, r**2 + 4*m**2, 2*m**2*r]
        return numerator, denominator
    
    def eval(self, t, r=None, m=None):
        """The Inverted Laplace Transform using partial fractioning
        
        To be explicit, this is the function f(t)
        
        Required Inputs
            t      :: float :: time (the inverse of \beta)
        
        Optional Inputs
            r   :: float    :: inverse average trajectory length
            m   :: float    :: mass parameter - the lowest frequency mode
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
#