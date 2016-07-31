from __future__ import division
from numpy import exp, real, cos, sin, pi, array, log, sqrt
from numpy import asscalar, array, absolute, nansum, abs
from scipy.signal import residuez
    
from _mathematicaFunctions import expCunit, expC
from hmc import checks

__doc__ = """::References::
[1] Cost of the Generalised Hybrid Monte Carlo Algorithm for Free Field Theory

Throughout the code the roots, $B_k$, are referred to as `poles` and the constants
of partial fractioning, $A_k$, are referred to as residues.
"""

#
class M2_Fix(object):
    """A/C of M^2 for tau = fixed
    (Magnetic Suceptibility)
    
    Intended for fixed length trajectories
    
    Optional Inputs
        tau     :: float    :: average trajectory length
        m       :: float    :: mass parameter - the lowest frequenxy mode
        pa      :: float :: average acceptance rate
        theta   :: float :: mixing angle
        p_thresh :: float :: point where pa \approx 1
    """
    def __init__(self, tau=0.1, m=1, pa=1, theta=.5*pi, p_thresh=0.95):
        super(M2_Fix, self).__init__()
        self.p_thresh = p_thresh
        self.setRoots(tau, m, pa, theta)
        pass
        
    def lapTfm(self, b, tau, m, pa, theta=.5*pi):
        """Laplace-transformed function
        
        Required Inputs
            b   :: float    :: Laplace-transformed time
            tau :: float    :: average trajectory length
            m   :: float    :: mass parameter - the lowest frequency mode
            pa  :: float    :: average acceptance probability
        
        Optional Inputs
            theta :: float :: mixing angle
        """
        cos_phi  = cos(m*tau)
        cos_phi2 = cos_phi**2
        
        if theta == .5*pi:
            if pa > self.p_thresh:
                # Example 7.2.2 and on page 28
                numerator = cos_phi2
                denominator = exp(b*tau) - cos_phi2
            else:
                # last Equation on page 27
                a0 = pa*cos_phi2 + 1 - pa
                numerator = a0
                denominator = exp(b*tau) - a0
        else:
            bt = b*tau
            cos_theta = cos(theta)
            cos_theta2 = cos_theta**2
            cos_theta3 = cos_theta**3
            e1bt, e2bt, e3bt = [exp(-i*bt) for i in [1,2,3]]
            
            a0 = -1 + 2*cos_phi2
            if pa > self.p_thresh:
                
                numerator = cos_phi2*e1bt           \
                            + cos_theta3*e3bt       \
                            - e2bt*a0*cos_theta2    \
                            - e2bt*cos_phi*cos_theta
                
                denominator = (cos_theta2*cos_phi2 + a0*cos_theta + cos_phi2)*e1bt   \
                            + (-e2bt*cos_phi2 + e3bt)*cos_theta3                    \
                            - e2bt*a0*cos_theta2                                    \
                            - e2bt*cos_phi2*cos_theta                               \
                            - 1
            else:
                a1 = -pa*cos_phi2 - 1 + pa
                a2 = -2*pa + 1 + 2*pa*cos_phi2
                a3 = +pa*cos_phi2 - 1 + pa
                
                numerator = a1*e1bt                         \
                            - e3bt*(-1 + 2*pa)*cos_theta3   \
                            + e2bt*a2*cos_theta3            \
                            + e2bt*a3*cos_theta
                
                denominator = (a1*cos_theta3 + (-2*pa*cos_phi2 + 1)*cos_theta + a1)*e1bt             \
                            + (-e2bt + e2bt*pa + e2bt*pa*cos_phi2 + e3bt - 2*e3bt*pa)*cos_theta3    \
                            + e2bt*a2*cos_theta3                                                    \
                            + e2bt*a3*cos_theta + 1
            numerator *= -1 # non HMC are BOTH multiplied by -1
            
        return numerator / denominator
    
    def setRoots(self, tau, m, pa, theta):
        """Gets the roots
        
        Required Inputs
            tau :: float    :: average trajectory length
            m   :: float    :: mass parameter - the lowest frequency mode
            pa  :: float    :: average acceptance probability
            theta :: float  :: mixing angle
        
        Numerator and Denominator are defined as polynomial arrays
        increasing in order stating from 0th order and ending at nth order
        """
        self.tau = tau
        self.m = m
        self.theta = theta
        self.pa = pa
        
        if theta==.5*pi: # checked with Mathematica
            # no scipy needed for theta = pi/2
            self.res = self.poles = self.const = None
            
            if pa > self.p_thresh:
                self.poles = array(cos(m*tau)**2)
            else:
                self.poles = array(pa*cos(m*tau)**2 + 1 - pa)
            # poles are 1/B_k as defined in the paper for pi/2 case
        else:
            cos_phi    = cos(m*tau)
            cos_phi2   = cos_phi*cos_phi
            cos_theta  = cos(theta)
            cos_theta2 = cos_theta*cos_theta
            cos_theta3 = cos_theta2*cos_theta
            print "\nWarning: Implementation may be incorrect." \
                + "\n> Read http://tinyurl.com/hjlkgsq for more information"
            
            if pa > self.p_thresh:
                numerator = [ # checked and verified with Mathematica
                    0,
                    -cos_phi2, 
                    2*cos_phi2*cos_theta2 + cos_phi2*cos_theta \
                        - cos_theta2,
                    -cos_theta3]
                
                denominator = [ # checked and verified with Mathematica
                    1,
                    -cos_phi2*cos_theta2 - 2*cos_phi2*cos_theta - cos_phi2 + cos_theta,
                    cos_theta*cos_phi2 + cos_theta3*cos_phi2 + cos_theta2*(-1 + 2*cos_phi2),
                    -cos_theta3]
            else:
                numerator = [ # checked with mathematica
                    0,
                    pa*cos_phi2 - pa + 1,
                    -2*pa*cos_phi2*cos_theta2 - pa*cos_phi2*cos_theta \
                        + 2*pa*cos_theta2 - pa*cos_theta - cos_theta2 \
                        + cos_theta,
                    (-1 + 2*pa)*cos_theta3]
                
                denominator = [
                    1,
                    -pa*cos_phi2*cos_theta2                         \
                        - 2*pa*cos_phi2*cos_theta - pa*cos_phi2     \
                        + pa*cos_theta2 + pa - cos_theta2           \
                        + cos_theta - 1,
                        (-1 + pa + pa*cos_phi2)*cos_theta           \
                            + (1 - 2*pa + 2*pa*cos_phi2)*cos_theta2 \
                            + (-1 + pa + pa*cos_phi2)*cos_theta3,
                    -2*pa*cos_theta3 + cos_theta3]
            self.d = denominator
            self.n = numerator
            self.res, self.poles, self.const = map(array, residuez(numerator, denominator))
        return self.poles
    
    def eval(self, t, tau = None, m = None, pa = None, theta = None):
        """The Inverted Laplace Transform using partial fractioning
        
        To be explicit, this is the function f(t)
        
        Required Inputs
            t      :: float :: time (the inverse of \beta)
        
        Optional Inputs
            tau :: float    :: average trajectory length
            m   :: float    :: mass parameter - the lowest frequency mode
            pa  :: float    :: average acceptance probability
            theta :: float  :: mixing angle
        
        Note the delta function at zero does not exist as C = 0
        """
        if tau is not None: self.tau = tau
        if m is not None: self.m = m
        if pa is not None: self.pa = pa
        if theta is not None: self.theta = theta
        self.setRoots(self.tau, self.m, self.pa, self.theta)
        
        print "\nWarning: Implementation may be incorrect." \
            + "\n> The Fixed autocorrelation doesn't have an analytic solution!"
        
        n = t/self.tau # the number of HMC trajectories (samples)
        b = self.poles
        c = nansum(self.const) if self.const is not None else 0
        
        # For for both P_acc = 1 and P_acc != 1 is the same
        if self.theta==.5*pi:
            req = (absolute(array(self.poles)) < 1).all()
            if not req:
                print '\n\tWarning: Magnitude of roots' \
                 + 'not all < 1:\n\t > {}'.format(self.poles)
            # see http://math.stackexchange.com/q/1869017/272850
            # this uses a different way of summing the geometric series
            # to the more general method below
            
            ans = array(b**(n))
            # in the case where there are muilple roots
            if b.size > 1: ans = nansum(ans, axis=0)
        else:
            req = (absolute(array(self.poles)) < 1).all()
            if not req:
                print '\n\tWarning: Magnitude of roots' \
                 + 'not all > 1:\n\t > {}'.format(self.poles)
            a = self.res
            # see http://math.stackexchange.com/q/1869017/272850
            ans = - array([a_i/b_i*b_i**(n) for a_i,b_i in zip(a, b)])
            
            # in the case where there are muilple roots
            if b.size > 1: ans = nansum(ans, axis=0)
            
            if ans.size > 1: # implement the delta function
                ans[t == 0] = c
            else:
                if t == 0: ans = c
            
        return real(ans)
    
    def integrated(self, tau, m, pa, theta):
        """Regular (inverted) Integrated function
        
        Required Inputs
            tau     :: float    :: average trajectory length
            m       :: float    :: mass parameter - the lowest frequency mode
            pa      :: float    :: average acceptance probability
            theta   :: float    :: mixing angle
        """
        ans = self.lapTfm(b=0, tau=tau, m=m, pa=pa, theta=theta)
        return ans
#
class M2_Exp(object):
    """A/C of M^2 for tau ~ Geom
    (Magnetic Suceptibility)
    
    Intended for exponentially distributed trajectories*
    Explicitly calculated in Appendix A.3 of [1]
    
    Optional Inputs
        tau     :: float    :: average trajectory length
        m       :: float    :: mass parameter - the lowest frequenxy mode
        pa      :: float :: average acceptance rate
        theta   :: float :: mixing angle
        p_thresh :: float :: point where pa \approx 1
    
    * An approximation for small r where Geom(r) \approx Expon(r)
    """
    def __init__(self, tau=2, m=1, pa=1, theta=.5*pi, p_thresh=0.95):
        super(M2_Exp, self).__init__()
        self.p_thresh = p_thresh
        self.setRoots(tau, m, pa, theta)
        pass
    
    def lapTfm(self, b, tau, m, pa, theta=.5*pi):
        """Laplace-transformed function
        
        Required Inputs
            tau :: float    :: average trajectory length
            m   :: float    :: mass parameter - the lowest frequency mode
            pa  :: float    :: average acceptance probability
            theta :: float  :: mixing angle
        """
        
        r = 1./tau
        b2 = b*b
        b3 = b2*b
        r2 = r*r
        r3 = r2*r
        phi = m*tau
        phi2 = phi*phi
        if theta == .5*pi:
            m2 = m*m
            
            if pa > self.p_thresh:
                numerator   = r3 + 2*r2*b + r*b2 + 2*m2*r
                denominator = b3 + 2*r*b2 +(r2 + 4*m2)*b + 2*m2*r
            else:
                numerator   = b2*r + 2*r2*b + (4 - 2*pa)*phi2*r + r3
                denominator = b3 + 2*r*b2 + (4*phi2 + 1)*r2*b + 2*pa*r3*phi2
        else:
            bt = b*tau
            cos_theta = cos(theta)
            cos_theta2 = cos_theta*cos_theta
            cos_theta3 = cos_theta2*cos_theta
            r4 = r2*r2
            r5 = r4*r
            b4 = b2*b2
            b5 = b4*b
            if pa > self.p_thresh:
                a7 = 2 - cos_theta - cos_theta2
                a8 = 1 - cos_theta - cos_theta2 + cos_theta3
                
                numerator = b2*r + b*r2*a7 + r3*(a8 + 2*phi2)
                
                denominator = b3 + b2*r*a7 + b*r2*(4*phi2 + a8)             \
                    + r3*(2*phi2 - 2*phi2*cos_theta2)
            else:
                a0 = 4 + (1 - 2*pa)*cos_theta - cos_theta2
                a1 = 6 + (3 - 6*pa)*cos_theta - 3*cos_theta2 + ( - 1 + 2*pa)*cos_theta3
                
                a2 = (3 - 6*pa + ( - 2 + pa)*( - 2 + 2*pa)*phi2)*cos_theta  \
                    + ( - 3 + ( - 4 + 4*pa)*phi2)*cos_theta2                \
                    + ( - 2 + 4*pa)*cos_theta3                              \
                    + 4 + (8 - 4*pa)*phi2 + a2a + a2b + a2c
                
                a3 = 4 + (8 + 2*pa)*phi2                                    \
                    + (3 - 6*pa + (4 - 4*pa)*phi2)*cos_theta                \
                    + ( - 3 + ( - 4 + 2*pa)*phi2)*cos_theta2                \
                    + ( - 2 + 4*pa)*cos_theta3
                
                a4 = (1 - 2*pa + ( - 2 + pa)*( - 2 + 2*pa)*phi2)*cos_theta  \
                    + ( - 1 + ( - 4 + 4*pa)*phi2)*cos_theta2                \
                    + ( - 1 + 2*pa - 4*( - 1 + pa)**2*phi2)*cos_theta3      \
                    + 1 + (4 - 2*pa)*phi2 + a4a + a4b + a4c
                
                a5 = 1 + (4 + 4*pa)*phi2                                    \
                    + (1 - 2*pa + ( - 4 - 2*pa)*( - 1 + pa)*phi2)*cos_theta \
                    + ( - 1 - 4*phi2)*cos_theta2                            \
                    + ( - 1 + 2*pa + (2 - 2*pa)*( - 2 + pa)*phi2)*cos_theta3
                
                a6 = 2*pa*phi2                                              \
                    - 2*( - 1 + pa)*pa*phi2*cos_theta                       \
                    - 2*pa*phi2*cos_theta2                                  \
                    + 2*( - 1 + pa)*pa*phi2*cos_theta3
                
                numerator = b4*r + b3*r2*a0 + b2*r3*((4 - 2*pa)*phi2 + a1)  \
                    + b*r4*a2 + r5*a4
                
                denominator = b5 + b4*r*a0 + b3*r2*(4*phi2 + a1)            \
                    + b2*r3*a3 + b*r4*a5 + r5*a6
                
        return numerator / denominator
    
    def setRoots(self, tau, m, pa, theta=.5*pi, verbose = False):
        """Laplace-transformed function
        
        Required Inputs
            b   :: float    :: Laplace-transformed time
            tau :: float    :: average trajectory length
            m   :: float    :: mass parameter - the lowest frequency mode
            pa  :: float    :: average acceptance probability
        
        Optional Inputs
            theta :: float :: mixing angle
            verbose :: bool :: prints out residues, poles and constant from
                                scipy's residuez function
        
        Numerator and Denominator are defined as polynomial arrays
        increasing in order stating from 0th order and ending at nth order
        """
        self.tau = tau
        self.r = r = 1./tau
        self.m = m
        self.theta = theta
        self.pa = pa
        
        if theta == .5*pi:
            m2 = m*m
            phi = m*tau
            phi2 = phi*phi
            r2 = r*r
            r3 = r2*r
            
            if pa > self.p_thresh:
                numerator = [r, 2*r2, 2*m2*r + r3]
                denominator = [1, 2*r, r2 + 4*m2, 2*m2*r]
            else:
                numerator = [r, 2*r2, (4 - 2*pa)*phi2*r + r3]
                denominator = [1, 2*r, (4*phi2 + 1)*r2, 2*pa*r3*phi2]
            
            # get the roots
            self.res, self.poles, self.const = map(array, residuez(numerator, denominator))
            if verbose:
                display = lambda t, x: '\n{}: {}'.format(t, x)
                print display('Residues', self.res)
                print display('Poles', self.poles)
                print display('Constant', self.const)
        else:
            if pa > self.p_thresh:
                denominator = numerator = None
            else:
                denominator = numerator = None
        
        self.d = denominator
        self.n = numerator
        return self.poles
    
    def eval(self, t, tau=None, m=None, pa=None, theta=None):
        """The Inverted Laplace Transform using partial fractioning
        
        To be explicit, this is the function f(t)
        
        Required Inputs
            t      :: float :: time (the inverse of \beta)
        
        Optional Inputs
            tau :: float    :: average trajectory length
            m   :: float    :: mass parameter - the lowest frequency mode
            pa  :: float    :: average acceptance probability
            theta :: float  :: mixing angle
        """
        if tau is not None: self.tau = tau
        if m is not None: self.m = m
        if pa is not None: self.pa = pa
        if theta is not None: self.theta = theta
        self.setRoots(self.tau, self.m, self.pa, self.theta)
        
        t = array(t)
        pa = self.pa
        if self.theta == .5*pi:
            if pa > self.p_thresh:
                # here this format has been analytically derived
                # thus we know that b1,b2 are two poles and a third
                # pole exists as a linear combination of b1,b2
                # therefore we discard the first pole
                b1, b2 = self.poles[1:]
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
                
                t2  = -4.*r3*b1                         \
                        + (-2.*b1_2-13.*m2-2.*b1b2)*r2  \
                        + (-7.*b2 - 8.*b1)*m2*r         \
                        + 16.*m4                        \
                        + (-4.*b1_2 - 4.*b1b2)*m2
                
                t3  = (2.*b1b2 + m2)*r2                 \
                        + (7.*b2 + 7.*b1)*m2*r          \
                        + 16.*m4                        \
                        + 4.*b1b2*m2
                
                ans = t1*exp(b1*t) + t2*exp(b2*t) + t3*exp(-(2.*r+b1+b2)*t)
                ans *= r/d
                return real(ans)
            else:
                b = self.poles
                a = self.res
                c = self.const
                if nansum(c) != 0: 
                    print 'Warning: Non-zero constant term:\n\t{}'.format(c)
                ans = array([a_i*exp(b_i*t) for a_i, b_i in zip(a, b)])
                ans = nansum(ans, axis=0)
                return real(ans)
        else:
            if pa > self.p_thresh:
                ans = expCunit(t, self.tau, self.m, self.theta)
                return ans
            else:
                ans = expC(t, self.tau, self.m, self.theta, self.pa)
                return ans
        
    def integrated(self, tau, m, pa, theta):
        """Regular (inverted) Integrated function
        
        Required Inputs
            tau     :: float    :: average trajectory length
            m       :: float    :: mass parameter - the lowest frequency mode
            pa      :: float    :: average acceptance probability
            theta   :: float    :: mixing angle
        
        This needs to be normalised by the inverted function
        """
        ans = self.lapTfm(b=0,tau=tau, m=m, pa=pa, theta=theta)
        norm = self.eval(t=0,tau=tau, m=m, pa=pa, theta=theta)
        ans /= norm
        return ans
#
if "__main__" == __name__:
    
    def test(ex, ac, s, r):
        print '\n{}'.format(s)
        print "Expected: {}".format(ex)
        print "Actual: {}".format(ac)
        print "result {}".format(r)
        pass
    
    e  = 1e-4 # tolerance
    
    print "Testing M2_Fix"
    print 'Testing for P_acc = 0.78'
    # numerator:    {0, 0.992226, -1.41564, 0.481734}
    # denominator:  1 - 2.40751 x + 1.89069 x^2 - 0.481734 x^3
    expected_n = array([0, 0.992226, -1.41564, 0.481734])
    expected_d = array([1, -2.40751, 1.89069, -0.481734])
    m = M2_Fix(tau=0.1, m=1, pa=0.78, theta=pi/10., p_thresh=1.)
    result = (abs(array(m.n) - expected_n)< e).all()
    test(expected_n, m.n, "numerator", result)
    result = (abs(array(m.d) - expected_d)< e).all()
    test(expected_d, m.d, "denominator", result)
    
    print '\n\nTesting for P_acc = 1'
    # numerator:    {0, 0.990033, -1.82806, (5/8 + Sqrt[5]/8)^(3/2)}
    # denominator:  1, -2.81763, 2.67972, -(5/8 + Sqrt[5]/8)^(3/2)
    c0 = (5/8. + sqrt(5)/8.)**(3/2.)
    expected_n = array([0, 0.990033, -1.82806, c0])
    expected_d = array([1, -2.81763, 2.67972, -c0])
    m = M2_Fix(tau=0.1, m=1, pa=1, theta=pi/10., p_thresh=1.)
    result = (abs(array(m.n) - expected_n)< e).all()
    test(expected_n, m.n, "numerator", result)
    result = (abs(array(m.d) - expected_d)< e).all()
    test(expected_d, m.d, "denominator", result)
    
    
    # plot some examples
    import matplotlib.pyplot as plt
    import numpy as np
    import theory.autocorrelations
    reload(theory.autocorrelations)
    import itertools
    
    plt.ion()
    plt.show()
    plt.cla()
    tau = np.linspace(0.01, 0.1, 3)
    theta =  np.pi/np.linspace(2, 10, 3)
    pa = np.linspace(0.2, 1.0, 3)
    x = np.linspace(0, 50, 1001)
    
    m = theory.autocorrelations.M2_Fix(0.1, 1)
    for ta, th, p in list(itertools.product(tau, theta, pa)):
        y = m.eval(x, ta, 1, p, th)
        plt.plot(x, y/y[0], label="tau:{}, theta:{}, pa:{}".format(ta, th, p))
    plt.legend()
    plt.draw()
    
    f1 = m.integrated(ta, 1, 1, th)
    f2 = np.trapz(y/y[0], x)
    print 'Func:', f1
    print 'Numpy integration:',f2
    print 'ratio:', f1/f2
    print 'res:', m.res
    print 'roots:', m.poles
    print 'const:', m.const