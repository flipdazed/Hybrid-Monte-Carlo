from __future__ import division
from numpy import cos, exp, sqrt, real
from numpy.polynomial.polynomial import polyroots

__doc__ = """These functions correspond to the inverted functions from the
Mathematica workbooks"""

s = open('theory/__CmainBody.txt', 'r').read()
s = s.replace('\n', '')

def expCunit(t, tau, m, theta):
    """The analytic a/c for GHMC with unit acceptance
        To be explicit, this is the function f(t)
    
    Required Inputs
        t      :: float :: time (the inverse of \beta)
        tau :: float    :: average trajectory length
        m   :: float    :: mass parameter - the lowest frequency mode
        theta :: float  :: mixing angle
    """
    phi = m*tau
    r = 1/tau
    ct = cos(theta)
    r2 = r**2
    r3 = r**3
    r4 = r2**2
    ct2 = ct**2
    ct3 = ct**3
    ct4 = ct2**2
    ct5 = ct**5
    ct6 = ct2**3
    p2 = phi**2
    
    f13 = 1/3.
    pf13 = 2**f13
    f23 = 2/3.
    pf23 = 2**f23
    c0 = -2*r+r*ct+r*ct2
    c0a = f13*c0
    c1 = -r2+12*r2*p2+r2*ct+r2*ct3-r2*ct4
    c3 = 1+1j*sqrt(3)
    c4 = 1-1j*sqrt(3)
    c5 = 2*r3 + 18*r3*p2 - 3*r3*ct - 36*r3*p2*ct - 6*r3*ct2 + 18*r3*p2*ct2  \
        + 14*r3*ct3 - 6*r3*ct4 - 3*r3*ct5 + 2*r3*ct6                        \
        + sqrt( 4*c1**3 + (2*r3 + 18*r3*p2 - 3*r3*ct - 36*r3*p2*ct          \
                - 6*r3*ct2 + 18*r3*p2*ct2 + 14*r3*ct3 - 6*r3*ct4            \
                - 3*r3*ct5 + 2*r3*ct6)**2)
    c6 = 6*pf13
    c6i = 1/c6
    c7 = 3*pf13
    c7i = 1/c7
    c8 = 3*pf23
    c9a = c5**f13
    c9 = c8*c9a
    c10 = c4*c9a
    c11 = c0a-(pf13*c1)/(3*c9a)+c7i*(c5**f13)
    c12 = c0a + (c3*c1)/c9 - c10/c6
    c13 = c0a + (c4*c1)/c9 - (c3*c9a)/c6
    c14 = c0a + (c3*c1)/c9 - c6i*c10
    c15 = c0a + (c4*c1)/c9 - c6i*c3*c9a
    c16 = c0a - (pf13*c1)/(3*c9a) + c9a/c7
    c17 = exp(t*c16)*r2
    c18 = exp(t*c12)*r2
    c19 = exp(t*c13)*r2
    c20 = 2*r - r*ct - r*ct2
    c21 = exp(t*c16)*r
    c22 = exp(t*c12)*r
    c23 = exp(t*c13)*r
    c24 = c7i*(c9a)
    c25 = f13*c20 + c0a
    
    numerator = - c18*c11 + c19*c11 - 2*c18*p2*c11 + 2*c19*p2*c11           \
        + c18*ct*c11 - c19*ct*c11 + c18*ct2*c11 - c19*ct2*c11 - c18*ct3*c11 \
        + c19*ct3*c11 + c17*c14 - c19*c14 + 2*c17*p2*c14 - 2*c19*p2*c14     \
        - c17*ct*c14 + c19*ct*c14 - c17*ct2*c14 + c19*ct2*c14 + c17*ct3*c14 \
        - c19*ct3*c14 + 2*c21*c11*c14 - 2*c22*c11*c14 - c21*ct*c11*c14      \
        + c22*ct*c11*c14 - c21*ct2*c11*c14 + c22*ct2*c11*c14                \
        + exp(t*c16)*c11**2*c14 - exp(t*c12)*c11*c14**2 - c17*c15 + c18*c15 \
        - 2*c17*p2*c15 + 2*c18*p2*c15 + c17*ct*c15 - c18*ct*c15             \
        + c17*ct2*c15 - c18*ct2*c15 - c17*ct3*c15 + c18*ct3*c15             \
        - 2*c21*c11*c15 + 2*c23*c11*c15 + c21*ct*c11*c15 - c23*ct*c11*c15   \
        + c21*ct2*c11*c15 - c23*ct2*c11*c15 - exp(t*c16)*c11**2*c15         \
        + 2*c22*c14*c15 - 2*c23*c14*c15 - c22*ct*c14*c15 + c23*ct*c14*c15   \
        - c22*ct2*c14*c15 + c23*ct2*c14*c15 + exp(t*c12)*c14**2*c15         \
        + exp(t*c13)*c11*c15**2 - exp(t*c13)*c14*c15**2
    
    numerator *= r
    
    denominator  = c25 - (pf13*c1)/(3*c9a) - (c3*c1)/c9 + c24 + c6i*c10
    denominator *= c25 - (pf13*c1)/(3*c9a) - (c4*c1)/c9 + c24 + c6i*c3*c9a
    denominator *= c25 - (c4*c1)/c9 + (c3*c1)/c9 - c6i*c10 + c6i*c3*c9a
    
    ans = numerator/denominator
    return real(ans)

def expC(t, tau, m, theta, pa):
    """The analytic a/c for GHMC with unit acceptance
        To be explicit, this is the function f(t)
    
    Required Inputs
        t      :: float :: time (the inverse of \beta)
        tau :: float    :: average trajectory length
        m   :: float    :: mass parameter - the lowest frequency mode
        theta :: float  :: mixing angle
        pa :: float :: acceptance probability
    """
    phi = m*tau
    r = 1/tau
    
    ct = cos(theta)
    r2 = r*r
    r3 = r2*r
    r4 = r3*r
    r5 = r4*r
    ct2 = ct*ct
    ct3 = ct2*ct
    ct4 = ct3*ct
    ct5 = ct4*ct
    ct6 = ct5*ct
    p2 = phi*phi
    pa2 = pa*pa
    # the Root object polynomial as returned by Mathematica
    # f = poly[0] + poly[1]*x**1 + poly[2]*x**2 + ... + poly[n]*x**n
    poly = [
        (2*pa2*p2*ct3 - 2*pa2*p2*ct - 2*pa*p2*ct3 - 2*pa*p2*ct2         \
            + 2*pa*p2*ct + 2*pa*p2)*r5,
        (-2*pa2*p2*ct3 - 2*pa2*p2*ct + 6*pa*p2*ct3 - 2*pa*p2*ct         \
            + 4*pa*p2 + 2*pa*ct3 - 2*pa*ct - 4*p2*ct3 - 4*p2*ct2            \
            + 4*p2*ct + 4*p2 - ct3 - ct2 + ct + 1)*r4,
        (2*pa*p2*ct2 - 4*pa*p2*ct + 2*pa*p2 + 4*pa*ct3 - 6*pa*ct - 4*p2*ct2 \
            + 4*p2*ct + 8*p2 - 2*ct3 - 3*ct2 + 3*ct + 4)*r3,
        (2*pa*ct3 - 6*pa*ct + 4*p2 - ct3 - 3*ct2 + 3*ct + 6)*r2,
        (-2*pa*ct - ct2 + ct + 4)*r,
        1
    ]
    # Equivalent to rt1 = Root[func & , 1]
    rt1,rt2,rt3,rt4,rt5  = polyroots(poly)
    
    etr1 = exp(t*rt1)
    etr2 = exp(t*rt2)
    etr3 = exp(t*rt3)
    etr4 = exp(t*rt4)
    etr5 = exp(t*rt5)
    rt1p2 = rt1**2
    rt2p2 = rt2**2
    rt3p2 = rt3**2
    rt4p2 = rt4**2
    rt5p2 = rt5**2
    rt1p3 = rt1**3
    rt2p3 = rt2**3
    rt3p3 = rt3**3
    rt4p3 = rt4**3
    rt5p3 = rt5**3
    rt1p4 = rt1**4
    rt2p4 = rt2**4
    rt3p4 = rt3**4
    rt4p4 = rt4**4
    rt5p4 = rt5**4
    
    numerator = eval(s)
    denominator = (rt1-rt2)*(rt1-rt3)*(rt2-rt3)*(rt1-rt4)*(rt2-rt4)*(rt3-rt4)*(rt1-rt5)*(rt2-rt5)*(rt3-rt5)*(rt4-rt5)
    
    ans = numerator/denominator
    return real(ans)
    
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from autocorrelations import M2_Exp
    
    
    tau = 2
    m = M2_Exp(tau, 1)
    
    x = np.linspace(0, 10, 1000)
    y1 = expC(x, tau, 1, np.pi/2, 1)
    y2 = m.eval(x)
    y3 = expCunit(x, tau, 1, np.pi/2)
    
    plt.plot(x, y3/y3[0], label='expCunit', linewidth=4.0, alpha=.2)
    plt.plot(x, y1/y1[0], label='expC', linewidth=2.0, linestyle='--', alpha=.2)
    plt.plot(x, y2/y2[0], label='M2_EXP', linestyle='--', alpha=.2)
        
    plt.legend()
    plt.show()