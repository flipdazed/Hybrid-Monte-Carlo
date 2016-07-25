from __future__ import division
from numpy import cos, exp, sqrt, real

__doc__ = """These functions correspond to the inverted functions from the
Mathematica workbooks"""

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

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from autocorrelations import M2_Exp
    
    tau = 2
    x = np.linspace(0, 10, 1000)
    
    # y = fn(x, np.pi/2, tau, tau)
    y3 = expCunit(x, tau, 1, np.pi/2)
    m = M2_Exp(tau, 1)
    y2 = m.eval(x)
    
    # plt.plot(x, y/y[0], label='long func cython')
    plt.plot(x, y2/y2[0], label='M2_EXP')
    plt.plot(x, y3/y3[0], label='long func')
    plt.legend()
    plt.show()