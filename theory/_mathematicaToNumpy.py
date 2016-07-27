from sympy import Symbol, poly, cos, exp, fraction
from sympy.parsing import mathematica as mm
#
def m2n(fnStr, verbose=True):
    """A function to convert mathematica output into numpy format
    WARNING: suspect this may be buggy.
    
    Required Inputs
        fnStr :: str :: The mathematica string in FortranFormat
    
    Optional Inputs
        verbose :: bool :: True prints to screen so it can be copied
    
    Only supports the output from the two mathematica files in this directory
    
    The best way to use this is inside iPython: This example assumes the
    `Fn // FortranFormat` output from Mathematica is in the clipboard
        
        %paste fnStr
        fnStr = '\n'.join(fnStr)
        n,d = m2n(fnStr, verbose=True)
    
    """
    
    b = Symbol(r'\beta')
    pa = Symbol(r'\rho')
    phi = Symbol(r'\phi')
    theta = Symbol(r'\theta')
    tau = Symbol(r'\tau')
    x = Symbol('x')
    # list if items to remove and replace: [(new, old)]
    replaceLst = [("E**", "exp"), ("Cos", "cos"), ("/exp(b*tau)", "*x"), 
        ('\n',''), (' ', ''), ("--", "+")]
    replaceLst.extend([("/exp(%d*b*tau)"%i, "*x**%d"%i) for i in range(2, 5)])
    replaceLst.extend([("exp(-%d*b*tau)"%i, "x**%d"%i) for i in range(2, 5)])
    
    for old, new in replaceLst:
        fnStr = fnStr.replace(old, new)
    
    f = mm.sympify(fnStr)
    numerator, denominator = [poly(item, x) for item in fraction(f)]
    
    # get coefficients starting lowest order to highest order
    numerator = numerator.all_coeffs()
    denominator = denominator.all_coeffs()
    
    # numpy is in the opposite order
    numerator.reverse()
    denominator.reverse()
    if verbose:
        print '::: polynomial coefficients: Ordering is 0th -> nth :::'
        print 'Numerator Array:\n',numerator
        print '\nDenominator Array:\n',denominator
    else:
        return numerator, denominator