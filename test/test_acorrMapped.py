import numpy as np
import sys, traceback
from scipy.special import binom

def testFn(results, test, res_pairs=None):
    """Takes in two arrays of your function results and compares to test data"""
    outcomes = ["Failed","Passed"]
    
    print "\nTest outcomes..."
    if res_pairs is None: res_pairs = len(test)*[None]
    for i,(r,t, pairs) in enumerate(zip(results, test, res_pairs)):
        try:
            np.testing.assert_almost_equal(r,t)
            passed = True
        except:
            passed = False
        pr = "  test:{} :: {} :: res: {:6.3f} actual: {:6.3f}".format(i+1, outcomes[passed], r, t)
        if pairs is not None: pr += " pairs: "+" ".join(["({:3.1f} {:3.1f})".format(i,j) for i,j in pairs])
        print pr
    pass

def genTestData():
    """Generate test data
        test_cases :: list of 4 test arrays each of length 10
        test_set1  :: the four test cases with 0.1 separation
        test_set2  :: the four test cases with no separation
    """
    n = 10
    
    # Examples to catch most common errors
    a = np.array([0.1]*10)         # case of everything the same
    b = np.linspace(0.1, 1, 10)    # everything spaced equally
    c = np.array([0.1]*5+[0.2]*5)  # intersection of two repeating segments
    d = np.array([0.1, 0.2, 0.3] + [0.4]*5 + [0.5]*2) # a mash-up
    e = np.array([0.1]*3 + [0.2]*3 + [0.3]*4) # series of identicals
    
    
    # a quick function used a fair bit in the case of equal incrementation
    equalSpacing = lambda seg, mean, sep: np.sum((seg[:seg.size-sep]-mean)*(seg[sep:]-mean))
    nCr52 = binom(5,2)  # ways of choosing 2 from 5 where order matters
    dm = d.mean()      # d has a lot of cases so declaring saves space
    
    # the test cases for 0.1 separation
    sep = 0.1
    t1a = np.nan
    t1b = equalSpacing(b, b.mean(), 1)/float(n-1)
    t1c = (0.1-c.mean())*(0.2-c.mean()) #*5**2/5**2
    t1d = (equalSpacing(d[:2], dm, 1) + (.3-dm)*(.4-dm)*5.**2 + 2*(.4-dm)*(.5-dm))/(2.+5.**2+2.)
    
    # test caess for 0 separation
    sep = 0.0
    t2a = 0.0
    t2b = np.nan
    t2c = ((0.1-c.mean())**2*nCr52 + (0.2-c.mean())**2*nCr52)/(nCr52+nCr52)
    t2d = ((0.4-dm)**2*nCr52 + (.5-dm)**2)/(nCr52+1)
    
    cases = [a,b,c,d]
    test_set1 = [t1a, t1b, t1c, t1d]
    test_set2 = [t2a, t2b, t2c, t2d]
    return cases, test_set1, test_set2
    
def debugRoutine(func, verbose=False):
    """include **kwargs with func if it doesn't have same as mine
    
    Required Inputs
        sep :: float :: separation
        cases :: as defined above
    """
    seps = [0.1, 0.0]
    cases, test_set1, test_set2 = genTestData()
    
    for test, sep in zip([test_set1, test_set2], seps):
        print '\nStart separation of {}'.format(sep)
        res = []
        res_pairs = []
        for i, arr in enumerate(cases):
            if verbose: print "\nRunning: test {}".format(i+1)
            try:
                mean = arr.mean()
                sol, pairs = func(arr, sep=sep, mean=mean, n=arr.size, debug=True, verbose=verbose)
                res.append(sol)
                res_pairs.append(pairs)
            except Exception as e:
                res.append(False)
                res_pairs.append([])
                
                exc_type, exc_value, exc_traceback = sys.exc_info()
                err = traceback.format_exc().splitlines()[-1]
                line = traceback.extract_tb(exc_traceback)[-1][1]
                expr = traceback.extract_tb(exc_traceback)[-1][-1]
                print '   > Error: test {}, line: {}, type: {}, expr: {}'.format(i+1, line, err, expr)
        
        testFn(res, test, res_pairs)
    pass

if __name__ == "__main__":
    
    from sweeper import attempt
    debugRoutine(attempt, verbose=True)