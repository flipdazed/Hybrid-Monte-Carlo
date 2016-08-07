from scipy.special import binom
import numpy as np

def testFn(results, test):
    """Takes in two arrays of your function results and compares to test data"""
    outcomes = ["Failed","Passed"]
    for i,(r,t) in enumerate(zip(results, test)):
        try:
            np.testing.assert_equal(r,t)
            passed = True
        except:
            passed = False
        print "test:{} :: {} :: result: {:8.4f}; actual:{:8.4f}".format(i+1, outcomes[passed], r, t)
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
    cases = [a,b,c,d]
    
    # a quick function used a fair bit in the case of equal incrementation
    equalSpacing = lambda seg, mean, sep: np.sum((seg[:seg.size-sep]-mean)*(seg[sep:]-mean))
    nCr52 = binom(5,2)  # ways of choosing 2 from 5 where order matters
    dm = d.mean()      # d has a lot of cases so declaring saves space
    
    # the test cases for 0.1 separation
    sep = 0.1
    t1a = np.nan
    t1b = np.mean(equalSpacing(b, b.mean(), 1))
    t1c = (0.1-c.mean())*(0.2-c.mean()) #*5**2/5**2
    t1d = (equalSpacing(d[:2], dm, 1) + (.3-dm)*(.4-dm)*5.**2 + 2*(.4-dm)*(.5-dm))/(2.+5.**2+2.)
    
    # test caess for 0 separation
    sep = 0.0
    t2a = 0.0
    t2b = np.nan
    t2c = (0.0*nCr52 + (0.1-c.mean())**2*nCr52)/(2*nCr52)
    t2d = ((0.4-dm)**2*nCr52 + (.4-dm)*(.5-dm)*2)/(nCr52+2)
    
    test_set1 = [t1a, t1b, t1c, t1d]
    test_set2 = [t2a, t2b, t2c, t2d]
    return cases, test_set1, test_set2

def attempt(arr, sep, mean, n, tol=1e-7):
    # fast return
    if sep == 0: return ((arr-mean)**2).mean()
    
    front = 0   # front "pythony-pointer-thing"
    back  = 0   # back "pythony-pointer-thing"
    l_ans = 0.0 # left half of pair
    r_ans = 0.0 # right half of pair
    counter = 0.0
    while front < n:   # keep going until exhausted array
        diff = arr[front]-arr[back]
    
        if abs(diff-sep) < tol:     # if equal subject to tol: pair found
        
            # calculate the correlation function for matched pairs
            r_ans += arr[front]
            l_ans += arr[back]
            counter += 1
        
            # I can't remember why I put these non-standard lines in
            # I think it was due to the normal algorithm missing cases...
            if front!=back: back+=1 # don't run off yet!
            elif front==n-1:back+=1 # hold front at n-1 until back gets there 
            else: front+=1          # if front = back budge up front
        elif diff > sep: back+=1    # close diff with back
        else: front+=1              # front is greater so back can catch up
    
    # note that when no pairs are detected we cannot make any statement
    if counter == 0: return np.nan
    
    # saved up operations
    l_ans -= mean*counter
    r_ans -= mean*counter
    
    return l_ans*r_ans

if __name__ == "__main__":
    
    cases, test_set1, test_set2 = genTestData()
    res1  = [attempt(arr, 0.1, arr.mean(), arr.size) for arr in cases]
    res2  = [attempt(arr, 0.0, arr.mean(), arr.size) for arr in cases]
    
    print 'test set 1'
    testFn(res1, test_set1)
    
    print 'test set 2'
    testFn(res2, test_set2)