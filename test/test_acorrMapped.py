from scipy.special import binom
import numpy as np
from collections import Counter

import sys, traceback

def testFn(results, test, res_pairs=None):
    """Takes in two arrays of your function results and compares to test data"""
    outcomes = ["Failed","Passed"]
    
    if res_pairs is None: res_pairs = len(test)*[None]
    for i,(r,t, pairs) in enumerate(zip(results, test, res_pairs)):
        try:
            np.testing.assert_almost_equal(r,t)
            passed = True
        except:
            passed = False
        pr = "  test:{} :: {} :: res: {:6.3f} actual: {:6.3f}".format(i+1, outcomes[passed], r, t)
        if pairs is not None: pr += " pairs: "+repr(["{:3.1f},{:3.1f}".format(i,j) for i,j in pairs])
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
    cases = [a,b,c,d]
    
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
    
    test_set1 = [t1a, t1b, t1c, t1d]
    test_set2 = [t2a, t2b, t2c, t2d]
    return cases, test_set1, test_set2
    
def debugRoutine(func):
    """include **kwargs with func if it doesn't have same as mine
    
    Required Inputs
        sep :: float :: separation
        cases :: as defined above
    """
    seps = [0.1, 0.0]
    cases, test_set1, test_set2 = genTestData()
    
    for test, sep in zip([test_set1, test_set2], seps):
        print 'separation of {}'.format(sep)
        res = []
        res_pairs = []
        for i, arr in enumerate(cases):
            print "test",i
            try:
                mean = arr.mean()
                sol, pairs = func(arr, sep=sep, mean=mean, n=arr.size, debug=True)
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

def attempt(arr, sep, mean, n, tol=1e-7, debug=True):
    pairs = []
    if sep == 0: # fast return
        # this is a lot faster than np.unique as latter requires a mask over counts>1
        # must be one variable if np.array([])
        uniqueElms_counts = np.asarray([(v,c) for v,c in Counter(arr).iteritems() if c>1])
        
        if not uniqueElms_counts.size: # handle no unique items
            result = np.nan
            return (result, pairs) if debug else result
        
        combinations = binom(uniqueElms_counts[:,1],2)
        
        result = ((uniqueElms_counts[:,0]-mean)**2*combinations).sum() / combinations.sum()
        
        if debug: pairs = np.asarray([uniqueElms_counts[:,0]]*2).T
        return (result, pairs) if debug else result
    
    front = 0   # front "pythony-pointer-thing"
    back  = 0   # back "pythony-pointer-thing"
    ans = 0.0
    counter = 0.0
    lst_f = np.nan
    while front < n:   # keep going until exhausted array
        diff = arr[front]-arr[back]
        
        if abs(diff-sep) < tol:     # if equal subject to tol: pair found
            counter += 1
            
            # calculate the correlation function for matched pairs
            ans += (arr[front] - mean)*(arr[back] - mean)
            if debug: pairs.append([arr[back], arr[front]])
            
            print "front {}, back {}, diff {}".format(front, back, abs(arr[front]-lst_f))
            while abs(arr[front]-lst_f) < tol: # hold back while front scans dups
                print "scanning front",front
                
                # calculate the correlation function for matched pairs
                ans += (arr[front] - mean)*(arr[back] - mean)
                if debug: pairs.append([arr[back], arr[front]])
                
               
                if front!=n-1: front+=1
                else: back+=1               # revert to moving back if front at limit
            
            if front-1==back: front+=1
            elif front>back: back+=1        # always the case
            else: raise ValueError('Expected not to be called: front:{}, back:{}'.format(front, back))
            
            lst_f = arr[front]
        
        elif diff > sep: back+=1    # close diff with back
        elif front>back: back+=1   # front is greater so back can catch up
        elif front==back:front+=1
        else: raise ValueError('Expected not to be called: front:{}, back:{}'.format(front, back))
    
    # note that when no pairs are detected we cannot make any statement
    if counter == 0:
        result = np.nan
        return (result, pairs) if debug else result
    
    result = ans/float(counter)
    return (result, pairs) if debug else result

def attempt2(arr, sep, mean, n, tol=1e-7, debug=True):
    pairs = []
    if sep == 0: # fast return
        # this is a lot faster than np.unique as latter requires a mask over counts>1
        # must be one variable if np.array([])
        uniqueElms_counts = np.asarray([(v,c) for v,c in Counter(arr).iteritems() if c>1])
        
        if not uniqueElms_counts.size: # handle no unique items
            result = np.nan
            return (result, pairs) if debug else result
        
        combinations = binom(uniqueElms_counts[:,1],2)
        
        result = ((uniqueElms_counts[:,0]-mean)**2*combinations).sum() / combinations.sum()
        
        if debug: pairs = np.asarray([uniqueElms_counts[:,0]]*2).T
        return (result, pairs) if debug else result
    
    front = 1   # front "pythony-pointer-thing"
    back  = 0   # back "pythony-pointer-thing"
    bssp  = 0   # back sweep start point
    bsep  = 0   # back sweep end point
    ans   = 0.0
    count = 0.0
    while front < n:   # keep going until exhausted array
        diff = arr[front] - arr[back]
        if debug: print "f {}, b {}, bssp {}, bsep {} :: diff {}".format(
            front, back, bssp, bsep, abs(arr[front]-arr[back])<tol)
        if abs(diff-sep) < tol:     # if equal subject to tol: pair found
            if arr[front] - arr[front-1] < tol:
                bssp = sep   # move sweet start point
                back = sep   # and back to last front point
                bsep = front # send start end point to front's position
            else: 
                bssp = front
            
            while back < bsep:
                count += 1
                # calculate the correlation function for matched pairs
                ans += (arr[front] - mean)*(arr[back] - mean)
                
                if debug: pairs.append([arr[back], arr[front]])
                back += 1
            back = bssp # reset back to the sweep start point
        front +=1
    
    # note that when no pairs are detected we cannot make any statement
    if count == 0:
        result = np.nan
        return (result, pairs) if debug else result
    
    result = ans/float(count)
    return (result, pairs) if debug else result
if __name__ == "__main__":
    
    debugRoutine(attempt2)