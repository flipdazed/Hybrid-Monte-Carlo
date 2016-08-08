from collections import Counter
import numpy as np
from scipy.special import binom

def attempt(arr, sep, mean, n, tol=1e-7, debug=True, verbose=False):
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
    bsfp  = 0   # back sweep finish point
    ans   = 0.0
    count = 0
    while front < n:   # keep going until exhausted array
        diff = arr[front] - arr[back]
        if verbose: print "f {}, b {}, bssp {}, bsfp {}, valb {:3.1f}, valf {:3.1f} :: diff {}".format(
            front, back, bssp, bsfp, arr[back], arr[front], abs(diff-sep)<tol)
        
        if abs(diff-sep) < tol:     # if equal subject to tol: pair found
            if verbose: print "... prepare sweep: a[f] {:3.1f}, a[f-1] {:3.1f} ".format(arr[front], arr[front-1]),
            
            if not (arr[front] - arr[front-1] < tol):
                if verbose: print "diff = True"
                bssp = bsfp   # move sweet start point
                back = bsfp   # and back to last front point
                bsfp = front # send start end point to front's position
            else:
                if verbose: print "diff = False"
                back = bssp # reset back to the sweep start point
            while back < bsfp:
                
                if verbose: print "... running sweep: b {}".format(back)
                count += 1
                
                # calculate the correlation function for matched pairs
                ans += (arr[front] - mean)*(arr[back] - mean)
                
                if debug: pairs.append([arr[back], arr[front]])
                back += 1
            back = bssp
        front +=1
    
    # note that when no pairs are detected we cannot make any statement
    if count == 0:
        result = np.nan
        return (result, pairs) if debug else result
    
    result = ans/float(count)
    return (result, pairs) if debug else result