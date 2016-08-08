from collections import Counter
import numpy as np
from scipy.special import binom

def attempt(arr, sep, mean, n, tol=1e-7, debug=True, verbose=False):
    pairs = []
    if sep == 0: # fast return
        # this is a lot faster than np.unique as latter requires a mask over counts>1
        # must be one variable if np.array([])
        uniqueElms_counts = np.asarray([(v,c) for v,c in Counter(arr).iteritems() if c>1])
        
        if not uniqueElms_counts.size:  # handle no unique items
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
    ans   = 0.0 # store the answer
    count = 0   # counter for averaging
    new_front = True # the first front value is new
    
    while front < n:   # keep going until exhausted array
        new_front = (arr[front] - arr[front-1] > tol)       # check if front value is a new one
        back = bsfp if new_front else bssp                  # this is the magical step
        
        diff = arr[front] - arr[back]
        if verbose: print "f {}, b {}, bssp {}, bsfp {}, valb {:3.1f}, valf {:3.1f} :: diff {}".format(
            front, back, bssp, bsfp, arr[back], arr[front], abs(diff-sep)<tol)
        
        if abs(diff-sep) < tol:     # if equal subject to tol: pair found
            if verbose: print "... prepare sweep: a[f] {:3.1f}, a[f-1] {:3.1f} ".format(arr[front], arr[front-1]),
            
            if new_front:
                if verbose: print "diff = True"
                bssp = bsfp     # move sweep start point
                back = bsfp     # and back to last front point
                bsfp = front    # send start end point to front's position
            else:
                if verbose: print "diff = False"
                back = bssp     # reset back to the sweep start point
            while back < bsfp:
                
                if verbose: print "... running sweep: b {}".format(back)
                count += 1
                
                # calculate the correlation function for matched pairs
                ans += (arr[front] - mean)*(arr[back] - mean)
                
                if debug: pairs.append([arr[back], arr[front]])
                back += 1
        else:
            if abs(arr[bssp+1]- arr[bssp]) > tol: bsfp = front
        
        front +=1
    # note that when no pairs are detected we cannot make any statement
    if count == 0:
        result = np.nan
        return (result, pairs) if debug else result
    
    result = ans/float(count)
    return (result, pairs) if debug else result

def attemptShort(arr, sep, mean, n, tol=1e-7, **kwargs):
    """Shortened version for Stack Exchange"""
    if sep == 0: # fast exit for 0 separations
        # faster than np.unique as latter requires a mask over counts>1
        unique_counts = np.asarray([(v,c) for v,c in Counter(arr).iteritems() if c>1])
        if not unique_counts.size: return np.nan    # handle no unique items
        combinations = binom(unique_counts[:,1],2)  # get combinations
        return ((unique_counts[:,0]-mean)**2*combinations).sum() / combinations.sum()
    # This is for a direct test of teh fucntion in  ../correlations/acorr.py
    # if sep == 0: # fast exit for 0 separations
    #     # faster than np.unique as latter requires a mask over counts>1
    #     uniques,idx,counts = np.unique(arr, return_index=True, return_counts=True)
    #     if not uniques.size: return np.nan    # handle no unique items
    #     combinations = binom(counts,2)        # get combinations
    #     return ((arr[idx]-mean)**2*combinations).sum() / combinations.sum()
    
    front = 1   # front "pythony-pointer-thing"
    back  = 0   # back "pythony-pointer-thing"
    bssp  = 0   # back sweep start point
    bsfp  = 0   # back sweep finish point
    ans   = 0.0 # store the answer
    count = 0   # counter for averaging
    new_front = True # the first front value is new
    while front < n:            # keep going until exhausted array
        new_front = (arr[front]-arr[front-1]>tol)  # check if front value is a new one
        back = bsfp if new_front else bssp         # this is the magical step
        
        diff = arr[front] - arr[back]
        if abs(diff - sep) < tol: # if equal subject to tol: pair found
            if new_front:
                bssp  = bsfp    # move sweep start point
                back  = bsfp    # and back to last front point
                bsfp  = front   # send start end point to front's position
            else:
                back  = bssp    # reset back to the sweep start point
            while back < bsfp:  # calculate the correlation function for matched pairs
                count+= 1
                ans  += (arr[front] - mean)*(arr[back] - mean)
                back += 1
        else:
            if abs(arr[bssp+1]- arr[bssp]) > tol: bsfp = front
            
        front +=1
    return ans/float(count) if count > 0 else np.nan # cannot calculate if no pairs
