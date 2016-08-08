import numpy as np
import sys, traceback
from scipy.special import binom
from collections import Counter

def testFn(results, test, res_pairs=None):
    """Takes in two arrays of your function results and compares to test data
    If you pass the pairs you found to res_pairs it will also neatly display those
    
    Required Inputs
        results :: list :: list of results from test function
        test :: list :: list of test cases to compare against
    
    Optional Inputs
        res_pairs :: list of lists/np.arrays :: list of pairs that were found
    """
    outcomes = ["Failed","Passed"]
    
    print "\nTest outcomes..."
    if res_pairs is None: res_pairs = len(test)*[None]
    for i,(r,t, pairs) in enumerate(zip(results, test, res_pairs)):
        try:
            np.testing.assert_almost_equal(r,t)
            passed = True
        except:
            passed = False
        pr = "  test:{} :: {} :: res: {:7.4f} actual: {:7.4f}".format(i+1, outcomes[passed], r, t)
        if pairs is not None: pr += " pairs: "+" ".join(["{:d}x({:3.1f} {:3.1f})".format(n,i,j) for (i,j),n in Counter(tuple(p) for p in pairs).iteritems()])
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
    e = np.array([0.4]*3 + [0.5]*3 + [0.6]*3 + [5]) # series of identicals
    f = np.asarray([0, 0.2, 0.4, 0.6, 0.8] + [0.9]*3 + [1.0]*2) # no match then matches
    
    
    # a quick function used a fair bit in the case of equal incrementation
    equalSpacing = lambda seg, mean, sep: np.sum((seg[:seg.size-sep]-mean)*(seg[sep:]-mean))
    nCr52 = binom(5,2)  # ways of choosing n from r where order matters
    nCr32 = binom(3,2)
    dm = d.mean()      # both means used a lot so declaring saves space
    em = e.mean()
    fm = f.mean()
    
    # the test cases for 0.1 separation
    sep = 0.1
    t1a = np.nan
    t1b = equalSpacing(b, b.mean(), 1)/float(n-1)
    t1c = (0.1-c.mean())*(0.2-c.mean()) #*5**2/5**2
    t1d = (equalSpacing(d[:3], dm, 1) + (.3-dm)*(.4-dm)*5. + 5.*2*(.4-dm)*(.5-dm))/(2.+5.+2.*5.)
    t1e = ((0.4-em)*(0.5-em)*3.*3. + (0.5-em)*(0.6-em)*3*3)/(3.*3.+3.*3)
    t1f = ((0.8-fm)*(0.9-fm)*3. + (0.9-fm)*(1.0-fm)*2.*3.)/(3.+2.*3.)
    
    # test caess for 0 separation
    sep = 0.0
    t2a = 0.0
    t2b = np.nan
    t2c = ((0.1-c.mean())**2*nCr52 + (0.2-c.mean())**2*nCr52)/(nCr52+nCr52)
    t2d = ((0.4-dm)**2*nCr52 + (.5-dm)**2)/(nCr52+1)
    t2e = ((0.4-em)**2*nCr32 + (.5-em)**2*nCr32 + (0.6-em)**2*nCr32)/(3*nCr32)
    t2f = ((0.9-fm)**2*nCr32 + (1.0-fm)**2)/(nCr32+1)
    
    cases = [a,b,c,d,e, f]
    test_set1 = [t1a, t1b, t1c, t1d, t1e, t1f]
    test_set2 = [t2a, t2b, t2c, t2d, t2e, t2f]
    return cases, test_set1, test_set2

def debugRoutine(func, verbose=False, debug=False):
    """include **kwargs with func if it doesn't have same as mine
    
    Required Inputs
        func :: func :: accepts at least sep, arr others can be absorbed as **kwargs
                        will also accept n, mean, debug, verbose
    Optional Inputs
        verbose :: bool :: print out optional text - passed as kwarg to func
        debug :: bool :: expect func will output (solution, pairs) enables printing of pairs found
    """
    seps = [0.1, 0.0] # the separations to test
    cases, test_set1, test_set2 = genTestData() # generate the test cases
    
    for test, sep in zip([test_set1, test_set2], seps): # loop over separations
        print '\nStart separation of {}'.format(sep)
        res = []
        if debug: res_pairs = []    # if debug enabled then output pairs
        else: res_pairs = None
        for i, arr in enumerate(cases): # loop test cases
            if verbose: print "\nRunning: test {}".format(i+1)
            try:                    # allow error handling
                mean = arr.mean()   # get mean and run function
                out = func(arr, sep=sep, mean=mean, n=arr.size, debug=debug, verbose=verbose)
                if debug:
                    sol, pairs = out    # if debug then expand tuple to grab pairs and solution
                    res_pairs.append(pairs)
                else: sol = out
                res.append(sol)         # append solutions to list
            
            except Exception as e:      # catch any errors encountered
                if debug: res_pairs.append([])
                res.append(False)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                err = traceback.format_exc().splitlines()[-1]
                line = traceback.extract_tb(exc_traceback)[-1][1]
                expr = traceback.extract_tb(exc_traceback)[-1][-1]
                print '   > Error: test {}, line: {}, type: {}, expr: {}'.format(i+1, line, err, expr)
        # run the test case comparison function for each separation
        testFn(res, test, res_pairs)
    pass

if __name__ == "__main__":
    from sweeper import attempt, attemptShort
    
    debugRoutine(attempt, verbose=False, debug=True)
    debugRoutine(attemptShort)
    

# # This can be run from the main directory of the repository in Ipython!

# from plotter.pretty_plotting import Pretty_Plotter
# pp._teXify()
# pp.params['text.latex.preamble'] = [r"\usepackage{amsmath}"]
# pp._updateRC()
#
# fig = plt.figure(figsize=(8, 8)) # make plot
# ax =[]
# ax.append(fig.add_subplot(111))
#
# ax[0].set_ylabel(r'Averaged Time, $s$')
# ax[0].set_xlabel(r'Length of Sorted Array, $n$')
#
# x = [5, 10, 50, 100, 500, 1000,5000, 10000, 50000, 100000, 500000, 1000000]
# seps = [0.1, 0.2, 0.5]
#
# all_res = []
# for s in seps:
#     res = []
#     for i in x:
#         arr = np.cumsum(np.around(np.random.geometric(diff, n),1))
#         mean = arr.mean()
#         a = %timeit -o attempt(arr, s, mean, i) # can only be run in ipython
#         res.append(np.mean(a))
#         ax[0].plot(x, res, linewidth=2.0, alpha=0.6, label='separation = ${:3.1f}$; $ds/dn = {:5.3f}$'.format(s, np.mean(np.gradient(res[3:]))))
# ax[0].legend(loc='best', shadow=True, fontsize = pp.ipfont, fancybox=True)
# fig.suptitle('Showing scaling behaviour of $\mathcal{O}(n)$', fontsize=16)