# -*- coding: utf-8 -*- 
import numpy as np
import pandas as pd
import itertools

import utils
import theory.clibs.autocorrelations as ctheory
import theory.autocorrelations as theory

LOC_DAT = "theory/mathematicaDerivations/"
LOC_EXP = "exponential/"
LOC_FIX = "fixed/"
LOC_LT  = "unittestLT_LT.csv"
LOC_IN = "unittestLT_iAC.csv"
INF  = 1e7
ZERO = 1e-7

def compareVals(columns_to_compare, tol):
    """Compares all values of columns from a given dataframe
    in the context of this unit test
    
    Required Inputs
        columns_to_compare :: list :: list of column names
        tol :: float :: tolerance
    """
    all_combinations = itertools.combinations(columns_to_compare, 2)
    all_combinations = list(all_combinations)
    
    failed_df = []
    for a,b in all_combinations:
        test = (df[a] - df[b] <= tol).all()
        passed *= test
        if not passed: 
            if a not in failed_df: failed_df.append(a)
            if b not in failed_df: failed_df.append(b)
    return failed_df

class Fixed(object):
    """Runs tests for autocorrelation theory
    
    Required Inputs
        rng :: np.random.RandomState :: must be able to call rng.uniform
    
    Optional Inputs
        length :: int :: 1d lattice length
        dim :: int  :: number of dimensions
        spacing :: float :: lattice spacing
    """
    def __init__(self, tol=ZERO, verbose = False):
        self.id  = 'Fixed Trajectory Lengths'
        self.tol = tol
        
    def integratedAutocorrelations(self, data_loc=LOC_DAT + LOC_FIX + LOC_IN, print_out=True):
        """Compare the integrated autocorrelation calculations
        Python vs. C++ vs. Mathematica derivations
        
        Required Inputs
            data_loc :: int :: LF trajectory lengths
        """
        passed = True
        
        m2fix = theory.M2_Fix()
        # open the autocorrelation test data
        pd.options.display.float_format = '{:10,.4f}'.format
        df = pd.read_csv(data_loc)
        df['ighmc cpp'] = df.apply(lambda row: ctheory.fixed.ighmc(row['p_acc'], row['phi'], row['theta']), axis=1)
        df['ighmc python'] = df.apply(lambda row: m2fix.integrated(
            tau=row['tau'], m=row['phi']/row['tau'], pa=row['p_acc'], theta=row['theta']), axis=1)
        df[df<ZERO] = 0
        df[df>INF] = np.nan
        df.fillna(0)  # replace nans
        
        columns_to_compare = ["A[phi, theta, p_acc]",
                                "ghmcLt[0, ... , 1/tau]",
                                "ighmc cpp",
                                "ighmc python"]
        
        failed_df = compareVals(columns_to_compare, self.tol)
        
        inputs = ['phi', 'theta', 'p_acc', 'tau']
        failed_df = df[inputs + failed_df].index(false)
        if print_out:
            utils.display(self.id+': integrated autocorrelations', passed,
                details = {
                    'Inputs':[str(item[0]) +"\tvs.\t"+ str(item[1]) for item in all_combinations],
                    'Failures':failed_df
                    })
        
        return passed

if __name__ == '__main__':
    
    fixed = Fixed()
    fixed.integratedAutocorrelations()