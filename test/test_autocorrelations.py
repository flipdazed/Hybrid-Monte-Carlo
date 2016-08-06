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
LOC_AC = "unittestLT_iLT.csv"
INF  = 1e5
ZERO = 1e-5
TOL  = 1e-4

pd.options.display.float_format = '{:10,.4f}'.format

def compareVals(df, columns_to_compare, tol):
    """Compares all values of columns from a given dataframe
    in the context of this unit test
    
    Required Inputs
        df :: pd.DataFrame :: pandas dataframe with the data
        columns_to_compare :: list :: list of column names
        tol :: float :: tolerance
    """
    passed = True
    df[df<ZERO] = 0
    df[df>INF] = np.nan
    df = df.fillna(0)  # replace nans
    
    all_combinations = itertools.combinations(columns_to_compare, 2)
    all_combinations = list(all_combinations)
    
    failed = []
    errors = []
    for a,b in all_combinations:
        diff = df[a] - df[b]
        test = (diff <= tol).all()
        passed *= test
        if not test:
            if a not in failed: 
                failed.append([a, b])
                errors.append([diff.argmax(), diff.max()])
    return passed, failed, errors

class Fixed(object):
    """Runs tests for fixed autocorrelation theory
    
    Required Inputs
        tol :: float :: tolerance for tests
    """
    def __init__(self, tol=TOL):
        self.id  = 'Fixed Trajectory Lengths'
        self.tol = tol
        self.m2fix = theory.M2_Fix()
    
    def laplaceTransformAutocorrelations(self, data_loc=LOC_DAT + LOC_FIX + LOC_LT, print_out=True):
        """Compare the laplace transformed autocorrelation calculations
        Python vs. C++ vs. Mathematica derivations
        
        Required Inputs
            data_loc :: int :: LF trajectory lengths
        
        Optional Inputs
            print_out :: bool :: prints output to screen
        """
        passed = True
        
        # open the autocorrelation test data
        df = pd.read_csv(data_loc)
        df['ghmcLt cpp'] = df.apply(lambda r: ctheory.fixed.ghmcLt(
            r['beta'], r['p_acc'], r['phi'], 1./r['tau'], r['theta']), axis=1)
        df['lapTfm python'] = df.apply(lambda r: self.m2fix.lapTfm(
            b=r['beta'], tau=r['tau'], m=r['phi']/r['tau'], pa=r['p_acc'], theta=r['theta']), axis=1)
        
        columns_to_compare = ["ghmcLt[... , 1/tau]",
                                "ghmcLt cpp",
                                "lapTfm python"]
        
        passed, failed, errors = compareVals(df, columns_to_compare, self.tol)
        
        failed_df = list(set([item for l in failed for item in l]))
        failed = [str(item)+' max err:{}, rw:{}'.format(err, r) for item, (r,err) in zip(failed, errors)]
        inputs = [u'beta', u'phi', u'theta', u'p_acc', u'tau']
        failed_df = df[inputs + failed_df]
        if print_out:
            utils.display(self.id+': integrated autocorrelations', passed,
                details = {
                    'Inputs':columns_to_compare,
                    'Failed Comparisons':failed,
                    'Failures':failed_df
                    })
        
        return passed
    
    def integratedAutocorrelations(self, data_loc=LOC_DAT + LOC_FIX + LOC_IN, print_out=True):
        """Compare the integrated autocorrelation calculations
        Python vs. C++ vs. Mathematica derivations
        
        Required Inputs
            data_loc :: int :: LF trajectory lengths
        
        Optional Inputs
            print_out :: bool :: prints output to screen
        """
        passed = True
        
        # open the autocorrelation test data
        df = pd.read_csv(data_loc)
        df['ighmc cpp'] = df.apply(lambda r: ctheory.fixed.ighmc(r['p_acc'], r['phi'], r['theta']), axis=1)
        df['ighmc python'] = df.apply(lambda r: self.m2fix.integrated(
            tau=r['tau'], m=r['phi']/r['tau'], pa=r['p_acc'], theta=r['theta']), axis=1)
        
        columns_to_compare = ["A[phi, theta, p_acc]",
                                "ghmcLt[0, ... , 1/tau]",
                                "ighmc cpp",
                                "ighmc python"]
        
        passed, failed, errors = compareVals(df, columns_to_compare, self.tol)
        
        failed_df = list(set([item for l in failed for item in l]))
        failed = [str(item)+' max err:{}, rw:{}'.format(err, r) for item, (r,err) in zip(failed, errors)]
        inputs = ['phi', 'theta', 'p_acc', 'tau']
        failed_df = df[inputs + failed_df]
        if print_out:
            utils.display(self.id+': integrated autocorrelations', passed,
                details = {
                    'Inputs':columns_to_compare,
                    'Failed Comparisons':failed,
                    'Failures':failed_df
                    })
        
        return passed

class Exp(object):
    """Runs tests for exponential autocorrelation theory
    
    Required Inputs
        tol :: float :: tolerance for tests
    """
    def __init__(self, tol=TOL, verbose = False):
        self.id  = 'Fixed Trajectory Lengths'
        self.tol = tol
        self.m2exp = theory.M2_Exp()
        
        
    def autocorrelations(self, data_loc=LOC_DAT + LOC_EXP + LOC_AC, print_out=True):
        """Compare the autocorrelation calculations
        Python vs. C++ vs. Mathematica derivations
        
        Required Inputs
            data_loc :: int :: LF trajectory lengths
        
        Optional Inputs
            print_out :: bool :: prints output to screen
        """
        passed = True
        
        # open the autocorrelation test data
        df = pd.read_csv(data_loc)
        # df['ghmc cpp'] = df.apply(lambda r: ctheory.exponential.ghmc(
        #     r['t'], r['p_acc'], r['phi'], 1./r['tau'], r['theta']), axis=1)
        df['ghmc python'] = df.apply(lambda r: self.m2exp.eval(
            t=r['t'], tau=r['tau'], m=r['phi']/r['tau'], pa=r['p_acc'], theta=r['theta']), axis=1)
        
        columns_to_compare = ['invL[ghmcLt[... , 1/tau]](t)', 
                                'invL[ghmcLtDerived[... , tau]](t)',
                                # "ghmc cpp",
                                "ghmc python"]
        
        passed, failed, errors = compareVals(df, columns_to_compare, self.tol)
        
        failed_df = list(set([item for l in failed for item in l]))
        failed = [str(item)+' max err:{}, rw:{}'.format(err, r) for item, (r,err) in zip(failed, errors)]
        inputs = [u't', u'phi', u'theta', u'p_acc', u'tau']
        failed_df = df[inputs + failed_df]
        if print_out:
            utils.display(self.id+': integrated autocorrelations', passed,
                details = {
                    'Inputs':columns_to_compare,
                    'Failed Comparisons':failed,
                    'Failures':failed_df
                    })
        
        return passed
    
    def laplaceTransformAutocorrelations(self, data_loc=LOC_DAT + LOC_EXP + LOC_LT, print_out=True):
        """Compare the laplace transformed autocorrelation calculations
        Python vs. C++ vs. Mathematica derivations
        
        Required Inputs
            data_loc :: int :: LF trajectory lengths
        
        Optional Inputs
            print_out :: bool :: prints output to screen
        """
        passed = True
        
        # open the autocorrelation test data
        df = pd.read_csv(data_loc)
        df['ghmcLt cpp'] = df.apply(lambda r: ctheory.exponential.ghmcLt(
            r['beta'], r['p_acc'], r['phi'], 1./r['tau'], r['theta']), axis=1)
        df['lapTfm python'] = df.apply(lambda r: self.m2exp.lapTfm(
            b=r['beta'], tau=r['tau'], m=r['phi']/r['tau'], pa=r['p_acc'], theta=r['theta']), axis=1)
        
        columns_to_compare = ["ghmcLt[... , 1/tau]",
                                "ghmcLt cpp",
                                "lapTfm python"]
        
        passed, failed, errors = compareVals(df, columns_to_compare, self.tol)
        
        failed_df = list(set([item for l in failed for item in l]))
        failed = [str(item)+' max err:{}, rw:{}'.format(err, r) for item, (r,err) in zip(failed, errors)]
        inputs = [u'beta', u'phi', u'theta', u'p_acc', u'tau']
        failed_df = df[inputs + failed_df]
        if print_out:
            utils.display(self.id+': integrated autocorrelations', passed,
                details = {
                    'Inputs':columns_to_compare,
                    'Failed Comparisons':failed,
                    'Failures':failed_df
                    })
        
        return passed
    
    def integratedAutocorrelations(self, data_loc=LOC_DAT + LOC_EXP + LOC_IN, print_out=True):
        """Compare the integrated autocorrelation calculations
        Python vs. C++ vs. Mathematica derivations
        
        Required Inputs
            data_loc :: int :: LF trajectory lengths
        
        Optional Inputs
            print_out :: bool :: prints output to screen
        """
        passed = True
        
        # open the autocorrelation test data
        df = pd.read_csv(data_loc)
        df['ighmc cpp'] = df.apply(lambda r: ctheory.exponential.ighmc(r['p_acc'], r['phi'], r['theta']), axis=1)
        df['ighmc python'] = df.apply(lambda r: self.m2exp.integrated(
            tau=r['tau'], m=r['phi']/r['tau'], pa=r['p_acc'], theta=r['theta']), axis=1)
        
        columns_to_compare = ["A[phi, theta, p_acc]",
                                "ghmcLt[0, ... , 1/tau]",
                                "ighmc cpp",
                                "ighmc python"]
        
        passed, failed, errors = compareVals(df, columns_to_compare, self.tol)
        
        failed_df = list(set([item for l in failed for item in l]))
        failed = [str(item)+' max err:{}, rw:{}'.format(err, r) for item, (r,err) in zip(failed, errors)]
        inputs = ['phi', 'theta', 'p_acc', 'tau']
        failed_df = df[inputs + failed_df]
        if print_out:
            utils.display(self.id+': integrated autocorrelations', passed,
                details = {
                    'Inputs':columns_to_compare,
                    'Failed Comparisons':failed,
                    'Failures':failed_df
                    })
        
        return passed

if __name__ == '__main__':
    
    fixed = Fixed()
    fixed.integratedAutocorrelations()
    fixed.laplaceTransformAutocorrelations()
    
    exp = Exp()
    exp.integratedAutocorrelations()
    exp.laplaceTransformAutocorrelations()
    exp.autocorrelations()