import numpy as np

from contextlib import contextmanager
from StringIO import StringIO

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def display(test_name, outcome, details = False, minimal=False):
    """
    Require Inputs
        test_name   :: string   :: name of test
        outcome     :: bool     :: True = passed test
    
    Optional Inputs
        details     :: dict     :: {'detail':['subdetail1', 'subdetail2], 'detail2'}
        minimal     :: Bool     :: prints minimal results
    """
    print '\n\n TEST: {}'.format(test_name)
    
    if details and (not minimal):
        np.set_printoptions(precision=2, suppress=True)
        for detail, sub_details in details.iteritems():
            print '\t ' + detail
            for sub_detail in sub_details:
                print '\t  ... ' + sub_detail
    
    print ' OUTCOME: {}'.format(['Failed','Passed'][outcome])
    pass
    
if __name__ == '__main__':
    
    display(
        test_name = 'Test of a Test',
        outcome = True,
        details = {
            'a detail ({}, {})'.format(1, 1):
                ['a sub detail of: '+'a detail ({}, {})'.format(1, 1)],
            'more details':[]
            }
        ,
        minimal = False)