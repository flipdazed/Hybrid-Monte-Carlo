import numpy as np
import logging
from logs import *

logging.root.setLevel(logging.INFO)

def newTest(test_id):
    logger.newtest('')
    logger.newtest('>>> Testing: ' + test_id)
    pass

def display(test_name, outcome, details = False, minimal=False):
    """
    Require Inputs
        test_name   :: string   :: name of test
        outcome     :: bool     :: True = passed test
    
    Optional Inputs
        details     :: dict     :: {'detail':['subdetail1', 'subdetail2], 'detail2'}
        minimal     :: Bool     :: prints minimal results
    """
    if outcome: 
        overview = logger.info
        extra = logger.debug
    else:
        overview = logger.error
        extra = logger.warn
    
    overview('')
    overview(' {}'.format(test_name))
    
    if details and (not minimal):
        np.set_printoptions(precision=2, suppress=True)
        for detail, sub_details in details.iteritems():
            extra('   ' + detail)
            for sub_detail in sub_details:
                extra('    ... ' + sub_detail)
    
    overview(' OUTCOME: {}'.format(['Failed','Passed'][outcome]))
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