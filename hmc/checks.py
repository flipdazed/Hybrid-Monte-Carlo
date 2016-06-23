import numpy as np
import sys
import traceback
from inspect import getframeinfo, stack

def debuginfo(message=None):
    caller = getframeinfo(stack()[1][0])
    print "%s:%d - %s" % (caller.filename, caller.lineno, message)
#
def fullTrace():
    """Prints the full traceback if there is an error
    
    Usage:
        try:
            p !&*&$)%) this is an error
        except Exception, e:
            fullTrace("ARARG")
    """
    print '\nERROR...'
    _, _, tb = sys.exc_info()
    traceback.print_tb(tb) # Fixed format
    tb_info = traceback.extract_tb(tb)
    filename, line, func, text = tb_info[-1]
    print 'line {} in {}'.format(line, text)
    caller = getframeinfo(stack()[1][0])
    print 'Caller Details: {} : {} in {}\n'.format(caller.lineno, caller.function, caller.filename)
    pass
#
def tryAssertEqual(val1, val2, error_msg):
    """asserts that two values are equal
    
    Required Inputs
        val1        :: anything :: the first value to compare
        val2        :: anything :: the second value to compare
        error_msg   :: string   :: message if error
    
    will check that val1 == val2
    """
    try:
        check = (val1 == val2)
        if hasattr(check, 'all'): check = check.all() # for np arrays
        assert check
    except Exception, e:
        # _, _, tb = sys.exc_info()
        # traceback.print_tb(tb) # Fixed format
        # tb_info = traceback.extract_tb(tb)
        # filename, line, func, text = tb_info[-1]
        # print 'line {} in {}'.format(line, text)
        caller = getframeinfo(stack()[1][0])
        print '\nError: Failed equal assertion...'
        print '> {} : {} ~ {}\n'.format(caller.lineno, caller.filename, caller.function)
        print error_msg
        sys.exit(1)
    pass

#
def tryAssertLtEqual(lessthan, val2, error_msg):
    """asserts that two values are equal
    
    Required Inputs
        val1        :: anything :: the first value to compare
        val2        :: anything :: the second value to compare
        error_msg   :: string   :: message if error
    
    will check that val1 == val2
    """
    try:
        assert lessthan <= val2
    except Exception, e:
        # _, _, tb = sys.exc_info()
        # traceback.print_tb(tb) # Fixed format
        # tb_info = traceback.extract_tb(tb)
        # filename, line, func, text = tb_info[-1]
        # print 'line {} in {}'.format(line, text)
        caller = getframeinfo(stack()[1][0])
        print '\nError: Failed equal assertion...'
        print '> {} : {} ~ {}\n'.format(caller.lineno, caller.filename, caller.function)
        print error_msg
        sys.exit(1)
    pass