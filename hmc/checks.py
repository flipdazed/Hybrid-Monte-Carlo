import numpy as np
import sys
# import traceback
from inspect import getframeinfo, stack

def debuginfo(message):
    caller = getframeinfo(stack()[1][0])
    print "%s:%d - %s" % (caller.filename, caller.lineno, message)

def tryAssertEqual(val1, val2, error_msg):
    """asserts that two values are equal
    
    Required Inputs
        val1        :: anything :: the first value to compare
        val2        :: anything :: the second value to compare
        error_msg   :: string   :: message if error
    
    will check that val1 == val2
    """
    try:
        assert val1 == val2
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