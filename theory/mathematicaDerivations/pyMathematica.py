import subprocess
from sys import exit

def eval(expr, link = './runMath'):
    """evaluates Mathematica and returns as string
    
    Required Input
        expr :: string :: Valid Mathematica Expression
    
    Optional Input
        link :: string :: links to script will evaluate and print expr
    """
    try:
        res = subprocess.check_output([link, expr])
        return [r for r in res.split('\n') if r]
    except subprocess.CalledProcessError, e:
        print "Mathematica stdout output:\n", e.output
        sys.exit(1)